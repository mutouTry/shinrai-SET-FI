"""Tap set, tap -> expression resolution, and the generated SystemVerilog.

The recorder wraps the user's testbench unchanged::

    module setfi_tb_wrapper;
        <tb_top_module> tb();        // the user testbench
        cycle_counter                // +1 at every posedge of tb.<clk> while out of reset
        [SETFI-CYC] first= / last=   // cycles run after reset (last from a `final` block)
        `include "setfi_tb_rec.svh"  // pass 2 only: the recorder
    endmodule

The recorded value of net ``i`` is read through the hierarchical reference
``tb.<dut_inst>.<net path with '/' -> '.'>``.

Two recorder styles produce the same observation -- the settled state at the
falling clock edge of cycle N, after every event of that time step:

``program`` (VCS)
    A SystemVerilog ``program`` samples in the Reactive region and writes
    ``cycle_<N>.hex`` itself.
``strobe`` (Icarus, which runs ``program`` blocks in the Active region)
    Every tap is a 1-bit wire, and at each falling edge whose cycle counter is a
    candidate the wrapper issues
    ``$fstrobe(events, "%0d %b %h", counter, reset_guard, {taps})``.  ``$fstrobe``
    evaluates its arguments in the Postponed region, so both the state and the
    reset guard are the settled values.  Python then replays the ``program``
    recorder's pointer logic over those events (see ``replay_strobe_events``),
    which reproduces its cycle selection exactly, including the reset guard.

Why not ``always @(negedge clk)`` alone: a testbench drives stimulus on the
same falling edge, the order of two Active-region processes is undefined, and a
recorder that runs first reads the previous cycle's stimulus.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .errors import WorkloadError
from .sampling import n_hex_digits

WRAPPER_MODULE = "setfi_tb_wrapper"
RECORDER_PROGRAM = "setfi_recorder"
GENERATOR = "setfi.workload"

RECORDER_STYLES = ("program", "strobe")


# ---------------------------------------------------------------------------
# Tap set
# ---------------------------------------------------------------------------
def load_taps(net_index: Path, ff_index: Optional[Path] = None
              ) -> Tuple[List[str], Optional[Set[str]]]:
    """Return (taps ordered by net index, flip-flop D/Q nets or None).

    The tap order IS the global net numbering, so bit ``i`` of a recorded hex
    is net ``i``.  The flip-flop nets (of every sequential cell, observed or not)
    decide whether a tap that cannot be read is fatal; without ``ff_index`` every
    such tap is.
    """
    net_index = Path(net_index)
    if not net_index.is_file():
        raise WorkloadError("config_invalid", f"net_index not found: {net_index}")
    ni = json.loads(net_index.read_text())
    idx_to_net = {int(k): v for k, v in ni["idx_to_net"].items()}
    missing = [i for i in range(len(idx_to_net)) if i not in idx_to_net]
    if missing:
        raise WorkloadError("config_invalid",
                            f"{net_index}: idx_to_net is not dense 0..N-1 (missing {missing[:5]})")
    taps = [idx_to_net[i] for i in range(len(idx_to_net))]
    if ff_index is None:
        return taps, None
    ff_index = Path(ff_index)
    if not ff_index.is_file():
        raise WorkloadError("config_invalid", f"ff_index not found: {ff_index}")
    return taps, state_nets(json.loads(ff_index.read_text()), idx_to_net)


def state_nets(ff_index: Dict, idx_to_net: Dict[int, str]) -> Set[str]:
    """D and Q nets of every sequential cell (the maps are not reduced to the observed
    flip-flops)."""
    out: Set[str] = set()
    for key in ("dnet_to_ffids", "qnet_to_ffids"):
        for k in ff_index.get(key, {}):
            n = idx_to_net.get(int(k))
            if n:
                out.add(n)
    return out


def verify_xmr_resolves(netlist_text: str, taps: Sequence[str]) -> List[str]:
    """Return the taps whose leaf identifier does not occur in the netlist text.

    Leaf = the last component after '.' or '/', with trailing bus subscripts
    stripped; synthetic concatenation / cast taps are skipped.  A cheap gate
    run before the simulator, which would otherwise fail at elaboration.
    """
    missing: List[str] = []
    for tap in taps:
        if "{" in tap or "}" in tap or "(" in tap:
            continue
        leaf = re.split(r"[./]", tap)[-1]
        leaf_no_bus = re.sub(r"(\[[^\]]+\])+$", "", leaf) or tap
        if not re.search(r"\b" + re.escape(leaf_no_bus) + r"\b", netlist_text):
            missing.append(tap)
    return missing


# ---------------------------------------------------------------------------
# Tap -> SystemVerilog expression
# ---------------------------------------------------------------------------
_SLICE_BIT_RE = re.compile(r"^(?P<base>.+?)\[(?P<hi>\d+):(?P<lo>\d+)\]\[(?P<k>\d+)\]$")
# The concatenation may be scoped by an instance path (`u_dut/{a, b}[k]`).
_CONCAT_BIT_RE = re.compile(
    r"^(?P<pfx>[A-Za-z_][A-Za-z0-9_$./\\]*/)?\{(?P<body>.*)\}\[(?P<k>\d+)\]$", re.S)
_PLAIN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_BIT_SELECT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_$./]*)\[\d+\]$")
_CHAINED_SELECT_RE = re.compile(r"\]\s*\[")
# Literal constants surface as taps when an assign ties a net to 1'b0 / 1'b1.
_LITERAL_CONST_RE = re.compile(r"^\s*(?:\d*'[bB][01xXzZ?]|0|1)\s*$")
# A bare clock-gate output (`.Q(ENCLK)`) lives inside the gating cell, not at
# the DUT scope, and the engine models the clock by its capture edge, never by
# this value: a don't-care, recorded as 0 to keep the bit layout.
_GATED_CLOCK_LEAF_RE = re.compile(r"^ENCLK$")


@dataclass(frozen=True)
class TapBinding:
    """How net ``idx`` is read: ``expr`` is a complete SV expression in the
    wrapper scope (``tb.<dut>.<path>``, a literal, or ``1'b0``)."""
    idx: int
    tap: str
    expr: str
    kind: str          # "xmr" | "resolved" | "literal" | "gated_clock" | "unresolved"
    note: str = ""


def resolve_taps(
    taps: Sequence[str], tb_dut_inst: str,
    ff_nets: Optional[Set[str]] = None,
) -> Tuple[List[TapBinding], Dict]:
    """Bind every tap to an SV expression.

    Per-bit names of a bus SLICE or CONCATENATION (as a netlist analysis names
    bit k of a port connection) are resolved with slice/concat semantics:
    ``BASE[hi:lo][k] -> BASE[lo+k]`` (descending slices only) and
    ``{a, ..., z}[k]`` -> the k-th element from the right, when every element
    is provably one bit.  Anything else stays unresolved and is recorded as
    ``1'b0``; if it carries flip-flop state that is fatal
    (``unresolvable_state_tap``), because a fabricated 0 for a flip-flop's
    state corrupts everything downstream.
    """
    bus_bases = {m.group(1) for m in
                 (re.match(r"^(.+?)\[\d+(?::\d+)?\]$", x) for x in taps) if m}

    def elem_is_one_bit(e: str, pfx: str = "") -> bool:
        if _BIT_SELECT_RE.match(e):
            return True
        if _PLAIN_IDENT_RE.match(e):
            # scalar only if never bit-selected, checked in the concat's own scope
            return (pfx + e) not in bus_bases and e not in bus_bases
        return False

    def resolve_one(tap: str) -> Optional[str]:
        m = _SLICE_BIT_RE.match(tap)
        if m:
            hi, lo, k = int(m["hi"]), int(m["lo"]), int(m["k"])
            if hi >= lo and k <= hi - lo:
                return f"{m['base']}[{lo + k}]"
            return None
        m = _CONCAT_BIT_RE.match(tap)
        if m:
            pfx = m["pfx"] or ""
            parts = [x.strip() for x in m["body"].replace("\n", " ").split(",")]
            parts = [x for x in parts if x]
            if parts and all(elem_is_one_bit(x, pfx) for x in parts):
                k = int(m["k"])
                if k < len(parts):
                    return pfx + parts[len(parts) - 1 - k]
            return None
        return None

    unresolved_state: List[str] = []
    n_unresolved_other = 0
    bindings: List[TapBinding] = []
    for idx, tap in enumerate(taps):
        is_synth = "{" in tap or "}" in tap or "(" in tap
        if is_synth or _CHAINED_SELECT_RE.search(tap):
            res = resolve_one(tap)
            if res is not None:
                bindings.append(TapBinding(
                    idx, tap, f"tb.{tb_dut_inst}.{res.replace('/', '.')}", "resolved",
                    f"resolved from {tap.replace(chr(10), ' ')[:60]!r}"))
                continue
            if ff_nets is None or tap in ff_nets:
                unresolved_state.append(repr(tap[:70]))
            else:
                n_unresolved_other += 1
            why = "synthetic-concat" if is_synth else "chained-part-select"
            bindings.append(TapBinding(idx, tap, "1'b0", "unresolved",
                                       f"UNRESOLVED {why}: {tap!r}"))
            continue
        if _LITERAL_CONST_RE.match(tap):
            bindings.append(TapBinding(idx, tap, tap.strip(), "literal",
                                       f"literal-const: {tap!r}"))
            continue
        if "/" not in tap and "." not in tap and _GATED_CLOCK_LEAF_RE.match(tap):
            bindings.append(TapBinding(idx, tap, "1'b0", "gated_clock",
                                       f"gated-clock output, not needed: {tap!r}"))
            continue
        bindings.append(TapBinding(idx, tap, f"tb.{tb_dut_inst}.{tap.replace('/', '.')}", "xmr"))

    if unresolved_state:
        raise WorkloadError(
            "unresolvable_state_tap",
            f"{len(unresolved_state)} tap(s) carrying (or not provably free of) FLIP-FLOP "
            f"STATE could not be resolved to a readable net, so their recorded value "
            f"would be a fabricated 0. Offenders (first 10): "
            + "; ".join(unresolved_state[:10])
            + (f" ... and {len(unresolved_state) - 10} more" if len(unresolved_state) > 10 else "")
            + ("" if ff_nets is not None else
               " (no ff_index given: pass one so non-state taps can be told apart)"),
        )
    return bindings, {"n_unresolved_taps": n_unresolved_other,
                      "n_unresolved_state_taps": 0}


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
def rst_guard_expr(rst_net: str, polarity: str) -> str:
    if not rst_net:
        return "1'b1"                  # no reset: every cycle counts
    """Expression (wrapper scope) that is 1 while the design is out of reset."""
    return f"tb.{rst_net}" if polarity == "active_low" else f"!tb.{rst_net}"


def input_timing_monitor(dut_inst: str, clk_net: str, guard: str,
                         inputs: Sequence[str]) -> str:
    """Verilog that reports a design input changing strictly between a falling and the
    next rising clock edge, in a half cycle that starts out of reset.  Every input
    has its own timestamp, so a change exactly at an edge (the clock's own, or a
    stimulus driven at an edge) never hides or fakes a late one."""
    if not inputs:
        return ""
    lines = ["",
             "    // ---- design inputs must not change between a falling and the next rising edge ----",
             "    // (times in ns, the unit of this module)",
             "    realtime setfi_t_neg;",
             "    reg setfi_guard_neg, setfi_late_seen;",
             "    initial begin setfi_t_neg = 0; setfi_guard_neg = 1'b0; setfi_late_seen = 1'b0; end",
             f"    always @(negedge tb.{clk_net}) begin setfi_t_neg = $realtime; "
             f"setfi_guard_neg = ({guard}); end"]
    for k, p in enumerate(inputs):
        lines.append(f"    realtime setfi_tc_{k}; initial setfi_tc_{k} = 0;")
        lines.append(f"    always @(tb.{dut_inst}.{p}) setfi_tc_{k} = $realtime;")
    lines.append(f"    always @(posedge tb.{clk_net}) if (setfi_guard_neg && !setfi_late_seen) begin")
    for k, p in enumerate(inputs):
        lines.append(f"        if (!setfi_late_seen && setfi_tc_{k} > setfi_t_neg && setfi_tc_{k} < $realtime) begin")
        lines.append(f'            $display("[SETFI-LATE-INPUT] t_ps=%0d port={p}", '
                     f"$rtoi(setfi_tc_{k} * 1000.0 + 0.5));")
        lines.append("            setfi_late_seen = 1'b1;")
        lines.append("        end")
    lines.append("    end")
    return "\n".join(lines) + "\n"


def emit_wrapper(
    out_path: Path, *, tb_top_module: str, clk_net: str, rst_net: str,
    rst_polarity: str, include_path: Optional[Path], version: str,
    monitor: str = "",
) -> None:
    guard = rst_guard_expr(rst_net, rst_polarity)
    body = f"""\
// AUTO-GENERATED by {GENERATOR} v{version}
// Recorder wrapper: instantiates the user testbench unchanged and observes it
// through hierarchical references.
`timescale 1ns/1ps

module {WRAPPER_MODULE};
    // Instantiate user testbench; clk/rst/everything stays inside it.
    {tb_top_module} tb();

    // ---- cycle_counter (lives at wrapper level, accessible from
    //      tb_rec.svh as cycle_counter via Verilog scope rules) ----
    reg [31:0] cycle_counter;
    initial cycle_counter = 32'd0;
    // Sample on posedge of user TB's clk via XMR
    always @(posedge tb.{clk_net}) begin
        if ({guard})
            cycle_counter <= cycle_counter + 32'd1;
        else
            cycle_counter <= 32'd0;
    end

    // ---- [SETFI-CYC] markers ----
    // first marker: emitted ONCE at the first cycle observed in functional
    // mode (rst deasserted). Robust under repeated reset assertions.
    reg setfi_first_seen;
    initial setfi_first_seen = 1'b0;
    always @(posedge tb.{clk_net}) begin
        if ({guard} && !setfi_first_seen) begin
            $display("[SETFI-CYC] first=%0d", cycle_counter);
            setfi_first_seen <= 1'b1;
        end
    end

    // last marker: emitted from `final` block on $finish. WARNING:
    // `final` does NOT fire on segfault/license-timeout/kill;
    // marker absence is treated as error kind sim_aborted_no_last_marker
    // by the parsing tool.
    final begin
        $display("[SETFI-CYC] last=%0d", cycle_counter);
    end

"""
    body += monitor
    if include_path is not None:
        body += f'    `include "{include_path}"\n'
    body += "endmodule\n"
    Path(out_path).write_text(body)


# ---------------------------------------------------------------------------
# Recorder, style "program" (VCS)
# ---------------------------------------------------------------------------
_REC_HEADER = """\
// AUTO-GENERATED by {generator} v{version}
// Format of the recorded states:
//   - one file per sampled cycle: <setfi_outdir>/cycle_<N>.hex
//   - file content: {n_hex_digits} hex chars (zero-padded) + '\\n'
//                   = {file_size_bytes} bytes per cycle
//   - bit ordering: bit[net_idx] = value; net_idx 0 is LSB (little-endian)
//   - cycle number N is the value of setfi_tb_wrapper.cycle_counter at
//     the negedge of {clk_net} when sampling.
// Tap count: {n_taps} nets (every net of net_index.json).
// Sample cycle count: {n_samples}; recorder style: {style}.
"""


def _rec_header(*, n_taps: int, n_samples: int, clk_net: str, style: str, version: str) -> str:
    n_hex = n_hex_digits(n_taps)
    return _REC_HEADER.format(generator=GENERATOR, version=version, n_hex_digits=n_hex,
                              file_size_bytes=n_hex + 1, clk_net=clk_net, n_taps=n_taps,
                              n_samples=n_samples, style=style)


def emit_recorder_program(
    svh_path: Path, prog_path: Path, *, bindings: Sequence[TapBinding],
    sample_cycles: Sequence[int], clk_net: str, rst_guard: str, version: str,
) -> None:
    """Style "program": declarations + sampling task in the included .svh, the
    Reactive-region sampling process in a sibling compilation-unit file."""
    n_taps = len(bindings)
    n_samples = len(sample_cycles)
    lines: List[str] = [_rec_header(n_taps=n_taps, n_samples=n_samples, clk_net=clk_net,
                                    style="program", version=version)]
    lines.append(f"`define SETFI_N_TAPS    {n_taps}")
    lines.append(f"`define SETFI_N_SAMPLES {n_samples}")
    lines.append("")
    lines.append(f"reg [{n_taps - 1}:0] setfi_net_state;")
    lines.append(f"integer setfi_cyc_list [0:{n_samples - 1}];")
    lines.append("integer setfi_cyc_ptr;")
    lines.append("string  setfi_outdir;")
    lines.append("")
    lines.append("initial begin : setfi_tb_rec_init")
    for i, c in enumerate(sample_cycles):
        lines.append(f"    setfi_cyc_list[{i:5d}] = {c};")
    lines.append("    setfi_cyc_ptr = 0;")
    lines.append('    if (!$value$plusargs("setfi_outdir=%s", setfi_outdir))')
    lines.append('        setfi_outdir = "states";')
    lines.append('    $system({"mkdir -p ", setfi_outdir});')
    lines.append('    $display("[SETFI-REC] outdir=%s n_taps=%0d n_samples=%0d",')
    lines.append("             setfi_outdir, `SETFI_N_TAPS, `SETFI_N_SAMPLES);")
    lines.append("end")
    lines.append("")
    lines.append("task automatic setfi_sample_all_taps;")
    lines.append("    begin")
    for b in bindings:
        tail = f"  // {b.note}" if b.note else ""
        lines.append(f"        setfi_net_state[{b.idx:5d}] = {b.expr};{tail}")
    lines.append("    end")
    lines.append("endtask")
    lines.append("")

    # Everything the program touches is qualified from compilation-unit scope
    # (`tb` is an instance inside the wrapper, not visible from here).
    W = WRAPPER_MODULE
    clk_q = f"{W}.tb.{clk_net}"
    rst_q = re.sub(r"\btb\.", f"{W}.tb.", rst_guard)
    prog = [
        f"// AUTO-GENERATED by {GENERATOR} -- do not edit.",
        "// Reactive-region recorder: runs after every Active-region event of the",
        "// time step, so the observation is settled by construction.",
        f"program {RECORDER_PROGRAM};",
        "    string  _outpath;",
        "    integer _fd;",
        "    initial forever begin",
        f"        @(negedge {clk_q});",
        # The sample count is inlined (the macro lives in the other file).
        f"        if ({rst_q} && {W}.setfi_cyc_ptr < {n_samples}) begin",
        f"            if ({W}.cycle_counter ==",
        f"                {W}.setfi_cyc_list[{W}.setfi_cyc_ptr]) begin",
        f"                {W}.setfi_sample_all_taps();",
        '                $sformat(_outpath, "%s/cycle_%0d.hex",',
        f"                         {W}.setfi_outdir,",
        f"                         {W}.cycle_counter);",
        '                _fd = $fopen(_outpath, "w");',
        "                if (_fd != 0) begin",
        f'                    $fwrite(_fd, "%h\\n", {W}.setfi_net_state);',
        "                    $fclose(_fd);",
        '                    $display("[SETFI-REC] cyc=%0d -> %s",',
        f"                             {W}.cycle_counter, _outpath);",
        "                end else",
        '                    $display("[SETFI-REC] WARN: cannot create %s", _outpath);',
        f"                {W}.setfi_cyc_ptr =",
        f"                    {W}.setfi_cyc_ptr + 1;",
        "            end",
        "        end",
        "    end",
        "endprogram",
    ]
    Path(prog_path).write_text("\n".join(prog) + "\n")
    Path(svh_path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Recorder, style "strobe" (Icarus)
# ---------------------------------------------------------------------------
STROBE_EVENT_RE = re.compile(r"^(\d+) (\S+) (\S+)$")


STROBE_CHUNK_BITS = 64


def emit_recorder_strobe(
    svh_path: Path, *, bindings: Sequence[TapBinding], sample_cycles: Sequence[int],
    clk_net: str, rst_guard: str, version: str,
) -> None:
    """Style "strobe": a Postponed-region ``$fstrobe`` of (counter, reset
    guard, state) at every negedge whose counter is a candidate.

    * Each tap is a 1-bit net declaration assignment, which truncates a
      (hypothetically) multi-bit tap to its LSB exactly as the "program"
      style's per-bit procedural assignment does.
    * Icarus's ``$fstrobe`` takes only plain signals, so the state is passed as
      ``STROBE_CHUNK_BITS``-wide chunk wires (each a concatenation of tap
      wires) printed ``%h%h...%h`` from the top chunk down.  Every chunk but
      the top one is a multiple of 4 bits wide, so the digits are exactly the
      ``%h`` digits of the whole vector, x/X/z/Z included.  (A single wide
      wire would be re-evaluated in full on every tap change: 30x slower on a
      3.8k-net design.)
    * The guard is ``(<reset guard>) && 1'b1``: the logical truth value the
      "program" recorder's ``if (<guard> && ...)`` tests.
    """
    n_taps = len(bindings)
    n_samples = len(sample_cycles)
    if [b.idx for b in bindings] != list(range(n_taps)):
        raise ValueError("bindings must be ordered by net index 0..N-1")
    lines: List[str] = [_rec_header(n_taps=n_taps, n_samples=n_samples, clk_net=clk_net,
                                    style="strobe", version=version)]
    lines.append(f"`define SETFI_N_TAPS    {n_taps}")
    lines.append(f"`define SETFI_N_SAMPLES {n_samples}")
    lines.append("")
    lines.append(f"integer setfi_cyc_list [0:{n_samples - 1}];")
    lines.append("integer setfi_cyc_k;")
    lines.append("integer setfi_ev_fd;")
    lines.append("string  setfi_events;")
    lines.append(f"wire setfi_guard = ({rst_guard}) && 1'b1;")
    lines.append("")
    for b in bindings:
        tail = f"  // {b.note}" if b.note else ""
        lines.append(f"wire setfi_t{b.idx} = {b.expr};{tail}")
    lines.append("")
    chunks: List[str] = []
    for c, lo in enumerate(range(0, n_taps, STROBE_CHUNK_BITS)):
        hi = min(lo + STROBE_CHUNK_BITS, n_taps) - 1
        name = f"setfi_c{c}"
        chunks.append(name)
        members = [f"setfi_t{i}" for i in range(hi, lo - 1, -1)]   # MSB first
        lines.append(f"wire [{hi - lo}:0] {name} = {{")
        for k in range(0, len(members), 8):
            sep = "," if k + 8 < len(members) else ""
            lines.append("    " + ", ".join(members[k:k + 8]) + sep)
        lines.append("};")
    lines.append("")
    lines.append("initial begin : setfi_tb_rec_init")
    for i, c in enumerate(sample_cycles):
        lines.append(f"    setfi_cyc_list[{i:5d}] = {c};")
    lines.append('    if (!$value$plusargs("setfi_events=%s", setfi_events))')
    lines.append('        setfi_events = "setfi_events.txt";')
    lines.append('    setfi_ev_fd = $fopen(setfi_events, "w");')
    lines.append('    if (setfi_ev_fd == 0)')
    lines.append('        $display("[SETFI-REC] WARN: cannot create %s", setfi_events);')
    lines.append('    $display("[SETFI-REC] events=%s n_taps=%0d n_samples=%0d",')
    lines.append("             setfi_events, `SETFI_N_TAPS, `SETFI_N_SAMPLES);")
    lines.append("end")
    lines.append("")
    fmt = "%0d %b " + "%h" * len(chunks)
    lines.append(f"always @(negedge tb.{clk_net}) begin : setfi_strobe_recorder")
    lines.append(f"    for (setfi_cyc_k = 0; setfi_cyc_k < {n_samples}; setfi_cyc_k = setfi_cyc_k + 1)")
    lines.append("        if (cycle_counter == setfi_cyc_list[setfi_cyc_k])")
    lines.append(f'            $fstrobe(setfi_ev_fd, "{fmt}", cycle_counter, setfi_guard,')
    top_down = list(reversed(chunks))
    for k in range(0, len(top_down), 8):
        sep = "," if k + 8 < len(top_down) else ");"
        lines.append("                     " + ", ".join(top_down[k:k + 8]) + sep)
    lines.append("end")
    lines.append("")
    lines.append("final begin")
    lines.append("    if (setfi_ev_fd != 0) $fclose(setfi_ev_fd);")
    lines.append("end")
    Path(svh_path).write_text("\n".join(lines) + "\n")


def replay_strobe_events(
    events_text: str, sample_cycles: Sequence[int], n_taps: int,
) -> Tuple[Dict[int, str], List[str]]:
    """Replay the "program" recorder's decision over strobe events.

    The program recorder, at every falling edge, records when
    ``guard && ptr < n && counter == list[ptr]`` and then advances ``ptr``.
    The strobe recorder logs (counter, settled guard, settled state) at every
    falling edge whose counter is ANY candidate -- a superset of the edges at
    which the program recorder can fire -- so replaying the pointer over them
    selects exactly the same (edge, state) pairs.

    Returns ({cycle: hex digits}, log lines).  Only a guard of exactly '1' is
    true (an X guard is false in Verilog's ``if``).
    """
    n_hex = n_hex_digits(n_taps)
    ptr = 0
    n = len(sample_cycles)
    out: Dict[int, str] = {}
    log: List[str] = []
    for ln, raw in enumerate(events_text.splitlines(), start=1):
        if not raw.strip():
            continue
        m = STROBE_EVENT_RE.match(raw.strip())
        if not m:
            raise WorkloadError("recorder_semantics",
                                f"strobe event line {ln} is malformed: {raw[:120]!r}")
        cyc, guard, hexs = int(m.group(1)), m.group(2), m.group(3)
        if len(hexs) != n_hex:
            raise WorkloadError("recorder_semantics",
                                f"strobe event line {ln}: {len(hexs)} hex digits, expected {n_hex}")
        if guard == "1" and ptr < n and cyc == sample_cycles[ptr]:
            out[cyc] = hexs
            log.append(f"[SETFI-REC] cyc={cyc} (event line {ln})")
            ptr += 1
    return out, log
