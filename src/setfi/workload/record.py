"""Record the per-cycle circuit state of a workload (two simulation passes).

Pass 1 runs the user's testbench against the netlist through the recorder
wrapper without recording, checks the testbench's pass string and input timing,
and counts the clock cycles after reset (the ``[SETFI-CYC]`` markers).  The
cycles to record are then sampled (see :mod:`.sampling`); pass 2 records them,
cycles with too many unknown (X/Z) values are replaced by reserve cycles, and the
unused recordings are deleted.

Output directory::

    states/cycle_<N>.hex   one per sampled cycle (format: sampling.py)
    cycles.json            the sampled cycles and how they were chosen
    result.json            status, simulator commands, counts
    record.log
    sim/                   simulator working directory (generated wrapper and
                           recorder, file lists, compile/run logs, staged data
                           files, testbench/ with the adapted testbench)
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import platform
import shutil
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .. import __version__ as _PKG_VERSION
from .adapt_tb import adapt_testbench
from .errors import WorkloadError
from .sampling import (load_cycle_window, sample_cycles, sample_cycles_in_window,
                       screen_cycles)
from .simulate import check_simulator, parse_cycle_markers, run_pass
from .spec import WorkloadSpec
from .tbgen import (input_timing_monitor, WRAPPER_MODULE, emit_recorder_program, emit_recorder_strobe, emit_wrapper,
                    load_taps, replay_strobe_events, resolve_taps, rst_guard_expr,
                    verify_xmr_resolves)

TOOL_NAME = "setfi record"
CYCLES_FORMAT = "setfi-cycles/1"

SAMPLE_INSTANT = {
    "program": ("the settled values at the falling clock edge of cycle N (SV program "
                "block, Reactive region): the data the rising edge ending cycle N samples"),
    "strobe": ("the settled values at the falling clock edge of cycle N ($fstrobe, "
               "Postponed region): the data the rising edge ending cycle N samples"),
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_filelist(path: Path, files: List[Path]) -> None:
    path.write_text("\n".join(str(p) for p in files) + "\n")


def _abs(p: Path) -> Path:
    """Absolute, without resolving symlinks (container binds need absolute paths)."""
    return Path(p).expanduser().absolute()


def _design_inputs(spec) -> List[str]:
    """Input ports of the design's top module, as the testbench drives them."""
    from ..build.netlist import parse_modules
    mods = parse_modules(Path(spec.netlist).read_text(errors="replace"))
    top = mods.get(spec.top_module)
    if top is None:
        return []
    return sorted(p for p, d in top.port_dirs.items() if d == "input")


def _tail(text: str, n: int = 12) -> str:
    return "\n".join("    " + line for line in text.rstrip().splitlines()[-n:])


def record_workload(spec: WorkloadSpec, log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Record ``spec``'s workload.  Returns the result dict (also written to
    ``<output_dir>/result.json``); raises :class:`WorkloadError` on failure
    after writing ``result.json`` and ``record.log``."""
    started = time.monotonic()
    started_iso = _now_iso()
    out_dir = _abs(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "sim"
    work.mkdir(exist_ok=True)
    cs_dir = out_dir / "states"
    sim = spec.simulator
    log_lines: List[str] = []

    def _log(m: str) -> None:
        log_lines.append(m)
        if log is not None:
            log(m)

    _log(f"[{started_iso}] {TOOL_NAME} starting (v{_PKG_VERSION}, simulator={sim.name}, "
         f"recorder={sim.recorder})")
    result: Dict[str, Any] = {
        "tool": TOOL_NAME, "version": _PKG_VERSION, "status": "ok", "error": None,
        "outputs": {}, "metrics": {},
        "meta": {"started_iso": started_iso, "finished_iso": None, "duration_s": None,
                 "host": socket.gethostname(), "python_version": platform.python_version()},
    }
    try:
        check_simulator(sim)
        netlist = _abs(spec.netlist)
        if not netlist.is_file():
            raise WorkloadError("config_invalid", f"netlist not found: {netlist}")
        clk_net, rst_net, rst_pol = spec.clock_port, spec.reset_port, spec.reset_polarity
        rst_guard = rst_guard_expr(rst_net, rst_pol)
        _log(f"  clk_net={clk_net} rst_net={rst_net} polarity={rst_pol}")

        # ---- taps: every net of the netlist, in net-index order ----
        taps, ff_nets = load_taps(spec.net_index, spec.ff_index)
        _log(f"  recording {len(taps)} nets"
             + (f" ({len(ff_nets)} flip-flop D/Q nets)" if ff_nets is not None else ""))
        missing = verify_xmr_resolves(netlist.read_text(errors="ignore"), taps)
        if missing:
            raise WorkloadError(
                "xmr_unresolved",
                f"{len(missing)} tap leaf names not found in {netlist.name}; sample: "
                f"{missing[:10]}. The net_index does not belong to this netlist, or the "
                f"netlist renamed them.")
        bindings, bind_meta = resolve_taps(taps, spec.tb_dut_inst, ff_nets)
        if bind_meta["n_unresolved_taps"]:
            _log(f"  WARN: {bind_meta['n_unresolved_taps']} net(s) outside flip-flops could not "
                 f"be read and are recorded as 0 (cycles.json n_unreadable_nets).")

        # ---- testbench adaptation ----
        _log("  adapting the testbench to the netlist ...")
        ad = adapt_testbench(
            tb_files=spec.testbench, netlist=netlist, cell_models=spec.cell_models,
            top_module=spec.top_module, tb_top_module=spec.tb_top_module,
            out_dir=work / "testbench", rtl_dut_files=spec.rtl_dut_files,
            tb_dut_inst=spec.tb_dut_inst, hier_substitutions=spec.hier_substitutions,
            auto_derive_hier_substitutions=spec.auto_derive_hier_substitutions,
            macro_rtl_subs=spec.macro_rtl_subs, netlist_strip_modules=spec.netlist_strip_modules,
            netlist_substitute_files=spec.netlist_substitute_files, log=_log,
        )
        sim_netlist = ad.sim_netlist if ad.netlist_was_stripped else netlist
        _log(f"  testbench for the netlist: {[str(p) for p in ad.tb_files]}")
        if ad.extra_sources:
            _log(f"  sim netlist={sim_netlist.name} (stripped); extra sources="
                 f"{[p.name for p in ad.extra_sources]}")

        preamble: List[Path] = []
        if sim.timescale_preamble:
            p = work / "setfi_timescale.v"
            p.write_text(f"`timescale {sim.timescale_preamble}\n")
            preamble = [p]

        def filelist(path: Path, wrapper: Path, recorder_sources: List[Path]) -> Path:
            _write_filelist(path, preamble + list(ad.cell_models) + [sim_netlist]
                            + list(ad.extra_sources) + recorder_sources
                            + list(ad.tb_files) + [wrapper])
            return path

        # ---- pass 1 ----
        wrapper1 = work / "setfi_tb_wrapper_pass1.sv"
        monitor = input_timing_monitor(spec.tb_dut_inst, clk_net, rst_guard_expr(rst_net, rst_pol),
                                       _design_inputs(spec))
        emit_wrapper(wrapper1, tb_top_module=spec.tb_top_module, clk_net=clk_net,
                     rst_net=rst_net, rst_polarity=rst_pol, include_path=None,
                     version=_PKG_VERSION, monitor=monitor)
        fl1 = filelist(work / "pass1_filelist.f", wrapper1, [])
        staged = []
        for f in spec.tb_data_files:
            src = Path(f).resolve()
            if not src.is_file():
                raise WorkloadError("config_invalid", f"tb_data_files entry missing: {src}")
            shutil.copy2(src, work / src.name)
            staged.append(str(work / src.name))
        _log(f"  staged data files: {staged}")

        _log("  pass 1: clean run (no recording)")
        r1 = run_pass(sim, work_dir=work, output_dir=out_dir, filelist=fl1, top=WRAPPER_MODULE,
                      defines=spec.defines, plusargs=[],
                      compile_timeout_s=int(spec.compile_timeout_s),
                      sim_timeout_s=int(spec.sim_timeout_s), log_prefix="pass1", log=_log)
        if r1.compile_rc != 0:
            raise WorkloadError("compile_failed",
                                f"pass-1 compile failed (rc={r1.compile_rc}):\n{_tail(r1.compile_log)}\n"
                                f"full log: {work / 'pass1_compile.log'}")
        if r1.sim_rc != 0:
            _log(f"  pass-1 simulation returncode={r1.sim_rc} (nonzero)")
        markers1 = parse_cycle_markers(r1.sim_log, r1.sim_rc)
        _log(f"  pass 1: {markers1['last'] - markers1['first']} clock cycles after reset")
        late = re.search(r"^\[SETFI-LATE-INPUT\] t_ps=(\d+) port=(\S+)", r1.sim_log, re.M)
        if late:
            raise WorkloadError(
                "late_input",
                f"the testbench changes the design input {late.group(2)!r} between a falling "
                f"and the next rising clock edge (first at {late.group(1)} ps simulation "
                f"time); the state recorded at the falling edge would not be the state the "
                f"rising edge samples. Drive the inputs at the falling edge or just after "
                f"the rising edge.")
        if markers1["last"] - markers1["first"] < 1:
            raise WorkloadError(
                "reset_never_released",
                f"no clock cycle ran after reset: check workload.reset "
                f"({spec.reset_port!r}) and workload.reset_active, and that "
                f"workload.clock ({spec.clock_port!r}) toggles.")
        expected_pass = spec.expected_pass_string
        if expected_pass not in r1.sim_log:
            raise WorkloadError(
                "sim_failed",
                f"the testbench did not print workload.pass_string ({expected_pass!r}): its "
                f"self-check failed on the netlist, or it prints a different text when it "
                f"passes. Last lines:\n{_tail(r1.sim_log)}\nfull log: {work / 'pass1_sim.log'}")

        # ---- sampling ----
        n_sample = int(spec.n_sample_cycles)
        seed = int(spec.sample_seed)
        padding = int(spec.padding_cycles)
        warmup = int(spec.warmup_cycles)
        window_set = None
        window_meta = None
        if spec.cycle_window_csv:
            window_set, window_meta = load_cycle_window(spec.cycle_window_csv)
            sampled, reserve, n_pool = sample_cycles_in_window(
                markers1["first"], markers1["last"], n_sample, padding, seed,
                window_set, warmup=warmup, reserve_factor=int(spec.reserve_factor))
            window_meta["n_cycles_available"] = n_pool
            _log(f"  cycle window {window_meta['csv']} (md5 {window_meta['md5']}): "
                 f"{window_meta['n_rows']} run(s), {window_meta['n_cycles']} cycles, "
                 f"{n_pool} of them run after reset")
        else:
            sampled, reserve = sample_cycles(
                markers1["first"], markers1["last"], n_sample, padding, seed,
                warmup=warmup, reserve_factor=int(spec.reserve_factor))
        candidates = sorted(set(sampled) | set(reserve))
        _log(f"  sampled {len(sampled)} cycles in "
             f"[{markers1['first'] + 1 + padding + warmup}, {markers1['last'] - padding}]"
             + (f" (warmup={warmup})" if warmup else "") + f": first 5 = {sampled[:5]}")
        _log(f"  + {len(reserve)} reserve cycles, to replace cycles with unknown values "
             f"({len(candidates)} candidates total)")

        # ---- pass 2 ----
        svh = work / "setfi_tb_rec.svh"
        recorder_sources: List[Path] = []
        events_path = work / "setfi_events.txt"
        if sim.recorder == "program":
            prog = work / "setfi_tb_rec_prog.sv"
            emit_recorder_program(svh, prog, bindings=bindings, sample_cycles=candidates,
                                  clk_net=clk_net, rst_guard=rst_guard, version=_PKG_VERSION)
            recorder_sources = [prog]
            plusargs = [f"+setfi_outdir={cs_dir}"]
        else:
            emit_recorder_strobe(svh, bindings=bindings, sample_cycles=candidates,
                                 clk_net=clk_net, rst_guard=rst_guard, version=_PKG_VERSION)
            plusargs = [f"+setfi_events={events_path}"]
            events_path.unlink(missing_ok=True)
        _log(f"  emitted {svh} ({len(bindings)} taps, recorder={sim.recorder})")
        wrapper2 = work / "setfi_tb_wrapper_pass2.sv"
        emit_wrapper(wrapper2, tb_top_module=spec.tb_top_module, clk_net=clk_net,
                     rst_net=rst_net, rst_polarity=rst_pol, include_path=svh,
                     version=_PKG_VERSION)
        # The program recorder compiles at compilation-unit scope, after any
        # macro substitutes and before the testbench.
        fl2 = filelist(work / "pass2_filelist.f", wrapper2, recorder_sources)

        cs_dir.mkdir(exist_ok=True)
        for stale in cs_dir.glob("cycle_*.hex"):
            stale.unlink()
        _log("  pass 2: record run")
        r2 = run_pass(sim, work_dir=work, output_dir=out_dir, filelist=fl2, top=WRAPPER_MODULE,
                      defines=spec.defines, plusargs=plusargs,
                      compile_timeout_s=int(spec.compile_timeout_s),
                      sim_timeout_s=int(spec.sim_timeout_s), log_prefix="pass2", log=_log)
        if r2.compile_rc != 0:
            raise WorkloadError("compile_failed",
                                f"pass-2 compile failed (rc={r2.compile_rc}):\n{_tail(r2.compile_log)}\n"
                                f"full log: {work / 'pass2_compile.log'}")
        markers2 = parse_cycle_markers(r2.sim_log, r2.sim_rc)
        if markers2 != markers1:
            _log(f"  WARN: the recording run had {markers2['last'] - markers2['first']} cycles "
                 f"after reset, the first run {markers1['last'] - markers1['first']}")
        if expected_pass not in r2.sim_log:
            raise WorkloadError(
                "sim_failed",
                f"the recording run did not print workload.pass_string ({expected_pass!r}) "
                f"although the first run did. Last lines:\n{_tail(r2.sim_log)}\n"
                f"full log: {work / 'pass2_sim.log'}")

        if sim.recorder == "strobe":
            if not events_path.is_file():
                raise WorkloadError("sim_failed",
                                    f"strobe recorder wrote no event file {events_path}; "
                                    f"Last lines:\n{_tail(r2.sim_log)}\nfull log: {work / 'pass2_sim.log'}")
            states, rep_log = replay_strobe_events(events_path.read_text(), candidates, len(taps))
            for c, hexs in states.items():
                (cs_dir / f"cycle_{c}.hex").write_text(hexs + "\n")
            for m in rep_log:
                _log(f"    {m}")

        missing_hex = [c for c in candidates if not (cs_dir / f"cycle_{c}.hex").is_file()]
        if missing_hex:
            raise WorkloadError(
                "sim_failed",
                f"pass 2 should have recorded {len(candidates)} cycles but "
                f"{len(missing_hex)} are missing (sample: {missing_hex[:5]}). "
                f"Last lines:\n{_tail(r2.sim_log)}\nfull log: {work / 'pass2_sim.log'}")

        # ---- replace cycles with unknown values, prune ----
        sampled, x_screen = screen_cycles(cs_dir, sampled, reserve, n_sample,
                                          max_x_frac=float(spec.max_x_frac), log=_log)
        keep_set = set(sampled)
        for c in candidates:
            if c not in keep_set:
                (cs_dir / f"cycle_{c}.hex").unlink(missing_ok=True)
        _log(f"  states: {len(sampled)} cycles kept, first 5 = {sampled[:5]}")

        if window_meta is not None:
            outside = [c for c in sampled if c not in window_set]
            if outside:
                raise WorkloadError("no_cycles_in_range",
                                    f"{len(outside)} selected cycle(s) are outside the cycle "
                                    f"window {window_meta['csv']}: {outside[:10]}")
        meta: Dict[str, Any] = {
            "format": CYCLES_FORMAT,
            "cycles": sampled,
            "n_nets": len(taps),
            "net_numbering": "bit i of states/cycle_<N>.hex = net i of build/net_index.json",
            "sample_instant": SAMPLE_INSTANT[sim.recorder],
            "cycles_after_reset": markers1["last"] - markers1["first"],
            "sampling": {"seed": seed, "padding_cycles": padding, "warmup_cycles": warmup,
                         "cycle_window": window_meta, "candidates": candidates},
            "unknown_values": x_screen,
            "n_unreadable_nets": bind_meta["n_unresolved_taps"],
            "testbench": {"top": spec.tb_top_module, "dut_instance": spec.tb_dut_inst,
                          "clock": clk_net, "reset": rst_net, "reset_polarity": rst_pol},
            "simulator": sim.name,
            "recorder": sim.recorder,
        }
        meta_path = out_dir / "cycles.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        _log(f"  wrote {meta_path}")

        result["outputs"] = {
            "states": str(cs_dir),
            "cycles": str(meta_path),
            "log": str(out_dir / "record.log"),
            "sim_dir": str(work),
            "adapted_testbench": [str(p) for p in ad.tb_files],
            "compile_cmd_pass1": r1.compile_cmd,
            "run_cmd_pass1": r1.run_cmd,
            "compile_cmd_pass2": r2.compile_cmd,
            "run_cmd_pass2": r2.run_cmd,
        }
        result["metrics"] = {
            "n_nets": len(taps),
            "n_cycles": len(sampled),
            "cycles_after_reset": markers1["last"] - markers1["first"],
            "pass1_compile_rc": r1.compile_rc,
            "pass1_sim_rc": r1.sim_rc,
            "pass2_compile_rc": r2.compile_rc,
            "pass2_sim_rc": r2.sim_rc,
            "wall_clock_s": round(time.monotonic() - started, 4),
        }
        _log("  ok -- workload recorded")
    except WorkloadError as e:
        result["status"] = "error"
        result["error"] = {"kind": e.kind, "message": e.message, "trace": traceback.format_exc()}
        _log(f"ERROR: {e.message}")
        raise
    except Exception as e:  # unexpected: still leave a result behind
        result["status"] = "error"
        result["error"] = {"kind": "unexpected", "message": str(e), "trace": traceback.format_exc()}
        _log(f"ERROR [unexpected]: {e}")
        raise
    finally:
        result["meta"]["finished_iso"] = _now_iso()
        result["meta"]["duration_s"] = round(time.monotonic() - started, 4)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (out_dir / "record.log").write_text("\n".join(log_lines) + "\n")
    return result
