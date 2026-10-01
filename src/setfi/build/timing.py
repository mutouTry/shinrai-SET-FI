"""Timing part of the build: SDF + cell functions -> timing data.

Reads the SDF once and produces, keyed to the net and instance indices:

  arc delays (in memory)        per instance arc: default rise/fall delay and the
                                conditional (COND / edge-sensitive) delay patterns
  delay_table.npz + arc_index   arc delays flattened to integer-ps arrays for the engine
  cell_arcs.json                per (cell, arc): function, side pins, truth table,
                                unmask conditions
  timing_checks.json            per FF instance: setup/hold/recovery/removal checks
  interconnect.json             per (net, sink pin): wire transport delay

SDF dialects: DC (OVI 2.1, ``(min:typ:max)``, split SETUP/HOLD) and Innovus
(SDF 3.0, ``(min::max)``, merged SETUPHOLD/RECREM) are both accepted and normalised;
Any TIMESCALE from the header is converted; a DIVIDER of ``.`` is converted to ``/``.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from statistics import median
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .celllib import (TRUTH_TABLE_MAX_VARS, CellFunctions, normalize_space, sort_conjs)
from .netlist import strip_comments
from .spec import BuildError

_NS_TO_PS = Decimal("1000")
_PRINT_EXAMPLES = 20


@dataclass(frozen=True)
class TimingOptions:
    rail: str = "late"
    time_precision_ps: int = 1
    side_pins_topk: int = 10
    exclude_src_from_side: bool = True
    mirror_scan_data_checks: bool = True
    scan_data_pin_pairs: Tuple[Tuple[str, str], ...] = (("D", "SCD"),)


def q_ns_to_ps(ns_val: Any, time_precision_ps: int) -> int:
    """ns -> integer ps on the time lattice, rounding half up (exact decimal)."""
    q = Decimal(str(max(time_precision_ps, 1)))
    d_ps = Decimal(str(ns_val)) * _NS_TO_PS
    ticks = (d_ps / q).to_integral_value(rounding=ROUND_HALF_UP)
    return int(ticks * q)


# ============================================================
# Stats
# ============================================================
@dataclass
class Stats:
    counters: Counter = field(default_factory=Counter)
    samples: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    def hit(self, key: str, sample: Optional[str] = None) -> None:
        self.counters[key] += 1
        if sample and len(self.samples[key]) < _PRINT_EXAMPLES:
            self.samples[key].append(sample)

    def to_json(self) -> Dict[str, Any]:
        return {"counters": dict(self.counters), "samples": dict(self.samples)}


# ============================================================
# COND expressions
# ============================================================
_RE_CMP = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*(==|!=)\s*(1'b[01]|[01])\s*$")
_CONST_MAP = {"0": 0, "1": 1, "1'b0": 0, "1'b1": 1}


def unquote(tok: str) -> str:
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1]
    return tok


def merge_conj(a: Dict[str, int], b: Dict[str, int]) -> Optional[Dict[str, int]]:
    out = dict(a)
    for k, v in b.items():
        if k in out and out[k] != v:
            return None
        out[k] = v
    return out


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s


def _split_top_level(expr: str, op: str) -> List[str]:
    expr = expr.strip()
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth -= 1
            buf.append(ch)
            i += 1
            continue
        if depth == 0 and expr.startswith(op, i):
            parts.append("".join(buf).strip())
            buf = []
            i += len(op)
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf).strip())
    return [p for p in parts if p]


def compile_cond_to_conjs(expr: str) -> Optional[List[Dict[str, int]]]:
    """A COND expression as a disjunction of pin==value conjunctions (None if it is
    not of that shape)."""
    if not expr:
        return None
    expr = _strip_outer_parens(expr)
    conjs: List[Dict[str, int]] = []
    for term in _split_top_level(expr, "||"):
        conj: Optional[Dict[str, int]] = {}
        for lit in _split_top_level(_strip_outer_parens(term), "&&"):
            lit = _strip_outer_parens(lit)
            m = _RE_CMP.match(lit)
            if not m:
                return None
            pin, op, c = m.group(1), m.group(2), m.group(3)
            val = _CONST_MAP.get(c)
            if val is None:
                return None
            if op == "!=":
                val = 1 - val
            if pin in conj and conj[pin] != val:
                conj = None
                break
            conj[pin] = val
        if conj is not None:
            conjs.append(conj)
    return sort_conjs(conjs) if conjs else None


def conj_to_expr(conj: Dict[str, int]) -> Optional[str]:
    if not conj:
        return None
    return " && ".join(f"{k} == 1'b{v}" for k, v in sorted(conj.items()))


def pattern_from_conj(conj: Dict[str, int], side_pins: List[str]) -> Tuple[int, int]:
    mask = bits = 0
    for i, p in enumerate(side_pins):
        if p in conj:
            mask |= (1 << i)
            if conj[p]:
                bits |= (1 << i)
    return mask, bits


def _popcount(x: int) -> int:
    return bin(x).count("1")


# ============================================================
# SDF S-expression parser
# ============================================================
# ( | ) | "string" (a backslash escapes the next char) | bare token (no space, no paren)
_SDF_TOKEN_RE = re.compile(r'[()]|"(?:[^"\\]|\\.)*"|[^\s()]+', re.S)
_SDF_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"', re.S)


def tokenize_sexpr(s: str) -> List[str]:
    tokens = _SDF_TOKEN_RE.findall(s)
    for t in tokens:
        if t[0] == '"' and not _SDF_STRING_RE.fullmatch(t):
            raise BuildError("unterminated string literal in the SDF")
    return tokens


def parse_sexpr(tokens: List[str]) -> List[Any]:
    stack: List[List[Any]] = []
    cur: List[Any] = []
    for t in tokens:
        if t == "(":
            stack.append(cur)
            new_list: List[Any] = []
            cur.append(new_list)
            cur = new_list
        elif t == ")":
            if not stack:
                raise BuildError("unbalanced ')' in the SDF")
            cur = stack.pop()
        else:
            cur.append(t)
    if stack:
        raise BuildError("unbalanced '(' in the SDF")
    return cur


def iter_cells(tree: Any):
    if isinstance(tree, list):
        for node in tree:
            if isinstance(node, list) and node and node[0] == "CELL":
                yield node
            if isinstance(node, list):
                yield from iter_cells(node)


def find_first_list(form: List[Any], head: str) -> Optional[List[Any]]:
    for x in form:
        if isinstance(x, list) and x and x[0] == head:
            return x
    return None


def flatten_tokens(x: Any) -> str:
    if isinstance(x, list):
        return "(" + " ".join(flatten_tokens(i) for i in x) + ")"
    return str(x)


# ============================================================
# Endpoints
# ============================================================
@dataclass(frozen=True)
class EndpointInfo:
    pin: str
    edge: Optional[str]


def fmt_endpoint(node: Any) -> str:
    if isinstance(node, list) and len(node) == 2:
        return f"{unquote(str(node[0]))}:{unquote(str(node[1]))}"
    return unquote(str(node))


def split_endpoint_str(ep: str) -> EndpointInfo:
    if ":" in ep:
        a, b = ep.split(":", 1)
        if a in ("posedge", "negedge"):
            return EndpointInfo(pin=b, edge=a)
    return EndpointInfo(pin=ep, edge=None)


def split_iopath_endpoint(node: Any) -> EndpointInfo:
    if isinstance(node, list) and len(node) == 2:
        return EndpointInfo(pin=unquote(str(node[1])), edge=unquote(str(node[0])))
    return EndpointInfo(pin=unquote(str(node)), edge=None)


def resolve_pin_net_idx(inst: str, pin: str, pin_to_net: Dict[str, str],
                        net_to_idx: Dict[str, int]) -> Optional[int]:
    net = pin_to_net.get(f"{inst}.{pin}")
    if net is None:
        return None
    return net_to_idx.get(net)


def resolve_endpoint_net_idx(inst: str, endpoint: str, pin_to_net: Dict[str, str],
                             net_to_idx: Dict[str, int]) -> Optional[int]:
    ep = split_endpoint_str(endpoint)
    return resolve_pin_net_idx(inst, ep.pin, pin_to_net, net_to_idx)


# ============================================================
# SDF raw rows
# ============================================================
@dataclass
class Triplet:
    dmin: float
    dtyp: Optional[float]   # None == the writer left this field EMPTY (min::max)
    dmax: float


@dataclass
class DelayArcRow:
    celltype: str
    inst: str
    src_pin: str
    src_edge: Optional[str]
    dst_pin: str
    cond_expr: Optional[str]
    rise: Optional[Triplet]
    fall: Optional[Triplet]


@dataclass
class TimingCheckRow:
    celltype: str
    inst: str
    check_type: str
    data_event: str
    ref_event: str
    enable_cond: Optional[str]
    val: Triplet


@dataclass
class InterconnectRow:
    """One ``(INTERCONNECT <src> <sink> (rise) (fall))`` record.

    `src`/`sink` are SDF endpoints: `inst/pin` (hierarchy divider `/`) or a bare
    top-level port name.  Both name the SAME electrical net; the record is a
    per-SINK transport delay on that net.
    """
    src: str
    dst: str
    rise: Optional[Triplet]
    fall: Optional[Triplet]


def parse_triplet_atom(atom: str) -> Triplet:
    """Parse an SDF `(min:typ:max)` atom POSITIONALLY.

    An empty field means "the writer supplied no value for this rail", not "one
    fewer element": `(a::b)` is min a, no typ, max b.

    Fields are NOT sorted.  `min > max` is legal and common: the triplet is
    (early, nominal, late) of a SIGNED quantity, so a derated negative hold
    reads `-0.012:-0.014:-0.014`.
    """
    parts = atom.split(":")
    if _UNIT_NS[0] != 1:
        parts = [str(Decimal(p) * _UNIT_NS[0]) if p != "" else "" for p in parts]
    if len(parts) == 1:
        v = float(parts[0])
        return Triplet(v, v, v)
    if len(parts) != 3:
        raise ValueError(
            f"Bad SDF triplet {atom!r}: expected 1 or 3 colon-separated fields "
            f"(min:typ:max), got {len(parts)}")
    lo = float(parts[0]) if parts[0] != "" else None
    typ = float(parts[1]) if parts[1] != "" else None
    hi = float(parts[2]) if parts[2] != "" else None
    if lo is None and hi is None:
        if typ is None:
            raise ValueError(f"Bad SDF triplet {atom!r}: all three fields empty")
        lo = hi = typ
    elif lo is None:
        lo = hi
    elif hi is None:
        hi = lo
    return Triplet(float(lo), typ, float(hi))


def rail_value(t: Triplet, rail: str) -> float:
    """The ONE place the min::max rail is chosen.

    Deliberately does NOT consult `dtyp`: a writer-supplied typ and the chosen rail
    are different quantities, and mixing them is how the choice became invisible.
    On a DC SDF `typ == max` holds everywhere, so rail="late" equals the typ values.
    """
    if rail == "late":
        return float(t.dmax)
    if rail == "early":
        return float(t.dmin)
    raise ValueError(f"sdf rail must be 'late' or 'early', got {rail!r}")


def parse_delay_list(lst: Any) -> Triplet:
    if not isinstance(lst, list) or not lst:
        raise ValueError(f"Bad delay list: {lst}")
    return parse_triplet_atom(str(lst[0]))


def parse_delay_list_opt(lst: Any) -> Optional[Triplet]:
    if not isinstance(lst, list) or len(lst) == 0:
        return None
    return parse_triplet_atom(str(lst[0]))


def _emit_iopath(out: List[DelayArcRow], celltype: str, inst: str,
                 cond_expr: Optional[str], iopath: List[Any]) -> None:
    if len(iopath) < 4:
        return
    src_ep = split_iopath_endpoint(iopath[1])
    dst_ep = split_iopath_endpoint(iopath[2])
    try:
        if len(iopath) == 4:
            one = parse_delay_list_opt(iopath[3])
            rise, fall = one, one
        else:
            rise, fall = parse_delay_list_opt(iopath[3]), parse_delay_list_opt(iopath[4])
    except Exception:   # noqa: BLE001
        return
    if rise is None and fall is None:
        return
    out.append(DelayArcRow(
        celltype=celltype, inst=inst,
        src_pin=src_ep.pin, src_edge=src_ep.edge,
        dst_pin=dst_ep.pin, cond_expr=cond_expr,
        rise=rise, fall=fall,
    ))


# ------------------------------------------------------------
# SDF dialect -- writer-independent timing-check front-end
# ------------------------------------------------------------
# SDF writers spell the same checks differently, e.g.
#
#   synthesis (OVI 2.1, min:typ:max)   SETUP / HOLD / RECOVERY
#   P&R       (SDF 3.0, min::max)      SETUPHOLD / RECREM / WIDTH
#
# The merged forms are standard SDF and semantically identical, argument order included:
#     (SETUPHOLD <data> <ref> <setup> <hold>) == (SETUP <data> <ref> <setup>)
#                                              + (HOLD  <data> <ref> <hold>)
#     (RECREM    <data> <ref> <rec>   <rem> ) == (RECOVERY ...) + (REMOVAL ...)
# so the rewrite is LOSSLESS and both are accepted, normalised to the split form.
MERGED_CHECKS: Dict[str, Tuple[str, str]] = {
    "SETUPHOLD": ("SETUP", "HOLD"),
    "RECREM": ("RECOVERY", "REMOVAL"),
}

# Checks the engine has no model for.  Counted and reported as a known limit rather than
# dropped by an incidental length guard (a WIDTH row is 3 long).
UNMODELLED_CHECKS: Tuple[str, ...] = (
    "WIDTH", "PERIOD", "NOCHANGE", "SKEW", "BIDIRECTSKEW", "PATHCONSTRAINT",
)


def extract_timingchecks_from_cell(cell: List[Any], celltype: str, inst: str,
                                   notes: Counter) -> List[TimingCheckRow]:
    out: List[TimingCheckRow] = []
    tc = find_first_list(cell, "TIMINGCHECK")
    if not tc:
        return out
    for item in tc[1:]:
        if not isinstance(item, list) or not item:
            notes["timing_check_not_a_list"] += 1
            continue
        check_type = str(item[0]).upper()
        notes[f"timing_checks_read:{check_type}"] += 1
        if check_type in UNMODELLED_CHECKS:
            notes[f"timing_checks_not_modelled:{check_type}"] += 1
            continue
        split = MERGED_CHECKS.get(check_type)
        need = 5 if split else 4
        if len(item) < need:
            notes[f"timing_checks_malformed:{check_type}"] += 1
            continue
        data_event = fmt_endpoint(item[1])
        enable_cond = None
        ref_node = item[2]
        if isinstance(ref_node, list) and len(ref_node) >= 3 and ref_node[0] == "COND":
            enable_cond = normalize_space(flatten_tokens(ref_node[1]))
            ref_event = fmt_endpoint(ref_node[2])
        else:
            ref_event = fmt_endpoint(ref_node)
        emit: List[Tuple[str, Any]] = (
            [(split[0], item[3]), (split[1], item[4])] if split else [(check_type, item[3])]
        )
        for ct, node in emit:
            try:
                val = parse_delay_list(node)
            except Exception as exc:   # noqa: BLE001
                notes[f"timing_check_bad_value:{ct}"] += 1
                notes[f"timing_check_bad_value_detail:{exc}"] += 1
                continue
            if split:
                notes[f"timing_checks_split:{check_type}->{ct}"] += 1
            out.append(TimingCheckRow(
                celltype=celltype, inst=inst, check_type=ct,
                data_event=data_event, ref_event=ref_event,
                enable_cond=enable_cond, val=val,
            ))
    return out


# TIMESCALE spellings -> picoseconds per unit.  DC writes `1ns`, Innovus `1.0 ns`.
_TIMESCALE_PS: Dict[str, int] = {
    "1ps": 1, "10ps": 10, "100ps": 100,
    "1ns": 1000, "10ns": 10000, "100ns": 100000,
    "1us": 1000000,
}


def _normalise_timescale(raw: str) -> str:
    """`1.0 ns` / `1 ns` / `1ns` -> `1ns`.  Returns the raw string if unparsable."""
    s = normalize_space(raw).replace(" ", "").lower()
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)(ps|ns|us|s)", s)
    if not m:
        return s
    num = float(m.group(1))
    num_s = str(int(num)) if num == int(num) else str(num)
    return f"{num_s}{m.group(2)}"


def extract_sdf_header(tree: Any) -> Dict[str, str]:
    """Read the DELAYFILE header.  TIMESCALE and DIVIDER are validated here and
    applied by ``read_sdf`` (values are converted to ns, hierarchy to `/`)."""
    root = find_first_list(tree, "DELAYFILE") or tree
    hdr: Dict[str, str] = {}
    for key in ("SDFVERSION", "DESIGN", "DATE", "VENDOR", "PROGRAM", "VERSION",
                "DIVIDER", "VOLTAGE", "PROCESS", "TEMPERATURE", "TIMESCALE"):
        node = find_first_list(root, key)
        if node is not None and len(node) >= 2:
            hdr[key] = unquote(" ".join(str(x) for x in node[1:]))

    ts_raw = hdr.get("TIMESCALE", "1ns")
    ts = _normalise_timescale(ts_raw)
    hdr["_timescale_normalised"] = ts
    if ts not in _TIMESCALE_PS:
        raise BuildError(f"SDF TIMESCALE {ts_raw!r} is not a unit this reader knows "
                         f"(one of {sorted(_TIMESCALE_PS)})")
    div = hdr.get("DIVIDER", "/").strip()
    if div not in ("", "/", "."):
        raise BuildError(f"SDF DIVIDER {div!r} is not supported (use '/' or '.')")
    return hdr


def _emit_interconnect(out: List[InterconnectRow], item: List[Any], notes: Counter) -> None:
    # (INTERCONNECT src sink (r) (f))  -- one delay list means rise == fall.
    if len(item) < 4:
        notes["interconnect_malformed"] += 1
        return
    try:
        if len(item) == 4:
            one = parse_delay_list_opt(item[3])
            rise, fall = one, one
        else:
            rise = parse_delay_list_opt(item[3])
            fall = parse_delay_list_opt(item[4])
    except Exception as exc:   # noqa: BLE001
        notes["interconnect_bad_value"] += 1
        notes[f"interconnect_bad_value_detail:{exc}"] += 1
        return
    if rise is None and fall is None:
        notes["interconnect_without_delay"] += 1
        return
    out.append(InterconnectRow(src=unquote(str(item[1])), dst=unquote(str(item[2])),
                               rise=rise, fall=fall))


@dataclass
class SdfData:
    arcs: List[DelayArcRow]
    tcs: List[TimingCheckRow]
    inst_celltype: Dict[str, str]
    ics: List[InterconnectRow]
    header: Dict[str, str]
    notes: Counter


# Size of one SDF time unit in ns while a file is being read (see read_sdf).
_UNIT_NS = [Decimal(1)]


def read_sdf(path: str) -> SdfData:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = strip_comments(f.read())
    tree = parse_sexpr(tokenize_sexpr(txt))
    header = extract_sdf_header(tree)
    _UNIT_NS[0] = Decimal(_TIMESCALE_PS[header["_timescale_normalised"]]) / Decimal(1000)
    try:
        data = _read_sdf_tree(tree, header)
    finally:
        _UNIT_NS[0] = Decimal(1)
    if header.get("DIVIDER", "/").strip() == ".":
        for r in data.arcs + data.tcs:
            r.inst = r.inst.replace(".", "/")
        data.inst_celltype = {k.replace(".", "/"): v for k, v in data.inst_celltype.items()}
        for r in data.ics:
            r.src, r.dst = r.src.replace(".", "/"), r.dst.replace(".", "/")
    return data


def _read_sdf_tree(tree: Any, header: Dict[str, str]) -> SdfData:
    notes: Counter = Counter()

    arcs: List[DelayArcRow] = []
    tcs: List[TimingCheckRow] = []
    ics: List[InterconnectRow] = []
    inst_celltype: Dict[str, str] = {}

    for cell in iter_cells(tree):
        ct = find_first_list(cell, "CELLTYPE")
        ins = find_first_list(cell, "INSTANCE")
        # An `(INSTANCE)` with no argument is the DESIGN-scoped cell.  Innovus puts
        # EVERY INTERCONNECT record in it, so it must not be skipped.
        instance_scoped = bool(ct) and bool(ins) and len(ct) >= 2 and len(ins) >= 2
        celltype = unquote(str(ct[1])) if (ct and len(ct) >= 2) else ""
        inst = unquote(str(ins[1])) if instance_scoped else ""
        if instance_scoped:
            inst_celltype[inst] = celltype

        delay = find_first_list(cell, "DELAY")
        if delay:
            abs_lst = next(
                (x for x in delay if isinstance(x, list) and x and x[0] == "ABSOLUTE"), None)
            if abs_lst is not None:
                for item in abs_lst[1:]:
                    if not isinstance(item, list) or not item:
                        continue
                    head = str(item[0])
                    if head == "INTERCONNECT":
                        _emit_interconnect(ics, item, notes)
                        continue
                    if not instance_scoped:
                        notes[f"delay_item_outside_instance:{head}"] += 1
                        continue
                    if head == "COND" and len(item) >= 3:
                        iopath = item[-1]
                        cond_expr = normalize_space(
                            " ".join(flatten_tokens(x) for x in item[1:-1]))
                        if isinstance(iopath, list) and iopath and iopath[0] == "IOPATH":
                            _emit_iopath(arcs, celltype, inst, cond_expr, iopath)
                        else:
                            notes["delay_cond_without_iopath"] += 1
                    elif head == "IOPATH":
                        _emit_iopath(arcs, celltype, inst, None, item)
                    else:
                        notes[f"delay_item_unhandled:{head}"] += 1
        if instance_scoped:
            tcs.extend(extract_timingchecks_from_cell(cell, celltype, inst, notes))

    notes["interconnect_rows_read"] = len(ics)
    return SdfData(arcs=arcs, tcs=tcs, inst_celltype=inst_celltype, ics=ics,
                   header=header, notes=notes)


# ============================================================
# arc delays
# ============================================================
def select_side_pins(rows: List[DelayArcRow], logic_side_pins: List[str],
                     opts: TimingOptions) -> List[str]:
    freq: Counter = Counter()
    for r in rows:
        if r.cond_expr:
            conjs = compile_cond_to_conjs(r.cond_expr)
            if conjs is not None:
                for c in conjs:
                    for p in c.keys():
                        freq[p] += 1
    ordered: List[str] = []
    seen: Set[str] = set()
    src_pin = rows[0].src_pin if rows else None
    for p in logic_side_pins:
        if opts.exclude_src_from_side and p == src_pin:
            continue
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    for p, _ in freq.most_common():
        if opts.exclude_src_from_side and p == src_pin:
            continue
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered[:opts.side_pins_topk]


def build_arc_delays(rows: List[DelayArcRow], inst_to_idx: Dict[str, int], net_to_idx: Dict[str, int],
             pin_to_net: Dict[str, str], funcs: CellFunctions, stats: Stats,
             opts: TimingOptions) -> Dict[str, Any]:
    rail = opts.rail
    grouped: Dict[Tuple[str, str, str, str], List[DelayArcRow]] = defaultdict(list)
    for r in rows:
        grouped[(r.inst, r.celltype, r.src_pin, r.dst_pin)].append(r)

    arcs_obj: List[Dict[str, Any]] = []
    for (inst, celltype, src_pin, dst_pin), grp in sorted(grouped.items()):
        inst_idx = inst_to_idx.get(inst)
        if inst_idx is None:
            stats.hit("sdf_instance_not_in_netlist", inst)
            continue
        in_lib = celltype in funcs.cells
        logic_sem = funcs.arc(celltype, src_pin, dst_pin) if in_lib else None
        if not in_lib:
            stats.hit("sdf_cell_not_in_library", celltype)
        elif logic_sem is None:
            stats.hit("sdf_arc_not_combinational", f"{celltype}:{src_pin}->{dst_pin}")

        side_pins = select_side_pins(grp, logic_sem.side_pins if logic_sem else [], opts)

        patt_rise: List[Dict[str, Any]] = []
        patt_fall: List[Dict[str, Any]] = []
        default_uncond_r: List[float] = []
        default_uncond_f: List[float] = []
        default_all_r: List[float] = []
        default_all_f: List[float] = []

        for r in grp:
            if r.rise is not None:
                default_all_r.append(rail_value(r.rise, rail))
            if r.fall is not None:
                default_all_f.append(rail_value(r.fall, rail))
            if r.cond_expr is None and r.src_edge is None:
                if r.rise is not None:
                    default_uncond_r.append(rail_value(r.rise, rail))
                if r.fall is not None:
                    default_uncond_f.append(rail_value(r.fall, rail))
                continue
            base_conjs: List[Dict[str, int]] = [dict()]
            if r.cond_expr:
                cc = compile_cond_to_conjs(r.cond_expr)
                if cc is None:
                    stats.hit("sdf_cond_not_understood", r.cond_expr)
                    continue
                base_conjs = cc
            if r.src_edge is not None:
                if logic_sem is None:
                    stats.hit("sdf_edge_arc_not_combinational",
                              f"{inst}:{src_pin}->{dst_pin}:{r.src_edge}")
                    continue
                if logic_sem.kind == "seq_async":
                    continue
                for tr_name, trip in (("rise", r.rise), ("fall", r.fall)):
                    if trip is None:
                        continue
                    logic_conjs = logic_sem.event_patterns.get(r.src_edge, {}).get(tr_name, [])
                    if not logic_conjs:
                        continue
                    for bc in base_conjs:
                        for lc in logic_conjs:
                            mc = merge_conj(bc, lc)
                            if mc is None:
                                continue
                            mask, bits = pattern_from_conj(mc, side_pins)
                            ent = {"mask": mask, "value_bits": bits,
                                   "cond_expr": conj_to_expr(mc),
                                   "delay_typ": rail_value(trip, rail), "src_edge": r.src_edge}
                            (patt_rise if tr_name == "rise" else patt_fall).append(ent)
                continue
            for bc in base_conjs:
                mask, bits = pattern_from_conj(bc, side_pins)
                if r.rise is not None:
                    patt_rise.append({"mask": mask, "value_bits": bits,
                                      "cond_expr": conj_to_expr(bc),
                                      "delay_typ": rail_value(r.rise, rail)})
                if r.fall is not None:
                    patt_fall.append({"mask": mask, "value_bits": bits,
                                      "cond_expr": conj_to_expr(bc),
                                      "delay_typ": rail_value(r.fall, rail)})

        def _dedup(lst: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            uniq: Dict[Tuple, Dict[str, Any]] = {}
            for e in lst:
                key = (e.get("src_edge"), int(e["mask"]), int(e["value_bits"]),
                       float(e["delay_typ"]), e.get("cond_expr"))
                uniq[key] = e
            out = list(uniq.values())
            out.sort(key=lambda e: (-_popcount(int(e["mask"])), str(e.get("src_edge") or ""),
                                    float(e["delay_typ"]), int(e["mask"]), int(e["value_bits"])))
            return out

        patt_rise = _dedup(patt_rise)
        patt_fall = _dedup(patt_fall)

        def_r = (float(max(default_uncond_r)) if default_uncond_r
                 else (float(median(default_all_r)) if default_all_r else 0.0))
        def_f = (float(max(default_uncond_f)) if default_uncond_f
                 else (float(median(default_all_f)) if default_all_f else 0.0))

        arcs_obj.append({
            "arc_id": f"{inst}:{src_pin}->{dst_pin}",
            "inst": inst, "inst_idx": inst_idx,
            "celltype": celltype, "src": src_pin, "dst": dst_pin,
            "src_net_idx": resolve_pin_net_idx(inst, src_pin, pin_to_net, net_to_idx),
            "dst_net_idx": resolve_pin_net_idx(inst, dst_pin, pin_to_net, net_to_idx),
            "side_pins": side_pins,
            "default_delay_typ": {"rise": def_r, "fall": def_f},
            "patterns": {"rise": patt_rise, "fall": patt_fall},
            "logic_kind": logic_sem.kind if logic_sem else None,
            "clock_pin": logic_sem.clock_pin if logic_sem else None,
            "clock_edge": logic_sem.clock_edge if logic_sem else None,
            "async_pin": logic_sem.async_pin if logic_sem else None,
            "async_edge": logic_sem.async_edge if logic_sem else None,
            "async_value": logic_sem.async_value if logic_sem else None,
        })

    arcs_obj.sort(key=lambda x: (x["inst_idx"], x["src"], x["dst"]))
    for i, arc in enumerate(arcs_obj):
        arc["arc_idx"] = i
    for arc in arcs_obj:
        arc["patterns"] = {s: [{k: v for k, v in p.items() if k != "cond_expr"} for p in pl]
                           for s, pl in arc["patterns"].items()}

    return {
        "version": 4,
        "side_pins_topk": opts.side_pins_topk,
        "exclude_src_from_side": opts.exclude_src_from_side,
        "edge_sensitive_patterns": True,
        # DECLARED, not inferred: which end of every (min:typ:max) triplet the numbers
        # below come from.
        "sdf_delay_rail": rail,
        "note_default_delay_typ": (
            "`default_delay_typ` and `patterns[*].delay_typ` hold the chosen rail "
            "(see sdf_delay_rail), not an SDF `typ` field."
        ),
        "arcs": arcs_obj,
    }


# ============================================================
# cell arcs
# ============================================================
def assert_side_pin_bases_agree(delay_arcs: List[Dict[str, Any]], cell_arcs: Dict[str, Any],
                                stats: Stats) -> int:
    """the arc delays and the cell arcs must agree on the side-pin ORDER, or every conditional delay is wrong.

    `patt_mask` bit i is assigned against the arc delays' per-arc `side_pins` (`select_side_pins`),
    while the consumer maps mask bits to nets through the cell arcs' `side_pins`.  If the two lists
    differ nothing errors: bits beyond the shorter list read as 0, so `cond == 1'b1` rows
    NEVER match and `cond == 1'b0` rows ALWAYS match, and the arc silently takes the wrong
    delay -- which is also the electrical-masking threshold.

    The two lists are built by DIFFERENT rules and can genuinely diverge:
      * `select_side_pins` drops `src_pin` when `exclude_src_from_side`, then APPENDS pins
        that only appear in a COND expression;
      * cell arcs takes the function's side pins verbatim.
    arc delays may legitimately be a PREFIX-compatible superset, but every shared position must
    name the same pin.  Nothing in the code guarantees this for every library and SDF
    writer, so it is asserted here, where both are in hand.
    """
    checked = 0
    bad: List[str] = []
    for a in delay_arcs:
        sp2 = a.get("side_pins")
        if sp2 is None:
            continue
        mk = f'{a.get("celltype")}:{a.get("src")}->{a.get("dst")}'
        ent = cell_arcs.get(mk)
        if not isinstance(ent, dict) or "side_pins" not in ent:
            stats.hit("side_pins_not_checked", mk)
            continue
        checked += 1
        sp3 = list(ent["side_pins"])
        n = min(len(sp2), len(sp3))
        if list(sp2)[:n] != sp3[:n]:
            if len(bad) < 10:
                bad.append(f"{mk}: delays={list(sp2)} cell_arcs={sp3}")
    if bad:
        raise BuildError(
            f"side-pin ORDER disagrees between the arc delays and the cell arcs on {len(bad)}+ of {checked} arcs: "
            f"mask bit i would be matched against a different pin than the one it was "
            f"assigned for, so conditional delays would be silently wrong. Offenders: "
            + "; ".join(bad))
    return checked


def build_cell_arcs(needed_cell_arcs: Set[Tuple[str, str, str]], funcs: CellFunctions,
                        stats: Stats, opts: TimingOptions) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for (celltype, src_pin, dst_pin) in sorted(needed_cell_arcs):
        if celltype not in funcs.cells:
            stats.hit("cell_arc_cell_not_in_library", celltype)
            continue
        sem = funcs.arc(celltype, src_pin, dst_pin)
        if sem is None:
            stats.hit("cell_arc_not_combinational", f"{celltype}:{src_pin}->{dst_pin}")
            continue
        side_pins = sem.side_pins[:opts.side_pins_topk]
        unmask_entries: List[Dict[str, Any]] = []
        for c in sem.unmask:
            c2 = {k: v for k, v in c.items() if k in side_pins}
            mask, bits = pattern_from_conj(c2, side_pins)
            unmask_entries.append({"cond_expr": conj_to_expr(c2), "mask": mask,
                                   "value_bits": bits})
        if not unmask_entries:
            unmask_entries.append({"cond_expr": None, "mask": 0, "value_bits": 0})
        vars_order: List[str] = [src_pin] + [p for p in side_pins if p != src_pin]
        n_vars = len(vars_order)
        truth: List[int] = []
        if sem.dst_expr_py and n_vars <= TRUTH_TABLE_MAX_VARS:
            # The expression is generated by celllib.ast_to_py from a parsed AST (names
            # and & | ^ ~ if/else only); evaluated without builtins.  A name outside
            # vars_order (an internal node the function could not be resolved through)
            # raises and leaves the table empty.
            try:
                code = compile(sem.dst_expr_py, "<cell_arc_truth>", "eval")
                for idx in range(1 << n_vars):
                    env = {v: (idx >> j) & 1 for j, v in enumerate(vars_order)}
                    truth.append(int(eval(code, {"__builtins__": {}}, env)) & 1)  # noqa: S307
            except Exception:   # noqa: BLE001
                truth = []
        rec: Dict[str, Any] = {
            "celltype": celltype, "src": src_pin, "dst": dst_pin,
            "kind": sem.kind, "dst_assign_py": sem.dst_expr_py,
            "side_pins": side_pins, "vars_order": vars_order,
            "n_vars": n_vars, "truth": truth, "unmask": unmask_entries,
        }
        if sem.kind != "comb":
            rec.update({
                "clock_pin": sem.clock_pin, "clock_edge": sem.clock_edge,
                "async_pin": sem.async_pin, "async_edge": sem.async_edge,
                "async_value": sem.async_value,
            })
        out[f"{celltype}:{src_pin}->{dst_pin}"] = rec
    return out


# ============================================================
# timing checks
# ============================================================
def _mirror_scan_data_checks(inst: str, checks: Dict[str, Any], pin_to_net: Dict[str, str],
                             net_to_idx: Dict[str, int], stats: Stats,
                             opts: TimingOptions) -> None:
    """For a scan FF with a check on one data pin (D) but not on its twin (scan data),
    or vice versa, copy the available check to the missing pin."""
    if not opts.mirror_scan_data_checks:
        return
    for d_pin, s_pin in opts.scan_data_pin_pairs:
        has_d = f"{inst}.{d_pin}" in pin_to_net
        has_s = f"{inst}.{s_pin}" in pin_to_net
        if not (has_d or has_s):
            continue
        pairs = [
            (f"posedge:{d_pin}", f"posedge:{s_pin}", d_pin, s_pin),
            (f"negedge:{d_pin}", f"negedge:{s_pin}", d_pin, s_pin),
            (f"posedge:{s_pin}", f"posedge:{d_pin}", s_pin, d_pin),
            (f"negedge:{s_pin}", f"negedge:{d_pin}", s_pin, d_pin),
        ]
        for dst_ev, src_ev, dst_pin, src_pin in pairs:
            if dst_ev in checks or src_ev not in checks:
                continue
            if f"{inst}.{dst_pin}" not in pin_to_net or f"{inst}.{src_pin}" not in pin_to_net:
                continue
            new_ref_map: Dict[str, Any] = {}
            for ref_ev, bucket in checks[src_ev].items():
                if not isinstance(bucket, dict):
                    continue
                new_ref_map[ref_ev] = {
                    check_type: [
                        {**dict(x),
                         "data_net_idx": resolve_pin_net_idx(inst, dst_pin, pin_to_net,
                                                             net_to_idx),
                         "synthetic_from": src_ev}
                        for x in arr
                    ]
                    for check_type, arr in bucket.items()
                }
            if new_ref_map:
                checks[dst_ev] = new_ref_map
                stats.hit("scan_data_check_copied", f"{inst}:{src_ev}->{dst_ev}")


def build_timing_checks(tcs: List[TimingCheckRow], inst_celltype: Dict[str, str],
                   inst_to_idx: Dict[str, int], net_to_idx: Dict[str, int],
                   pin_to_net: Dict[str, str], stats: Stats,
                   opts: TimingOptions) -> Dict[str, Any]:
    rail = opts.rail
    idx: Dict[str, Dict[str, Any]] = {}
    for r in tcs:
        inst_idx = inst_to_idx.get(r.inst)
        if inst_idx is None:
            continue
        key = str(inst_idx)
        if key not in idx:
            idx[key] = {
                "meta": {"ff_inst": r.inst, "inst_idx": inst_idx,
                         "celltype": inst_celltype.get(r.inst, "")},
                "checks": {},
            }
        checks = idx[key]["checks"]
        checks.setdefault(r.data_event, {}).setdefault(r.ref_event, {}).setdefault(
            r.check_type, []).append({
                "enable_cond": r.enable_cond,
                "min": r.val.dmin, "typ": r.val.dtyp, "max": r.val.dmax,
                # The value the consumer must use.  `typ` is null whenever the writer
                # emitted `(min::max)`; `value` is the declared rail, computed here.
                "value": rail_value(r.val, rail),
                "rail": rail,
                "celltype": r.celltype,
                "data_net_idx": resolve_endpoint_net_idx(r.inst, r.data_event, pin_to_net,
                                                         net_to_idx),
                "ref_net_idx": resolve_endpoint_net_idx(r.inst, r.ref_event, pin_to_net,
                                                        net_to_idx),
            })
        if r.val.dmin != r.val.dmax:
            stats.hit("timing_check_depends_on_sdf_rail",
                      f"{r.inst}:{r.check_type} {r.val.dmin}..{r.val.dmax}")

    for rec in idx.values():
        ff_inst = rec["meta"].get("ff_inst")
        if isinstance(ff_inst, str):
            _mirror_scan_data_checks(ff_inst, rec["checks"], pin_to_net, net_to_idx,
                                     stats, opts)

    # version 3: merged SETUPHOLD / RECREM are split into the four standard keys, and
    # every entry carries `value` (the declared rail) alongside min/typ/max.
    return {"version": 3, "sdf_delay_rail": rail, "index": idx}


# ============================================================
# Interconnect
# ------------------------------------------------------------
# A wire is NOT a gate.  A pulse crossing it arrives later but is not electrically
# masked by it -- the sink sees the same width the driver sent.  So an interconnect
# delay must never be folded into a cell arc: the engine uses that arc for TWO things
# (t_out = t0 + arc, and deadline = t0 + arc), and only the first is a transport delay.
#
# A sink is TRANSPORTABLE iff a combinational gate in the graph reads it, i.e. its
# instance is combinational (carries no timing check -- timing checks is per-FF) AND its pin drives
# at least one IOPATH arc in this SDF.  "Drives an IOPATH arc" alone is NOT sufficient:
# an FF's CP and CDN pins drive IOPATH arcs too, but a flip-flop is the cone BOUNDARY.
# ============================================================
def build_interconnect(ics: List[InterconnectRow], arcs: List[DelayArcRow],
                          tcs: List[TimingCheckRow], inst_celltype: Dict[str, str],
                          pin_to_net: Dict[str, str], net_to_idx: Dict[str, int],
                          stats: Stats, opts: TimingOptions) -> Dict[str, Any]:
    rail = opts.rail
    prec = opts.time_precision_ps
    arc_source_pins: Set[Tuple[str, str]] = {(r.inst, r.src_pin) for r in arcs}
    seq_insts: Set[str] = {r.inst for r in tcs}   # timing checks is per-FF by construction
    check_ref_pins: Set[Tuple[str, str]] = set()
    check_data_pins: Set[Tuple[str, str]] = set()
    for r in tcs:
        check_ref_pins.add((r.inst, split_endpoint_str(r.ref_event).pin))
        check_data_pins.add((r.inst, split_endpoint_str(r.data_event).pin))

    def _split_endpoint(ep: str) -> Tuple[Optional[str], Optional[str]]:
        """`a/b/PIN` -> ('a/b','PIN'); a bare name is a top-level port."""
        if "/" not in ep:
            return None, None
        inst, pin = ep.rsplit("/", 1)
        return inst, pin

    def _net_of(ep: str) -> Optional[str]:
        inst, pin = _split_endpoint(ep)
        if inst is None:
            return ep if ep in net_to_idx else None
        return pin_to_net.get(f"{inst}.{pin}")

    account: Counter = Counter()
    entries: List[Dict[str, Any]] = []
    for row in ics:
        account["total"] += 1
        rise_ps = q_ns_to_ps(rail_value(row.rise, rail), prec) if row.rise is not None else 0
        fall_ps = q_ns_to_ps(rail_value(row.fall, rail), prec) if row.fall is not None else 0
        nonzero = (rise_ps != 0 or fall_ps != 0)
        account["nonzero" if nonzero else "zero"] += 1

        sink_inst, sink_pin = _split_endpoint(row.dst)
        src_net = _net_of(row.src)
        sink_net = _net_of(row.dst)

        if sink_inst is None:
            account["sink_top_port"] += 1
            if nonzero:
                account["sink_top_port_nonzero"] += 1
            continue
        if sink_net is None:
            account["sink_unresolved"] += 1
            stats.hit("interconnect_sink_pin_not_in_netlist", row.dst)
            continue
        if src_net is not None and src_net != sink_net:
            # Both endpoints of an INTERCONNECT name the same electrical net.  A
            # mismatch means the netlist and the SDF disagree -- do not guess.
            account["net_mismatch"] += 1
            stats.hit("interconnect_pins_on_different_nets", f"{row.src} -> {row.dst}")
            continue

        if sink_inst in seq_insts:
            # Flip-flop = cone boundary.  Split by role so the residual is a number.
            if (sink_inst, sink_pin) in check_ref_pins:
                kind = "sink_seq_clock_pin"
            elif (sink_inst, sink_pin) in check_data_pins:
                kind = "sink_seq_data_pin"
            else:
                kind = "sink_seq_other_pin"
                stats.hit("interconnect_flip_flop_pin_unclassified", f"{row.dst}")
        elif (sink_inst, sink_pin) in arc_source_pins:
            kind = "sink_comb_input"          # <- the transportable set
            entries.append({
                "net": sink_net,
                "net_idx": net_to_idx.get(sink_net),
                "sink_inst": sink_inst,
                "sink_pin": sink_pin,
                "sink_celltype": inst_celltype.get(sink_inst, ""),
                "rise_ps": rise_ps,
                "fall_ps": fall_ps,
            })
        else:
            kind = "sink_no_arc_no_check"
            stats.hit("interconnect_sink_pin_unclassified",
                      f"{row.dst} ({inst_celltype.get(sink_inst, '?')})")
        account[kind] += 1
        if nonzero:
            account[kind + "_nonzero"] += 1

    # The account MUST balance: every parsed row lands in exactly one terminal bucket.
    terminal = ("sink_comb_input", "sink_top_port", "sink_unresolved",
                "net_mismatch", "sink_seq_clock_pin", "sink_seq_data_pin",
                "sink_seq_other_pin", "sink_no_arc_no_check")
    booked = sum(account[k] for k in terminal)
    if booked != account["total"]:
        raise BuildError(f"internal error: INTERCONNECT rows do not add up: booked={booked} "
                         f"total={account['total']}")
    # Name every terminal class, including the empty ones, so an ABSENT key means
    # exactly one thing: something is wrong.
    for k in terminal:
        account[k] = int(account[k])

    return {
        "version": 1,
        "sdf_delay_rail": rail,
        "semantics": (
            "TRANSPORT delay per (net, sink pin): the sink sees the driver's "
            "waveform shifted by rise_ps/fall_ps with its width unchanged. A "
            "wire performs NO electrical masking, so these values must not be "
            "added to any cell arc (an arc is also the masking threshold)."
        ),
        "row_account": dict(account),
        "entries": entries,
    }


# ============================================================
# Delay table + arc index
# ============================================================
def write_delay_table(arcs_obj: List[Dict[str, Any]], path: str, time_precision_ps: int
                      ) -> Tuple[int, int]:
    n = len(arcs_obj)
    total = sum(len(arc["patterns"].get("rise") or []) + len(arc["patterns"].get("fall") or [])
                for arc in arcs_obj)
    default_rise = np.zeros(n, dtype=np.int64)
    default_fall = np.zeros(n, dtype=np.int64)
    patt_ptr = np.zeros(n + 1, dtype=np.int32)
    patt_mask = np.zeros(total, dtype=np.int64)
    patt_val = np.zeros(total, dtype=np.int64)
    patt_dt = np.zeros(total, dtype=np.int64)
    patt_is_rise = np.zeros(total, dtype=np.uint8)
    p = 0
    for i, arc in enumerate(arcs_obj):
        d0 = arc.get("default_delay_typ") or {}
        default_rise[i] = q_ns_to_ps(d0.get("rise", 0.0), time_precision_ps)
        default_fall[i] = q_ns_to_ps(d0.get("fall", 0.0), time_precision_ps)
        patt = arc.get("patterns") or {}
        for is_rise, key in ((1, "rise"), (0, "fall")):
            for entry in (patt.get(key) or []):
                patt_mask[p] = int(entry.get("mask", 0))
                patt_val[p] = int(entry.get("value_bits", 0))
                dt = entry.get("delay_typ")
                patt_dt[p] = q_ns_to_ps(dt, time_precision_ps) if dt is not None else -1
                patt_is_rise[p] = is_rise
                p += 1
        patt_ptr[i + 1] = p
    np.savez(path,
             default_rise_ps=default_rise, default_fall_ps=default_fall,
             patt_ptr=patt_ptr, patt_mask=patt_mask,
             patt_valbits=patt_val, patt_dt_ps=patt_dt, patt_is_rise=patt_is_rise)
    return n, total


def build_arc_index(arcs_obj: List[Dict[str, Any]]) -> Dict[str, Any]:
    arc_idx_to_delay_key = [f"{a['inst']}:{a['src']}->{a['dst']}" for a in arcs_obj]
    delay_key_to_arc_idx = {k: i for i, k in enumerate(arc_idx_to_delay_key)}
    return {
        "version": 1,
        "n_arcs": len(arcs_obj),
        "arc_idx_to_delay_key": arc_idx_to_delay_key,
        "delay_key_to_arc_idx": delay_key_to_arc_idx,
    }


# ============================================================
# The stage
# ============================================================
def _dump(obj: Any, path: str, indent: Optional[int] = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, allow_nan=False)


def compile_stage(sdf: SdfData, funcs: CellFunctions, inst_to_idx: Dict[str, int],
                  net_to_idx: Dict[str, int], pin_to_net: Dict[str, str], out_dir: str,
                  opts: TimingOptions, intermediate_dir: Optional[str] = None
                  ) -> Dict[str, Any]:
    """Write cell arcs, timing checks, interconnect, delay_table.npz and arc_index.json into
    ``out_dir`` (arc delays into ``intermediate_dir`` when given).  Returns counts/diagnostics."""
    stats = Stats()
    arc_delays = build_arc_delays(sdf.arcs, inst_to_idx, net_to_idx, pin_to_net, funcs, stats, opts)
    arcs_obj = arc_delays["arcs"]
    if intermediate_dir:
        _dump(arc_delays, os.path.join(intermediate_dir, "arc_delays.json"), indent=2)
    _dump(build_arc_index(arcs_obj), os.path.join(out_dir, "arc_index.json"))
    n_arcs, n_patterns = write_delay_table(arcs_obj, os.path.join(out_dir, "delay_table.npz"),
                                           opts.time_precision_ps)

    needed = {(r.celltype, r.src_pin, r.dst_pin) for r in sdf.arcs}
    cell_arcs = build_cell_arcs(needed, funcs, stats, opts)
    n_side_checked = assert_side_pin_bases_agree(arcs_obj, cell_arcs, stats)
    _dump(cell_arcs, os.path.join(out_dir, "cell_arcs.json"), indent=2)

    timing_checks = build_timing_checks(sdf.tcs, sdf.inst_celltype, inst_to_idx, net_to_idx, pin_to_net,
                        stats, opts)
    _dump(timing_checks, os.path.join(out_dir, "timing_checks.json"), indent=2)

    interconnect = None
    if sdf.ics:
        interconnect = build_interconnect(sdf.ics, sdf.arcs, sdf.tcs, sdf.inst_celltype,
                                   pin_to_net, net_to_idx, stats, opts)
        _dump(interconnect, os.path.join(out_dir, "interconnect.json"), indent=2)

    return {
        "n_arcs": n_arcs,
        "n_delay_patterns": n_patterns,
        "n_timing_checks": len(sdf.tcs),
        "n_ff_with_checks": len(timing_checks["index"]),
        "n_cell_arcs": len(cell_arcs),
        "n_side_pin_bases_checked": n_side_checked,
        "n_interconnect": len(sdf.ics),
        "interconnect_row_account": interconnect["row_account"] if interconnect else None,
        "diagnostics": stats.to_json(),
    }
