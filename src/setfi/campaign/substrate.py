"""Load the timing graph produced by the build stage.

The build stage (``setfi.build``) writes one directory with:

    net_index.json            global net numbering
    site_cones.jsonl          per injection site: its fan-out cone in
                              topological order and the FF D nets it reaches
    edges.json                one entry per (instance, input pin -> output pin)
    pin_to_net.json           instance pin -> net
    ff_index.json             the FFs whose capture is observed
    cell_arcs.json            per (cell, arc): truth table and side pins
    timing_checks.json        per FF instance: setup/hold checks from the SDF
    delay_table.npz           per arc: rise/fall delay and conditional patterns
    arc_index.json            arc key -> row of delay_table.npz
    interconnect.json         optional, per load pin: wire delay

Everything here is read-only and shared by all trials of a campaign.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

INF_PS = int(2**62)


class SubstrateError(RuntimeError):
    """The build artifacts are inconsistent or incomplete."""


def q_ns_to_ps(ns_val: Any, timeprecision_ps: int = 1) -> int:
    """ns -> integer ps on the time lattice, rounding half up (exact decimal)."""
    q = Decimal(1 if timeprecision_ps <= 0 else int(timeprecision_ps))
    ticks = (Decimal(str(ns_val)) * Decimal(1000) / q).to_integral_value(rounding=ROUND_HALF_UP)
    return int(ticks * q)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass(frozen=True)
class Edge:
    eid: int
    src_net: str
    dst_net: str
    inst: str
    celltype: str
    inpin: str
    outpin: str
    mask_key: str
    delay_key: str


@dataclass(frozen=True)
class DelayTable:
    default_rise_ps: np.ndarray
    default_fall_ps: np.ndarray
    patt_ptr: np.ndarray
    patt_mask: np.ndarray
    patt_valbits: np.ndarray
    patt_dt_ps: np.ndarray
    patt_is_rise: np.ndarray


@dataclass
class FFInfo:
    ffid: int
    inst: str
    d_net: str
    d_pin: str
    setup_rise_ps: int   # setup check for a rising D (posedge D)
    hold_rise_ps: int
    setup_fall_ps: int
    hold_fall_ps: int


@dataclass
class Substrate:
    """All static inputs of a campaign."""
    net_to_idx: Dict[str, int]
    idx_to_net: Dict[int, str]
    skeleton: Dict[str, Dict[str, Any]]
    edges: Dict[int, Edge]
    inst_pins: Dict[str, Dict[str, str]]
    cell_arcs: Dict[str, Any]
    delaytab: DelayTable
    dkey2id: Dict[str, int]
    ffs: Dict[int, FFInfo]
    dnet_to_ffids: Dict[str, List[int]]
    wire_lut: Dict[Tuple[str, str], Tuple[int, int]]
    counts: Dict[str, int]


def load_net_index(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    obj = _load_json(path)
    net_to_idx = obj.get("net_to_idx")
    idx_to_net_raw = obj.get("idx_to_net")
    if not isinstance(net_to_idx, dict) or not isinstance(idx_to_net_raw, dict):
        raise SubstrateError(f"{path}: missing net_to_idx / idx_to_net")
    return {str(k): int(v) for k, v in net_to_idx.items()}, {int(k): str(v) for k, v in idx_to_net_raw.items()}


def load_skeleton(path: str, idx_to_net: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
    """Per injection site: cone nets in topological order, cone adjacency, and
    the FF D nets it can reach."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                if "src_net" in obj:
                    src = str(obj["src_net"])
                    topo = [str(x) for x in obj.get("cone_topo_nets", [])]
                    adj = [[str(it[0]), [int(e) for e in it[1]]] for it in (obj.get("cone_adj") or [])]
                    dnets = [str(x) for x in obj.get("reachable_dnets", [])]
                elif "src_net_idx" in obj:
                    src = idx_to_net[int(obj["src_net_idx"])]
                    topo = [idx_to_net[int(ni)] for ni in (obj.get("cone_topo_net_idxs") or [])]
                    adj = [[idx_to_net[int(u)], [int(e) for e in (eids or [])]]
                           for u, eids in (obj.get("cone_adj_idx") or [])]
                    dnets = [idx_to_net[int(ni)] for ni in (obj.get("reachable_dnet_idxs") or [])]
                else:
                    raise KeyError("neither src_net nor src_net_idx")
            except (KeyError, ValueError, TypeError) as e:
                raise SubstrateError(f"{path}:{lineno}: malformed row of site_cones.jsonl ({e})") from e
            out[src] = {"src_net": src, "cone_topo_nets": topo, "cone_adj": adj,
                        "reachable_dnets": dnets, "has_reconv": obj.get("has_reconv")}
    return out


def load_edges(path: str) -> Dict[int, Edge]:
    out: Dict[int, Edge] = {}
    for e in _load_json(path):
        eid = int(e["eid"])
        out[eid] = Edge(eid=eid, src_net=str(e["src_net"]), dst_net=str(e["dst_net"]),
                        inst=str(e["inst"]), celltype=str(e.get("celltype", "")),
                        inpin=str(e.get("inpin", "")), outpin=str(e.get("outpin", "")),
                        mask_key=str(e.get("mask_key", "")), delay_key=str(e.get("delay_key", "")))
    return out


def load_pin_to_net(path: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for k, net in _load_json(path).items():
        kk = str(k).strip().replace(":", ".")
        if "." not in kk:
            continue
        inst, pin = kk.split(".", 1)
        out.setdefault(inst, {})[pin] = str(net).strip()
    return out


def load_delay_table(path: str) -> DelayTable:
    d = np.load(path)
    return DelayTable(
        default_rise_ps=d["default_rise_ps"].astype(np.int64),
        default_fall_ps=d["default_fall_ps"].astype(np.int64),
        patt_ptr=d["patt_ptr"].astype(np.int32),
        patt_mask=d["patt_mask"].astype(np.int64),
        patt_valbits=d["patt_valbits"].astype(np.int64),
        patt_dt_ps=d["patt_dt_ps"].astype(np.int64),
        patt_is_rise=d["patt_is_rise"].astype(np.uint8),
    )


def load_arc_index(path: str) -> Dict[str, int]:
    raw = _load_json(path).get("delay_key_to_arc_idx", {})
    return {str(k): int(v) for k, v in raw.items()}


def load_timing_checks(path: str) -> Dict[str, Any]:
    """timing checks index keyed by FF instance name."""
    obj = _load_json(path)
    idx = obj.get("index", {}) if isinstance(obj, dict) else {}
    if not isinstance(idx, dict):
        raise SubstrateError(f"{path}: unsupported timing-check index")
    out: Dict[str, Any] = {}
    for k, ent in idx.items():
        if not isinstance(ent, dict):
            raise SubstrateError(f"{path}: bad entry {k!r}")
        meta = ent.get("meta") or {}
        ff_inst = meta.get("ff_inst") if isinstance(meta, dict) else None
        if isinstance(ff_inst, str) and ff_inst.strip():
            out[ff_inst.strip()] = ent
        elif isinstance(k, str) and not k.isdigit():
            out[k] = ent
        else:
            raise SubstrateError(f"{path}: entry {k!r} names no FF instance")
    return out


def _check_value(ent: Dict[str, Any]) -> Optional[float]:
    """Value of one timing-check entry: the declared rail if present, else
    typ -> max -> min (a fabricated typ midpoint is skipped for max)."""
    if not isinstance(ent, dict):
        return None
    if ent.get("value") is not None:
        try:
            return float(ent["value"])
        except (TypeError, ValueError):
            pass
    lo, hi = ent.get("min"), ent.get("max")
    if ent.get("typ") is None and lo is not None and hi is not None and lo != hi:
        try:
            return float(hi)
        except (TypeError, ValueError):
            pass
    for k in ("typ", "max", "min"):
        if ent.get(k) is not None:
            try:
                return float(ent[k])
            except (TypeError, ValueError):
                pass
    return None


def setup_hold_ns(checks_by_ff: Dict[str, Any], ff_inst: str, data_edge: str, d_pin: str,
                  clock_pins: List[str]) -> Optional[Tuple[float, float]]:
    """(setup, hold) in ns for a ``data_edge`` ('posedge' / 'negedge') on ``d_pin``,
    against the first listed clock pin that has both checks.  A check written without
    a data edge in the SDF applies to both data edges."""
    ent = checks_by_ff.get(ff_inst)
    if not isinstance(ent, dict):
        return None
    checks = ent.get("checks") or {}

    def pair(cp: Any) -> Optional[Tuple[float, float]]:
        if not isinstance(cp, dict):
            return None
        su, ho = cp.get("SETUP") or [], cp.get("HOLD") or []
        if not su or not ho:
            return None
        s, h = _check_value(su[0]), _check_value(ho[0])
        return None if s is None or h is None else (float(s), float(h))

    for dkey in (f"{data_edge}:{d_pin}", d_pin):
        dd = checks.get(dkey)
        if not isinstance(dd, dict):
            continue
        for pin in clock_pins:
            for ref in (f"posedge:{pin}", f"negedge:{pin}", pin):
                v = pair(dd.get(ref))
                if v is not None:
                    return v
        # none of the listed clock pins: use the reference pin if exactly one has both checks
        found = {ref: pair(cp) for ref, cp in dd.items() if pair(cp) is not None}
        if len({r.split(":")[-1] for r in found}) == 1:
            return next(iter(found.values()))
    return None


def load_ff_thresholds(stage_dir: str, *, clock_pins: List[str], allow_missing: bool = False,
                       timeprecision_ps: int = 1) -> Dict[int, FFInfo]:
    """Setup/hold of every observed flip-flop, from the build's ff_index and timing checks.

    A flip-flop needs SETUP and HOLD checks for both a rising and a falling data edge
    against one of ``clock_pins``; without them it could never capture a pulse, so it
    is an error unless ``allow_missing``."""
    p = lambda name: os.path.join(stage_dir, name)  # noqa: E731
    for name in ("ff_index.json", "timing_checks.json"):
        if not os.path.exists(p(name)):
            raise SubstrateError(f"build output {p(name)} is missing")
    checks_by_ff = load_timing_checks(p("timing_checks.json"))
    ffidx = _load_json(p("ff_index.json"))
    info_map = ffidx.get("ffid_to_info")
    if not isinstance(info_map, dict):
        raise SubstrateError("ff_index.json has no ffid_to_info map")
    if not info_map:
        meta = (ffidx.get("_meta") or {}).get("observe_filter") or {}
        raise SubstrateError(
            f"no flip-flop is observed: {meta.get('n_input', 0)} sequential cell(s) were found "
            f"and all were excluded ({meta.get('n_removed', {})}; filter: "
            f"{meta.get('predicates', [])}). Check library.d_pins (which pin of a cell is its "
            f"data pin), observe.data_pins, observe.exclude_cells and observe.exclude_instances.")
    ffs: Dict[int, FFInfo] = {}
    bad: List[str] = []
    for k, info in info_map.items():
        ffid = int(k)
        d_net = str(info.get("d_net", "")).strip()
        ff_inst = str(info.get("ff_inst", "")).strip()
        d_pin = str(info.get("d_pin", "D")).strip() or "D"
        if not d_net or not ff_inst:
            raise SubstrateError(f"ff_index.json: FF {ffid} has no d_net/ff_inst")
        pos = setup_hold_ns(checks_by_ff, ff_inst, "posedge", d_pin, clock_pins)
        neg = setup_hold_ns(checks_by_ff, ff_inst, "negedge", d_pin, clock_pins)
        if pos is None or neg is None:
            bad.append(f"{ff_inst} ({d_pin})")
            continue
        q = lambda v: q_ns_to_ps(v, timeprecision_ps)  # noqa: E731
        ffs[ffid] = FFInfo(ffid=ffid, inst=ff_inst, d_net=d_net, d_pin=d_pin,
                           setup_rise_ps=q(pos[0]), hold_rise_ps=q(max(0.0, pos[1])),
                           setup_fall_ps=q(neg[0]), hold_fall_ps=q(max(0.0, neg[1])))
    if bad and not allow_missing:
        raise SubstrateError(
            f"{len(bad)} flip-flop(s) have no SETUP and HOLD check in the SDF for both a rising "
            f"and a falling data edge against a clock pin in "
            f"library.timing_check_clock_pins = {clock_pins}, so their sampling window is "
            f"unknown. First: {bad[:5]}. Add the checks to the SDF, add the clock pin name to "
            f"library.timing_check_clock_pins, or set observe.allow_missing_timing_checks = true "
            f"to leave these flip-flops unobserved.")
    return ffs


def load_wire_lut(stage_dir: str, mode: str) -> Dict[Tuple[str, str], Tuple[int, int]]:
    if mode == "ignore":
        return {}
    if mode != "transport":
        raise SubstrateError(f"interconnect mode must be 'ignore' or 'transport', got {mode!r}")
    path = os.path.join(stage_dir, "interconnect.json")
    if not os.path.exists(path):
        return {}          # the build writes interconnect only when the SDF has INTERCONNECT rows
    lut: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for e in _load_json(path).get("entries", []) or []:
        lut[(str(e.get("sink_inst", "")), str(e.get("sink_pin", "")))] = (
            int(e.get("rise_ps", 0)), int(e.get("fall_ps", 0)))
    return lut


def load_substrate(stage_dir: str, *, clock_pins: List[str], interconnect: str = "ignore",
                   timeprecision_ps: int = 1, allow_missing_timing_checks: bool = False) -> Substrate:
    """Load the build directory and resolve every FF's setup/hold values."""
    p = lambda name: os.path.join(stage_dir, name)  # noqa: E731
    for name in ("net_index.json", "site_cones.jsonl", "edges.json", "pin_to_net.json",
                 "ff_index.json", "cell_arcs.json", "timing_checks.json",
                 "delay_table.npz", "arc_index.json"):
        if not os.path.exists(p(name)):
            raise SubstrateError(f"build output {p(name)} is missing")

    net_to_idx, idx_to_net = load_net_index(p("net_index.json"))
    skeleton = load_skeleton(p("site_cones.jsonl"), idx_to_net)
    edges = load_edges(p("edges.json"))
    inst_pins = load_pin_to_net(p("pin_to_net.json"))
    cell_arcs = _load_json(p("cell_arcs.json"))
    if not isinstance(cell_arcs, dict):
        raise SubstrateError("cell_arcs.json must be an object")
    delaytab = load_delay_table(p("delay_table.npz"))
    dkey2id = load_arc_index(p("arc_index.json"))
    wire_lut = load_wire_lut(stage_dir, interconnect)
    ffs = load_ff_thresholds(stage_dir, clock_pins=clock_pins,
                             allow_missing=allow_missing_timing_checks,
                             timeprecision_ps=timeprecision_ps)
    dnet_to_ffids: Dict[str, List[int]] = {}
    for ffid in sorted(ffs):
        dnet_to_ffids.setdefault(ffs[ffid].d_net, []).append(ffid)

    return Substrate(net_to_idx=net_to_idx, idx_to_net=idx_to_net, skeleton=skeleton, edges=edges,
                     inst_pins=inst_pins, cell_arcs=cell_arcs, delaytab=delaytab, dkey2id=dkey2id, ffs=ffs,
                     dnet_to_ffids=dnet_to_ffids, wire_lut=wire_lut,
                     counts={"n_ff_observed": len(ffs), "n_sites": len(skeleton),
                             "n_nets": len(idx_to_net), "n_arcs": len(dkey2id),
                             "n_wires": len(wire_lut)})
