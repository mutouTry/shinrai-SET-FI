"""Per-site fan-out cone: build it from the substrate and pack it for the C++ engine."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .substrate import Substrate, SubstrateError


@dataclass
class Gate:
    out_i: int
    var_nets: List[int]                       # input nets in truth-table bit order
    truth: List[int]
    arc_dkey: List[str]                       # per input: delay key ("" = no arc)
    arc_cause_i: List[int]                    # per input: net whose revert cancels (EM)
    arc_side_nets: List[Optional[List[int]]]  # per input: side nets for COND lookup
    var_nets_src: List[int] = field(default_factory=list)  # driver nets before wire splicing
    inst: str = ""
    var_pins: List[str] = field(default_factory=list)


@dataclass
class Cone:
    src_net: str
    net2i: Dict[str, int]
    i2net: List[str]
    fanout: List[List[int]]
    gates: List[Gate]
    rec_nets: List[int]
    rec_idx: List[int]
    wire_pairs: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class PackedCone:
    n_nets: int
    n_gates: int
    fanout_ptr: np.ndarray
    fanout_idx: np.ndarray
    gate_out: np.ndarray
    gate_var_ptr: np.ndarray
    gate_var_nets: np.ndarray
    gate_truth_ptr: np.ndarray
    gate_truth: np.ndarray
    arc_dkey_id: np.ndarray
    arc_cause_i: np.ndarray
    arc_side_mode: np.ndarray
    arc_side_ptr: np.ndarray
    arc_side_nets: np.ndarray
    rec_idx: np.ndarray
    wire_ptr: np.ndarray
    wire_zsplit: np.ndarray
    wire_sink: np.ndarray
    wire_rise_ps: np.ndarray
    wire_fall_ps: np.ndarray


@dataclass
class CapturePlan:
    """Which recorded D nets feed which observed FFs, and their thresholds.

    Local FF index li runs over the FFs this site can reach (sorted by ffid).
    ``thr_enter[base, li]`` / ``thr_leave[base, li]`` are absolute times (ps
    from the start of the cycle) bounding the FF's sampling window for a pulse
    away from fault-free D value ``base``.
    """
    ffids: np.ndarray            # int32[n_ff] global ffid per local index
    thr_enter: np.ndarray        # int64[2, n_ff]
    thr_leave: np.ndarray        # int64[2, n_ff]
    dp_rec_slot: np.ndarray      # int32[n_dp] record slot of each D net
    dp_ptr: np.ndarray           # int32[n_dp+1] CSR: D net -> local FF indices
    dp_idx: np.ndarray           # int32[...]
    dp_dnet_names: List[str]
    word_index: np.ndarray       # for capture_masks bit packing
    word_mask: np.ndarray
    n_words: int

    @property
    def n_ff(self) -> int:
        return int(self.ffids.shape[0])


class ConeStats:
    def __init__(self) -> None:
        self.unconnected_side_pins = 0
        self.dnets_without_observed_ff = 0


def build_cone(sub: Substrate, src_net: str, stats: ConeStats) -> Cone:
    src_obj = sub.skeleton[src_net]
    topo_nets: List[str] = list(src_obj["cone_topo_nets"])
    cone_nets = set(topo_nets)
    cell_arcs = sub.cell_arcs

    gate_key_to_gid: Dict[Tuple[str, str, str], int] = {}
    gate_inpin_arcs: List[Dict[str, Dict[str, str]]] = []
    fanout_by_net: Dict[str, List[int]] = defaultdict(list)
    for _u, eids in src_obj["cone_adj"]:
        for eid in eids:
            e = sub.edges.get(int(eid))
            if e is None:
                raise SubstrateError(f"site {src_net}: cone references unknown edge {eid}")
            if e.dst_net not in cone_nets:
                continue
            gkey = (e.inst, e.outpin, e.dst_net)
            gid = gate_key_to_gid.get(gkey)
            if gid is None:
                gid = len(gate_key_to_gid)
                gate_key_to_gid[gkey] = gid
                gate_inpin_arcs.append({})
            gate_inpin_arcs[gid][e.inpin] = {"in_net": e.src_net, "delay_key": e.delay_key,
                                             "mask_key": e.mask_key}
            fanout_by_net[e.src_net].append(gid)

    gid_to_gkey: List[Tuple[str, str, str]] = [("", "", "")] * len(gate_key_to_gid)
    for gkey, gid in gate_key_to_gid.items():
        gid_to_gkey[gid] = gkey

    all_nets = set(topo_nets)
    gate_build = []
    for gid, arcmap in enumerate(gate_inpin_arcs):
        inst, _outpin, out_net = gid_to_gkey[gid]
        mask_keys = [d["mask_key"] for d in arcmap.values()]
        ent = next((cell_arcs[mk] for mk in mask_keys if isinstance(cell_arcs.get(mk), dict)), None)
        if ent is None:
            raise SubstrateError(
                f"cell function missing for instance {inst} (output net {out_net}, arcs "
                f"{mask_keys[:4]}): the library parser produced no truth table for this cell, so "
                f"no pulse could propagate through it. Check that the cell's model is readable.")
        pinmap = sub.inst_pins.get(inst, {})
        vars_ = ent.get("vars_order")
        truth = ent.get("truth")
        if not (isinstance(truth, list) and truth and isinstance(vars_, list) and vars_):
            raise SubstrateError(f"cell entry for {mask_keys[0]} has no truth table")
        var_net_names = []
        for v in vars_:
            n = pinmap.get(v)
            if not n:
                raise SubstrateError(f"instance {inst}: input pin {v} is not connected in the netlist")
            var_net_names.append(str(n))

        arc_dkey: List[str] = []
        arc_cause_net: List[str] = []
        arc_side_names: List[Optional[List[str]]] = []
        for v in vars_:
            arc = arcmap.get(v)
            if not arc:
                arc_dkey.append("")
                arc_cause_net.append("")
                arc_side_names.append(None)
                continue
            arc_dkey.append(str(arc.get("delay_key") or ""))
            arc_cause_net.append(str(arc.get("in_net") or ""))
            arc_entry = cell_arcs.get(str(arc.get("mask_key") or ""))
            side_pins = (arc_entry.get("side_pins") or []) if isinstance(arc_entry, dict) else []
            if not side_pins:
                arc_side_names.append([])
                continue
            side_nets: List[str] = []
            for sp in side_pins:
                nn = pinmap.get(sp)
                if not nn:
                    stats.unconnected_side_pins += 1
                    side_nets = None  # type: ignore[assignment]
                    break
                side_nets.append(str(nn))
            arc_side_names.append(side_nets)

        all_nets.add(out_net)
        all_nets.update(var_net_names)
        all_nets.update(c for c in arc_cause_net if c)
        for sn in arc_side_names:
            if sn:
                all_nets.update(sn)
        gate_build.append((gid, inst, list(vars_), out_net, var_net_names, list(truth),
                           arc_dkey, arc_cause_net, arc_side_names))

    i2net: List[str] = []
    net2i: Dict[str, int] = {}
    for n in list(topo_nets) + sorted(all_nets):
        if n not in net2i:
            net2i[n] = len(i2net)
            i2net.append(n)
    n_nets = len(i2net)

    fanout_raw: List[List[int]] = [[] for _ in range(n_nets)]
    for net_name, gids in fanout_by_net.items():
        ni = net2i.get(net_name)
        if ni is not None:
            fanout_raw[ni].extend(gids)

    old_to_new: Dict[int, int] = {}
    gates: List[Gate] = []
    for (gid, inst, var_pins, out_net, var_net_names, truth, arc_dkey, arc_cause_net,
         arc_side_names) in gate_build:
        old_to_new[gid] = len(gates)
        var_nets = [net2i[n] for n in var_net_names]
        side: List[Optional[List[int]]] = []
        for sn in arc_side_names:
            if sn is None:
                side.append(None)
            elif not sn:
                side.append([])
            else:
                side.append([net2i.get(n, -1) for n in sn])
        gates.append(Gate(out_i=net2i[out_net], var_nets=var_nets, truth=truth, arc_dkey=arc_dkey,
                          arc_cause_i=[net2i.get(c, -1) if c else -1 for c in arc_cause_net],
                          arc_side_nets=side, var_nets_src=list(var_nets), inst=inst,
                          var_pins=var_pins))

    fanout: List[List[int]] = [[] for _ in range(n_nets)]
    for ni in range(n_nets):
        if fanout_raw[ni]:
            fanout[ni] = [old_to_new[g] for g in fanout_raw[ni] if g in old_to_new]

    wire_pairs = insert_transport_nodes(i2net, net2i, gates, fanout, sub.wire_lut)

    rec_nets: List[int] = []
    rec_idx = [-1] * len(i2net)
    for dn in src_obj["reachable_dnets"]:
        ni = net2i.get(dn)
        if ni is None:
            raise SubstrateError(f"site {src_net}: reachable D net {dn} is not in its cone")
        if rec_idx[ni] < 0:
            rec_idx[ni] = len(rec_nets)
            rec_nets.append(ni)

    return Cone(src_net=src_net, net2i=net2i, i2net=i2net, fanout=fanout, gates=gates,
                rec_nets=rec_nets, rec_idx=rec_idx, wire_pairs=wire_pairs)


def insert_transport_nodes(i2net, net2i, gates, fanout, wire_lut) -> List[Tuple[int, int, int, int]]:
    """Splice a transport node in front of every wired gate input pin.

    A wire delays a pulse and masks nothing, so it is not modelled as a gate:
    each wired input pin gets its own "pin node" net and the engine copies the
    driver's transitions onto it with a pure delay.  Nodes are created for every
    wired pin, including zero-delay ones.  Existing net indices never change.
    """
    wire_pairs: List[Tuple[int, int, int, int]] = []
    wire_of_sink: Dict[int, Tuple[int, int, int]] = {}
    if not wire_lut:
        return wire_pairs
    for gi, g in enumerate(gates):
        if not g.inst or not g.var_pins:
            continue
        for j in range(min(len(g.var_nets), len(g.var_pins))):
            d = wire_lut.get((g.inst, g.var_pins[j]))
            if d is None:
                continue
            rise_ps, fall_ps = int(d[0]), int(d[1])
            drv = int(g.var_nets[j])
            if drv < 0:
                continue
            name = f"{i2net[drv]}\x01{g.inst}/{g.var_pins[j]}"
            sink = net2i.get(name)
            if sink is None:
                sink = len(i2net)
                i2net.append(name)
                net2i[name] = sink
                fanout.append([])
                wire_pairs.append((drv, sink, rise_ps, fall_ps))
                wire_of_sink[sink] = (drv, rise_ps, fall_ps)
            elif wire_of_sink.get(sink, (drv, rise_ps, fall_ps)) != (drv, rise_ps, fall_ps):
                raise SubstrateError(f"pin node {name!r} reached twice with different wires")
            g.var_nets[j] = sink
            if j < len(g.arc_cause_i) and int(g.arc_cause_i[j]) == drv:
                g.arc_cause_i[j] = sink
            for k, sn in enumerate(g.arc_side_nets):
                if sn:
                    g.arc_side_nets[k] = [sink if int(x) == drv else int(x) for x in sn]
            if gi not in fanout[sink]:
                fanout[sink].append(gi)
    for gi, g in enumerate(gates):
        still = set(int(x) for x in g.var_nets)
        for drv in set(int(x) for x in g.var_nets_src):
            if drv < 0 or drv in still or drv >= len(fanout):
                continue
            if gi in fanout[drv]:
                fanout[drv] = [x for x in fanout[drv] if x != gi]
    return wire_pairs


def _pack_wires(n_nets: int, wire_pairs):
    by_drv: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for drv, sink, r, f in wire_pairs:
        by_drv[int(drv)].append((int(sink), int(r), int(f)))
    wire_ptr = np.zeros(n_nets + 1, dtype=np.int32)
    wire_zsplit = np.zeros(n_nets, dtype=np.int32)
    sinks: List[int] = []
    rises: List[int] = []
    falls: List[int] = []
    for ni in range(n_nets):
        wire_ptr[ni] = len(sinks)
        lst = by_drv.get(ni)
        if not lst:
            wire_zsplit[ni] = len(sinks)
            continue
        for sk, r, f in [x for x in lst if x[1] == 0 and x[2] == 0]:
            sinks.append(sk); rises.append(r); falls.append(f)  # noqa: E702
        wire_zsplit[ni] = len(sinks)
        for sk, r, f in [x for x in lst if not (x[1] == 0 and x[2] == 0)]:
            sinks.append(sk); rises.append(r); falls.append(f)  # noqa: E702
    wire_ptr[n_nets] = len(sinks)
    return (wire_ptr, wire_zsplit, np.asarray(sinks, dtype=np.int32),
            np.asarray(rises, dtype=np.int64), np.asarray(falls, dtype=np.int64))


def pack_cone(cone: Cone, dkey2id: Dict[str, int]) -> PackedCone:
    n_nets = len(cone.i2net)
    gates = cone.gates
    n_gates = len(gates)

    fanout_ptr = np.zeros(n_nets + 1, dtype=np.int32)
    np.cumsum([len(x) for x in cone.fanout], out=fanout_ptr[1:])
    fanout_idx = np.asarray([g for lst in cone.fanout for g in lst], dtype=np.int32)

    gate_out = np.asarray([g.out_i for g in gates], dtype=np.int32)
    gate_var_ptr = np.zeros(n_gates + 1, dtype=np.int32)
    np.cumsum([len(g.var_nets) for g in gates], out=gate_var_ptr[1:])
    sum_k = int(gate_var_ptr[-1])

    gate_truth_ptr = np.zeros(n_gates + 1, dtype=np.int32)
    np.cumsum([len(g.truth) for g in gates], out=gate_truth_ptr[1:])
    gate_truth = np.asarray([t for g in gates for t in g.truth], dtype=np.uint8)

    gate_var_nets = np.empty(sum_k, dtype=np.int32)
    arc_dkey_id = np.empty(sum_k, dtype=np.int32)
    arc_cause_i = np.empty(sum_k, dtype=np.int32)
    arc_side_mode = np.empty(sum_k, dtype=np.int8)
    arc_side_ptr = np.zeros(sum_k + 1, dtype=np.int32)
    side_flat: List[int] = []
    vk = 0
    for g in gates:
        for j, ni in enumerate(g.var_nets):
            gate_var_nets[vk] = int(ni)
            dk = g.arc_dkey[j] if j < len(g.arc_dkey) else ""
            arc_dkey_id[vk] = int(dkey2id.get(dk, -1)) if dk else -1
            arc_cause_i[vk] = int(g.arc_cause_i[j]) if j < len(g.arc_cause_i) else -1
            sn = g.arc_side_nets[j] if j < len(g.arc_side_nets) else None
            if sn is None:
                arc_side_mode[vk] = 0      # side values unknown: unconditional delay
            elif len(sn) == 0:
                arc_side_mode[vk] = 1      # no side pins
            else:
                arc_side_mode[vk] = 2
                side_flat.extend(int(x) for x in sn)
            arc_side_ptr[vk + 1] = len(side_flat)
            vk += 1

    w_ptr, w_zsplit, w_sink, w_rise, w_fall = _pack_wires(n_nets, cone.wire_pairs)
    return PackedCone(n_nets=n_nets, n_gates=n_gates, fanout_ptr=fanout_ptr, fanout_idx=fanout_idx,
                      gate_out=gate_out, gate_var_ptr=gate_var_ptr, gate_var_nets=gate_var_nets,
                      gate_truth_ptr=gate_truth_ptr, gate_truth=gate_truth,
                      arc_dkey_id=arc_dkey_id, arc_cause_i=arc_cause_i,
                      arc_side_mode=arc_side_mode, arc_side_ptr=arc_side_ptr,
                      arc_side_nets=np.asarray(side_flat, dtype=np.int32),
                      rec_idx=np.asarray(cone.rec_idx, dtype=np.int32),
                      wire_ptr=w_ptr, wire_zsplit=w_zsplit, wire_sink=w_sink,
                      wire_rise_ps=w_rise, wire_fall_ps=w_fall)


def thresholds(sub: Substrate, ffid: int, capture_edge_ps: int) -> Tuple[int, int, int, int]:
    """(enter0, leave0, enter1, leave1): a pulse away from fault-free value
    ``base`` overlaps the FF's sampling window iff it is present at the D pin at
    some time in (leave_base, enter_base).  base 0 means the pulse is a 0->1->0
    excursion: it enters with a rising D and leaves with a falling D."""
    ff = sub.ffs[ffid]
    enter0 = capture_edge_ps + ff.hold_rise_ps
    leave0 = capture_edge_ps - ff.setup_fall_ps
    enter1 = capture_edge_ps + ff.hold_fall_ps
    leave1 = capture_edge_ps - ff.setup_rise_ps
    return enter0, leave0, enter1, leave1


def build_capture_plan(sub: Substrate, cone: Cone, capture_edge_ps: int, stats: ConeStats) -> CapturePlan:
    reachable = sub.skeleton[cone.src_net]["reachable_dnets"]
    dn_to_ffids: Dict[str, List[int]] = {}
    ffid_set = set()
    for dn in reachable:
        ffids = [f for f in sub.dnet_to_ffids.get(dn, []) if f in sub.ffs]
        if not ffids:
            stats.dnets_without_observed_ff += 1
            continue
        dn_to_ffids[dn] = ffids
        ffid_set.update(ffids)
    ffids_local = sorted(ffid_set)
    n = len(ffids_local)
    li_of = {f: i for i, f in enumerate(ffids_local)}
    thr_enter = np.zeros((2, n), dtype=np.int64)
    thr_leave = np.zeros((2, n), dtype=np.int64)
    for i, f in enumerate(ffids_local):
        e0, l0, e1, l1 = thresholds(sub, f, capture_edge_ps)
        thr_enter[0, i], thr_leave[0, i], thr_enter[1, i], thr_leave[1, i] = e0, l0, e1, l1

    dp_names: List[str] = []
    dp_slot: List[int] = []
    ptr = [0]
    idx: List[int] = []
    if n:
        for dn in reachable:
            ffids = dn_to_ffids.get(dn)
            if not ffids:
                continue
            slot = cone.rec_idx[cone.net2i[dn]]
            if slot < 0:
                continue
            dp_names.append(dn)
            dp_slot.append(int(slot))
            idx.extend(li_of[f] for f in ffids)
            ptr.append(len(idx))
    li = np.arange(n, dtype=np.int32)
    return CapturePlan(ffids=np.asarray(ffids_local, dtype=np.int32), thr_enter=thr_enter,
                       thr_leave=thr_leave, dp_rec_slot=np.asarray(dp_slot, dtype=np.int32),
                       dp_ptr=np.asarray(ptr, dtype=np.int32), dp_idx=np.asarray(idx, dtype=np.int32),
                       dp_dnet_names=dp_names, word_index=(li >> 6).astype(np.int32),
                       word_mask=(np.uint64(1) << (li.astype(np.uint64) & np.uint64(63))).astype(np.uint64),
                       n_words=(n + 63) // 64)
