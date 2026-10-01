"""Graph part of the build: the static propagation graph.

From the elaborated netlist and the library's pin directions this builds

  * one combinational edge per (combinational instance, input pin -> output pin);
  * the FF index (every recognised sequential instance with a connected data pin);
  * the injection sites: nets in the one-cycle forward region (reachable from an FF
    output and able to reach an FF data input), plus FF D nets, minus FF Q nets
    (see ``BuildSpec.exclude_*_from_src``);
  * for every site its forward cone up to (and through dual-use) FF D nets, in a
    deterministic topological order, with the FF D nets it reaches and whether it
    reconverges.

Cones are computed once per "keep" node of a contracted graph (chains of
single-fanin/single-fanout nets collapsed to super-edges) by a reverse-topological
DP over sparse bitsets, then materialised per site.  A sampled self-check rebuilds
cones by plain BFS and requires an exact match.
"""
from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from .celllib import Cell
from .netlist import Instance
from .spec import BuildError

log = logging.getLogger("setfi.build")

# Safety limit on the chain follow and on the size of one cone.
CHAIN_GUARD = 2_000_000


@dataclass
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


@dataclass
class SuperEdge:
    seid: int
    src_net: str
    dst_net: str
    seg_eids: List[int]


# ============================================================
# Indices
# ============================================================
def build_pin_to_net_map(instances: Dict[str, Instance]) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for instname, inst in instances.items():
        for pin, net in inst.pin2net.items():
            mp[f"{instname}.{pin}"] = net
    return mp


def build_net_index(nets: Set[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    net_to_idx = {n: i for i, n in enumerate(sorted(nets))}
    idx_to_net = {i: n for n, i in net_to_idx.items()}
    return net_to_idx, idx_to_net


def build_inst_index(instances: Dict[str, Instance]) -> Tuple[Dict[str, int], List[str]]:
    idx_to_inst: List[str] = sorted(instances.keys())
    inst_to_idx: Dict[str, int] = {name: i for i, name in enumerate(idx_to_inst)}
    return inst_to_idx, idx_to_inst


def _expand_bus_expr(expr: str) -> List[str]:
    """Expand a Verilog bus expression to individual net names.

    Handles:  "slave_rdata[479:448]" -> [slave_rdata[448], ..., slave_rdata[479]]
              "{n1, n2, n3}"         -> [n1, n2, n3]
              "single_net"           -> [single_net]
    """
    expr = expr.strip()
    if expr.startswith("{") and expr.endswith("}"):
        inner = expr[1:-1]
        return [t.strip() for t in inner.split(",") if t.strip()]
    m = re.match(r"^(\w+)\[(\d+):(\d+)\]$", expr)
    if m:
        base, hi, lo = m.group(1), int(m.group(2)), int(m.group(3))
        lo2, hi2 = min(lo, hi), max(lo, hi)
        return [f"{base}[{i}]" for i in range(lo2, hi2 + 1)]
    return [expr]


# ============================================================
# Graph build + FF index
# ============================================================
@dataclass
class RawGraph:
    edges: List[Edge]
    adj: Dict[str, List[int]]
    nets: Set[str]
    ffid_to_info: Dict[int, Dict[str, Any]]
    dnet_to_ffids: Dict[str, List[int]]
    qnet_to_ffids: Dict[str, List[int]]
    all_stop_nets: Set[str]
    all_q_nets: Set[str]
    target_stop_dnets: Set[str]
    ff_order: List[str]
    missing_ff: List[str]
    unknown_types: Dict[str, int]
    non_lib_connected_nets: Set[str]


def build_graph(instances: Dict[str, Instance], lib: Dict[str, Cell],
                submodules: Set[str] = frozenset()) -> RawGraph:
    """``submodules``: cell types that are modules of the netlist.  Their contents are
    elaborated into ``instances`` already, so their instances are hierarchy, not
    black boxes."""
    edges: List[Edge] = []
    adj: Dict[str, List[int]] = defaultdict(list)
    nets: Set[str] = set()
    unknown_types: Dict[str, int] = defaultdict(int)
    non_lib_connected_nets: Set[str] = set()   # all nets touching a non-lib instance

    # Pass 1: combinational graph
    eid = 0
    for instname, inst in instances.items():
        if inst.celltype in submodules:
            continue
        cell = lib.get(inst.celltype)
        if cell is None:
            unknown_types[inst.celltype] += 1
            for pin_expr in inst.pin2net.values():
                non_lib_connected_nets.update(_expand_bus_expr(pin_expr))
            continue
        if cell.is_sequential:
            continue
        for outpin in cell.outputs:
            if outpin not in inst.pin2net:
                continue
            out_net = inst.pin2net[outpin]
            nets.add(out_net)
            for inpin in cell.inputs:
                if inpin not in inst.pin2net:
                    continue
                in_net = inst.pin2net[inpin]
                nets.add(in_net)
                edges.append(Edge(
                    eid=eid, src_net=in_net, dst_net=out_net,
                    inst=instname, celltype=inst.celltype,
                    inpin=inpin, outpin=outpin,
                    mask_key=f"{inst.celltype}:{inpin}->{outpin}",
                    delay_key=f"{instname}:{inpin}->{outpin}",
                ))
                adj[in_net].append(eid)
                eid += 1

    # Pass 2: collect ALL recognized FFs
    all_ff_records: List[Dict[str, Any]] = []
    all_dnet_to_insts: Dict[str, List[str]] = defaultdict(list)
    all_qnet_to_insts: Dict[str, List[str]] = defaultdict(list)
    for instname, inst in instances.items():
        cell = lib.get(inst.celltype)
        if cell is None or not cell.is_sequential:
            continue
        dnet, dp_used = None, None
        for dp in cell.d_pins:
            if dp in inst.pin2net:
                dnet, dp_used = inst.pin2net[dp], dp
                break
        if dnet is None:
            continue
        qnets: List[Tuple[str, str]] = []
        for qp in cell.q_pins:
            if qp in inst.pin2net:
                qnet = inst.pin2net[qp]
                qnets.append((qp, qnet))
                all_qnet_to_insts[qnet].append(instname)
                nets.add(qnet)
        nets.add(dnet)
        all_dnet_to_insts[dnet].append(instname)
        all_ff_records.append({
            "ff_inst": instname, "celltype": inst.celltype,
            "d_pin": dp_used, "d_net": dnet,
            "q_pins": [qp for qp, _ in qnets],
            "q_nets": [qn for _, qn in qnets],
        })

    # Pass 3: FF ids.  Order = all recognised sequential instances, sorted by name
    # (a sequential instance with no connected data pin is listed as "missing").
    ff_order = sorted(inst for inst, obj in instances.items()
                      if lib.get(obj.celltype) is not None
                      and lib[obj.celltype].is_sequential)
    all_ff_by_inst = {rec["ff_inst"]: rec for rec in all_ff_records}
    missing_ff = [inst for inst in ff_order if inst not in all_ff_by_inst]
    found_ff = [all_ff_by_inst[inst] for inst in ff_order if inst in all_ff_by_inst]

    ffid_to_info: Dict[int, Dict[str, Any]] = {}
    dnet_to_ffids: Dict[str, List[int]] = defaultdict(list)
    qnet_to_ffids: Dict[str, List[int]] = defaultdict(list)
    target_stop_dnets: Set[str] = set()
    for ffid, rec in enumerate(found_ff):
        ffid_to_info[ffid] = {
            "ff_inst": rec["ff_inst"], "celltype": rec["celltype"],
            "d_pin": rec["d_pin"], "d_net": rec["d_net"],
        }
        dnet_to_ffids[rec["d_net"]].append(ffid)
        target_stop_dnets.add(rec["d_net"])
        for qn in rec["q_nets"]:
            qnet_to_ffids[qn].append(ffid)

    return RawGraph(
        edges=edges, adj=adj, nets=nets,
        ffid_to_info=ffid_to_info, dnet_to_ffids=dnet_to_ffids, qnet_to_ffids=qnet_to_ffids,
        all_stop_nets=set(all_dnet_to_insts.keys()), all_q_nets=set(all_qnet_to_insts.keys()),
        target_stop_dnets=target_stop_dnets, ff_order=ff_order, missing_ff=missing_ff,
        unknown_types=dict(unknown_types), non_lib_connected_nets=non_lib_connected_nets,
    )


# ============================================================
# One-cycle window utilities
# ============================================================
def make_one_cycle_adj(adj: Dict[str, List[int]], stop: Set[str]) -> Dict[str, List[int]]:
    # Keep outgoing edges for ALL nets that have edges, including `stop` (FF.D)
    # nets with real combinational fanout.  Filtering `u not in stop` would drop the
    # downstream fanout of **dual-use FF.D nets** -- nets that are captured by an FF
    # *and* continue combinationally to more gates (common on processor pipelines:
    # e.g. `master_addr[29]` feeds both a CSR FF and `ex_reg_wdata_o[29]` which fans
    # to 31 GPR FFs).  Dropping the fanout made every src_net whose cone passed
    # through a dual-use FF.D under-report its reachable FF set.  The forward walk and
    # conedb logic handle the stop-vs-passthrough decision explicitly; raw edges stay
    # intact.
    return {u: eids for u, eids in adj.items() if eids}


def compute_can_reach_stop(adj1: Dict[str, List[int]], edges: List[Edge],
                           stop: Set[str]) -> Set[str]:
    radj: Dict[str, List[str]] = defaultdict(list)
    for u, eids in adj1.items():
        for eid in eids:
            radj[edges[eid].dst_net].append(u)
    vis = set(stop)
    q = deque(stop)
    while q:
        v = q.popleft()
        for u in radj.get(v, []):
            if u not in vis:
                vis.add(u)
                q.append(u)
    return vis


def derive_orig_src_nets(q_nets: Set[str], adj1: Dict[str, List[int]], edges: List[Edge],
                         can_reach_stop: Set[str], stop: Set[str]) -> Set[str]:
    orig: Set[str] = set()
    for qn in q_nets:
        if qn not in can_reach_stop:
            continue
        # Q-net is ALSO a dual-use FF.D (Q of some FF, D of another AND has outgoing
        # combinational fanout).  Include qn itself as a forward-walk seed so
        # stage_nets grows through it.  Without this, nets like ex_reg_wdata_o[29]
        # (Q of one pipeline FF, D of 11 GPRs, plus combinational fanout through one
        # inverter) produce empty cones because qn has no adj1 fanin (it's an FF.Q,
        # not a gate output) -- the BFS never visits it unless it's seeded as a root.
        if qn in stop and adj1.get(qn):
            orig.add(qn)
        for eid in adj1.get(qn, []):
            out_net = edges[eid].dst_net
            if out_net in stop:
                # dual-use FF.D: net is captured by an FF (in stop) AND has
                # downstream combinational fanout.  Include it in orig so the forward
                # walk uses its fanout endpoints.  Pure-terminal FF.D (no outgoing
                # fanout) still excluded -- the walk would add nothing new.
                if not adj1.get(out_net):
                    continue
                orig.add(out_net)
                continue
            if out_net in can_reach_stop:
                orig.add(out_net)
    return orig


def dual_use_ffd_seeds(stop_nets: Set[str], adj1: Dict[str, List[int]],
                       can_reach_stop: Set[str],
                       exclude: Iterable[str] = frozenset()) -> Set[str]:
    """FF.D nets that are ALSO combinational sources -- every one must seed the forward region.

    `derive_orig_src_nets` reaches a dual-use FF.D only through some FF.Q's fanout, and
    `forward_region_from_sources` only walks past one it has ALREADY reached.  A dual-use
    FF.D whose entire backward cone contains no FF.Q therefore never enters `stage_nets` --
    while the D-net injection policy makes it INJECTABLE regardless.  Its successors then
    never enter `relevant_nets`, `build_adj_subgraph` drops the edges to them, and the
    declared cone is truncated with nothing to say so.  (Typical case: an FF.D fed only
    by a primary input through buffers that also drives further logic.)

    The predicate is tight so that it changes nothing on graphs without such a net:
      * `"##" not in d`      -- virtual `##D` taps are graph-only markers, not real wires;
      * `adj1.get(d)`        -- a PURE-TERMINAL FF.D adds nothing to the region, so seeding
                               it would change `stage_nets` on designs with no defect;
      * `d in can_reach_stop`-- the same domain filter every other seed passes.
    """
    return {d for d in stop_nets
            if "##" not in d and d in can_reach_stop and adj1.get(d)} - set(exclude)


def forward_region_from_sources(sources: Set[str], adj1: Dict[str, List[int]],
                                edges: List[Edge], can_reach_stop: Set[str],
                                stop: Set[str]) -> Set[str]:
    """Forward-walk from sources; mark nets reachable to some FF.D.

    Terminology:
      - *pure terminal* FF.D: net in `stop` with no combinational outgoing fanout.
        Include in the visited set (so upstream paths see a valid endpoint)
        but do not walk past.
      - *dual-use* FF.D: net in `stop` that *also* fans out to more combinational
        logic.  These nets must be walked past so downstream FFs get discovered -- a
        SET pulse arriving at a dual-use FF.D is simultaneously captured by the FF AND
        propagated forward to every gate sinking on the same wire.

    BFS uses two sets:
      - enqueued: dedup against re-enqueuing (bounds worst-case queue ops)
      - vis:      final reachability result (= stage_nets)
    A discovered dual-use FF.D `v` is both marked in vis AND enqueued, so its own fanout
    is processed; the separate `enqueued` set keeps that from being skipped.
    """
    roots_set = {s for s in sources if s in can_reach_stop and s not in stop}
    dual_use_sources = {s for s in sources
                        if s in stop and s in can_reach_stop and adj1.get(s)}
    roots_set |= dual_use_sources
    vis: Set[str] = set()
    enqueued: Set[str] = set(roots_set)
    q = deque(roots_set)
    while q:
        u = q.popleft()
        vis.add(u)
        for eid in adj1.get(u, []):
            v = edges[eid].dst_net
            if v in stop:
                vis.add(v)                             # record reachable endpoint
                if adj1.get(v) and v not in enqueued:  # dual-use -> continue past
                    enqueued.add(v)
                    q.append(v)
                continue
            if v in can_reach_stop and v not in enqueued:
                enqueued.add(v)
                q.append(v)
    return vis


def build_adj_subgraph(adj1: Dict[str, List[int]], edges: List[Edge],
                       allowed: Set[str]) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for u, eids in adj1.items():
        if u not in allowed:
            continue
        lst = [eid for eid in eids if edges[eid].dst_net in allowed]
        if lst:
            out[u] = lst
    return out


def topo_sort(nets: Set[str], adj: Dict[str, List[int]], edges: List[Edge]) -> List[str]:
    # `sorted(nets)`, not `nets`: a topological order is not unique, and Kahn's choice
    # among the currently-ready nodes is decided by the order they entered the queue --
    # which, seeded from a dict built by iterating a SET OF STRINGS, is hash order and
    # therefore a function of PYTHONHASHSEED.  The rank this produces then orders every
    # cone, so the whole substrate would inherit the nondeterminism.  Sorting the seed
    # makes the chosen order a function of the graph and the names, nothing else.
    indeg = {n: 0 for n in sorted(nets)}
    for _u, eids in adj.items():
        for eid in eids:
            v = edges[eid].dst_net
            if v in indeg:
                indeg[v] += 1
    q = deque([n for n, d in indeg.items() if d == 0])
    topo: List[str] = []
    while q:
        u = q.popleft()
        topo.append(u)
        for eid in adj.get(u, []):
            v = edges[eid].dst_net
            if v not in indeg:
                continue
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != len(indeg):
        cyc = sorted(n for n, d in indeg.items() if d > 0)[:10]
        raise BuildError(f"combinational cycle in the one-cycle subgraph "
                         f"(nets on or behind it, first 10: {cyc})")
    return topo


# ============================================================
# Reconvergence check
# ============================================================
def has_reconvergence_sat2(src: str, cone_topo_nets: List[str], cone_adj_list: List[List],
                           edges: List[Edge]) -> bool:
    cone_adj: Dict[str, List[int]] = {}
    for item in cone_adj_list:
        if not item or len(item) != 2:
            continue
        u, eids = item[0], item[1]
        if eids:
            cone_adj[u] = eids
    reach: Dict[str, int] = {src: 1}
    for u in cone_topo_nets:
        ru = reach.get(u, 0)
        if ru == 0:
            continue
        for eid in cone_adj.get(u, []):
            v = edges[eid].dst_net
            s = reach.get(v, 0) + ru
            if s >= 2:
                if v != src:
                    return True
                reach[v] = 2
            else:
                reach[v] = s
    return False


# ============================================================
# SparseSeidSet -- sparse bitset over superedge ids
# ============================================================
class SparseSeidSet:
    __slots__ = ("blk",)

    def __init__(self, blk: Optional[Dict[int, int]] = None):
        self.blk = blk if blk is not None else {}

    def nblocks(self) -> int:
        return len(self.blk)

    def copy(self) -> "SparseSeidSet":
        return SparseSeidSet(self.blk.copy())

    def add(self, seid: int) -> None:
        b = seid >> 6
        self.blk[b] = self.blk.get(b, 0) | (1 << (seid & 63))

    def ior(self, other: "SparseSeidSet") -> None:
        if not other.blk:
            return
        for b, w in other.blk.items():
            self.blk[b] = self.blk.get(b, 0) | w

    def iter_sorted(self) -> Iterable[int]:
        for b in sorted(self.blk.keys()):
            w = self.blk[b]
            while w:
                lsb = w & -w
                bit = lsb.bit_length() - 1
                yield (b << 6) + bit
                w ^= lsb

    def to_blocks_list(self) -> List[List[int]]:
        return [[b, int(w)] for b, w in sorted(self.blk.items())]


# ============================================================
# Keep + contraction + ConeDB DP
# ============================================================
def compute_fanin_fanout(adj: Dict[str, List[int]], edges: List[Edge], allowed: Set[str]
                         ) -> Tuple[Dict[str, int], Dict[str, int]]:
    fanin = {n: 0 for n in allowed}
    fanout = {n: 0 for n in allowed}
    for u, eids in adj.items():
        if u in allowed:
            fanout[u] = len(eids)
        for eid in eids:
            v = edges[eid].dst_net
            if v in fanin:
                fanin[v] += 1
    return fanin, fanout


def build_keep_set(allowed: Set[str], orig_src: Set[str], stop: Set[str],
                   fanin: Dict[str, int], fanout: Dict[str, int]) -> Set[str]:
    keep = set(stop) | set(orig_src)
    for n in allowed:
        if fanin.get(n, 0) != 1 or fanout.get(n, 0) != 1:
            keep.add(n)
    return keep


def build_contracted(adj: Dict[str, List[int]], edges: List[Edge], keep: Set[str]
                     ) -> Tuple[List[SuperEdge], Dict[str, List[int]]]:
    superedges: List[SuperEdge] = []
    cadj: Dict[str, List[int]] = defaultdict(list)
    seid = 0

    def outdeg(x: str) -> int:
        return len(adj.get(x, []))

    for u, eids in adj.items():
        if u not in keep:
            continue
        for eid0 in eids:
            seg = [eid0]
            v = edges[eid0].dst_net
            while v not in keep and outdeg(v) == 1:
                eid1 = adj[v][0]
                seg.append(eid1)
                v = edges[eid1].dst_net
            superedges.append(SuperEdge(seid=seid, src_net=u, dst_net=v, seg_eids=seg))
            cadj[u].append(seid)
            seid += 1
    return superedges, cadj


def compute_refcount(relevant_keep: Set[str], cadj: Dict[str, List[int]],
                     superedges: List[SuperEdge]) -> Dict[str, int]:
    # `relevant_keep` admits dual-use FF.D (FF.D with combinational fanout); they are
    # counted too so build_conedb's `cones.pop` on refcnt==1 works for them.  Only
    # nets NOT in relevant_keep (= pure-terminal FF.D or pruned nets) are excluded.
    ref: DefaultDict[str, int] = defaultdict(int)
    for u in relevant_keep:
        for seid in cadj.get(u, []):
            v = superedges[seid].dst_net
            if v in relevant_keep:
                ref[v] += 1
    return ref


def build_conedb(relevant_keep: Set[str], topo_rank: Dict[str, int],
                 cadj: Dict[str, List[int]], superedges: List[SuperEdge],
                 stop: Set[str], counts: Counter) -> Dict[str, SparseSeidSet]:
    """Per keep net: the set of super-edges in its forward cone (reverse-topological DP).

    The working set `cones` hands a child's set over to its last parent (refcount 1)
    and mutates it in place, so the result for every net is snapshotted when it is
    produced."""
    order = sorted(relevant_keep, key=lambda n: (topo_rank.get(n, 1 << 60), n), reverse=True)
    refcnt = compute_refcount(relevant_keep, cadj, superedges)
    cones: Dict[str, SparseSeidSet] = {}
    db: Dict[str, SparseSeidSet] = {}
    for u in order:
        outs = cadj.get(u, [])
        if not outs:
            acc = SparseSeidSet()
        else:
            # Dual-use FF.D nets are in relevant_keep and are merged like any keep net,
            # so upstream cones absorb their downstream reach.  Pure-terminal FF.D are
            # not in relevant_keep and fall through.
            base_v, base_size = None, -1
            for seid in outs:
                v = superedges[seid].dst_net
                if v not in relevant_keep:
                    continue
                sv = cones.get(v)
                if sv is None:
                    continue
                if sv.nblocks() > base_size:
                    base_size = sv.nblocks()
                    base_v = v
            if base_v is None:
                acc = SparseSeidSet()
            else:
                acc = cones.pop(base_v) if refcnt.get(base_v, 0) == 1 else cones[base_v].copy()
            for seid in outs:
                v = superedges[seid].dst_net
                if v not in stop and v not in relevant_keep:
                    continue
                acc.add(seid)
                if v in relevant_keep and v != base_v:
                    acc.ior(cones[v])
        cones[u] = acc
        db[u] = SparseSeidSet(dict(sorted(acc.blk.items())))
        for seid in outs:
            v = superedges[seid].dst_net
            if v not in relevant_keep:
                continue
            refcnt[v] -= 1
            if refcnt[v] <= 0:
                cones.pop(v, None)
    return db


# ============================================================
# net_entry + cone materialisation
# ============================================================
def build_net_entry(src_list: List[str], keep: Set[str], stop: Set[str],
                    adj: Dict[str, List[int]], edges: List[Edge], guard: int
                    ) -> Dict[str, Dict[str, Any]]:
    def outdeg(x: str) -> int:
        return len(adj.get(x, []))

    ent: Dict[str, Dict[str, Any]] = {}
    for s in src_list:
        if s in stop:
            # Dual-use FF.D src (in stop AND has combinational fanout in adj): route to
            # "keep" so materialize_cone_expanded picks up its cone entry from conedb
            # (which covers downstream FFs).  Pure-terminal FF.D keeps "stop".
            if adj.get(s):
                ent[s] = {"type": "keep", "entry_keep": s, "prefix_eids": []}
            else:
                ent[s] = {"type": "stop"}
        elif s in keep:
            ent[s] = {"type": "keep", "entry_keep": s, "prefix_eids": []}
        else:
            cur, prefix, steps = s, [], 0
            while True:
                steps += 1
                if steps > guard:
                    raise BuildError(f"chain follow too long: {s}")
                if cur in stop:
                    # Dual-use FF.D (in stop AND has combinational fanout): route to
                    # "chain" so materialize_cone_expanded picks up its downstream cone
                    # from conedb.  Pure-terminal FF.D keeps "chain_stop".  Mirrors the
                    # `s in stop` handling above and the dual-use semantics in
                    # bfs_truth_eids / build_conedb; without it the cone truncates at
                    # the FF.D and the self-check fails.
                    if adj.get(cur):
                        ent[s] = {"type": "chain", "entry_keep": cur, "prefix_eids": prefix}
                    else:
                        ent[s] = {"type": "chain_stop", "stop_net": cur,
                                  "prefix_eids": prefix}
                    break
                if cur in keep:
                    ent[s] = {"type": "chain", "entry_keep": cur, "prefix_eids": prefix}
                    break
                od = outdeg(cur)
                if od == 0:
                    ent[s] = {"type": "dead", "prefix_eids": prefix}
                    break
                if od > 1:
                    ent[s] = {"type": "keep", "entry_keep": cur, "prefix_eids": prefix}
                    break
                eid = adj[cur][0]
                prefix.append(eid)
                cur = edges[eid].dst_net
    return ent


def materialize_cone_expanded(src: str, ent: Dict[str, Any], conedb: Dict[str, SparseSeidSet],
                              superedges: List[SuperEdge], edges: List[Edge],
                              topo_rank: Dict[str, int], max_nodes: int,
                              adj_sub_eids_set: Optional[Set[int]] = None
                              ) -> Tuple[List[str], List[List[Any]]]:
    vis: Set[str] = {src}
    cone_adj: DefaultDict[str, List[int]] = defaultdict(list)

    def _check_eid(eid: int) -> None:
        if adj_sub_eids_set is not None and eid not in adj_sub_eids_set:
            raise BuildError(f"expanded cone uses eid={eid} not in the one-cycle "
                             f"subgraph (src={src})")

    prefix = ent.get("prefix_eids", [])
    cur = src
    for eid in prefix:
        _check_eid(eid)
        cone_adj[cur].append(eid)
        nxt = edges[eid].dst_net
        vis.add(nxt)
        cur = nxt

    if ent.get("type") == "chain_stop":
        # The `n` tiebreak is load-bearing -- see the note at the second sort below.
        topo = sorted(vis, key=lambda n: (topo_rank.get(n, 1 << 60), n))
        adj_list = [[u, sorted(set(cone_adj[u]))] for u in topo if u in cone_adj]
        return topo, adj_list

    if ent.get("type") in ("stop", "dead"):
        return [src], []

    entry_keep = ent.get("entry_keep", src)
    seidset = conedb.get(entry_keep, SparseSeidSet())

    for seid in seidset.iter_sorted():
        se = superedges[seid]
        curr = se.src_net
        vis.add(curr)
        for eid in se.seg_eids:
            _check_eid(eid)
            cone_adj[curr].append(eid)
            nxt = edges[eid].dst_net
            vis.add(nxt)
            curr = nxt

    if len(vis) > max_nodes:
        raise BuildError(f"cone too large: src={src}, nodes={len(vis)}")

    # The tiebreak on the net NAME makes the order a function of the graph alone.
    # Without it, nets sharing a topo_rank keep the set's iteration order, which depends
    # on PYTHONHASHSEED; the order sets the campaign's local net indices and hence the
    # event tie-breaks, so results would differ between two builds of the same design.
    topo = sorted(vis, key=lambda n: (topo_rank.get(n, 1 << 60), n))
    adj_list = [[u, sorted(set(cone_adj[u]))] for u in topo if u in cone_adj]
    return topo, adj_list


def bfs_truth_eids(src: str, adj_sub: Dict[str, List[int]], edges: List[Edge],
                   stop: Set[str]) -> Set[int]:
    """Truth-set BFS matching the conedb semantics: a dual-use FF.D is passed through
    when it has outgoing combinational fanout in adj_sub; a pure-terminal FF.D is not
    walked.  Kept symmetric with forward_region_from_sources + build_conedb."""
    vis_n: Set[str] = {src}
    vis_e: Set[int] = set()
    q = deque([src])
    while q:
        u = q.popleft()
        if u in stop and not adj_sub.get(u):
            continue
        for eid in adj_sub.get(u, []):
            vis_e.add(eid)
            v = edges[eid].dst_net
            if v not in vis_n:
                vis_n.add(v)
                q.append(v)
    return vis_e


def self_check(src_list: List[str], sample_n: int, seed: int,
               net_entry: Dict[str, Dict[str, Any]], conedb: Dict[str, SparseSeidSet],
               superedges: List[SuperEdge], edges: List[Edge], topo_rank: Dict[str, int],
               adj_sub: Dict[str, List[int]], stop: Set[str],
               adj_sub_eids_set: Set[int]) -> int:
    """Rebuild `sample_n` random cones by plain BFS; raise on any difference."""
    if not src_list or sample_n <= 0:
        return 0
    picks = random.Random(seed).sample(src_list, min(sample_n, len(src_list)))
    ok = 0
    for s in picks:
        _topo, adj_list = materialize_cone_expanded(
            s, net_entry[s], conedb, superedges, edges, topo_rank,
            CHAIN_GUARD, adj_sub_eids_set)
        exp_eids: Set[int] = set()
        for _u, eids in adj_list:
            exp_eids.update(eids)
        truth = bfs_truth_eids(s, adj_sub, edges, stop)
        if exp_eids != truth:
            miss = sorted(truth - exp_eids)[:20]
            extra = sorted(exp_eids - truth)[:20]
            raise BuildError(f"cone self-check failed for src={s}: eid mismatch, "
                             f"missing(first20)={miss} extra(first20)={extra}")
        ok += 1
    return ok


# ============================================================
# The stage
# ============================================================
@dataclass
class GraphResult:
    edges: List[Edge]
    nets: Set[str]
    net_to_idx: Dict[str, int]
    idx_to_net: Dict[int, str]
    inst_to_idx: Dict[str, int]
    idx_to_inst: List[str]
    pin_to_net: Dict[str, str]
    ff_index: Dict[str, Any]
    src_list: List[str]
    non_lib_connected_nets: Set[str]
    counts: Dict[str, Any]
    # intermediates (only for keep_intermediates)
    superedges: List[SuperEdge] = field(default_factory=list)
    net_entry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    conedb: Dict[str, SparseSeidSet] = field(default_factory=dict)
    conedb_order: List[str] = field(default_factory=list)
    cone_edge_ids: Set[int] = field(default_factory=set)


def build_stage(instances: Dict[str, Instance], lib: Dict[str, Cell],
                primary_inputs: Set[str], skeleton_path: str, *,
                exclude_q_from_src: bool, exclude_d_from_src: bool,
                submodules: Set[str] = frozenset(),
                self_check_samples: int, self_check_seed: int,
                do_self_check: bool) -> GraphResult:
    """Build the graph.  Writes the site cones (one JSON line per injection site)
    to ``skeleton_path``; everything else is returned."""
    counts: Counter = Counter()
    g = build_graph(instances, lib, submodules)
    edges, adj, nets = g.edges, g.adj, g.nets

    stop_nets = set(g.all_stop_nets)
    q_nets = set(g.all_q_nets)

    adj1 = make_one_cycle_adj(adj, stop_nets)
    can_reach_stop = compute_can_reach_stop(adj1, edges, stop_nets)
    orig_src = derive_orig_src_nets(q_nets, adj1, edges, can_reach_stop, stop_nets)

    # ---- Non-library outputs as forward-region sources ----
    # Non-library instances (memories, black boxes, and the ports of kept wrapper
    # modules) drive library logic but are not driven by any library cell.  Without
    # seeding them the forward region misses every path downstream of these ports.
    # A non-lib output net: touches a non-lib instance, is the source of library-cell
    # edges, is never an edge destination.  They behave like FF.Q nets.
    lib_dst_nets: Set[str] = {edges[eid].dst_net for eids in adj.values() for eid in eids}
    nonlib_output_nets: Set[str] = set()
    for net in g.non_lib_connected_nets:
        if net not in lib_dst_nets and net in adj1 and net in can_reach_stop:
            nonlib_output_nets.add(net)
    if nonlib_output_nets:
        log.info("%d non-library output nets seed the forward region", len(nonlib_output_nets))
        orig_src = orig_src | nonlib_output_nets

    # The forward region is seeded by the INJECTABLE SOURCE SET, not by FF.Q fanout
    # alone -- see dual_use_ffd_seeds().
    d_seeds: Set[str] = set()
    if not exclude_d_from_src:
        d_seeds = dual_use_ffd_seeds(stop_nets, adj1, can_reach_stop,
                                     exclude=nonlib_output_nets)
        new_seeds = d_seeds - orig_src
        if new_seeds:
            log.info("%d dual-use FF.D net(s) added as forward-region sources: %s",
                     len(new_seeds), sorted(new_seeds)[:5])
    stage_nets = forward_region_from_sources(orig_src | d_seeds, adj1, edges,
                                             can_reach_stop, stop_nets)

    src_nets = set(stage_nets)
    if not exclude_d_from_src:
        # FF D-input nets are injection sites.  stage_nets alone does not contain the
        # pure-terminal ones (the forward walk never enters them as roots).  Real
        # D-nets only -- skip virtual ##D taps (graph-only markers, not wires).
        src_nets |= {d for d in stop_nets if "##" not in d}
    if exclude_q_from_src:
        # Preserve nets that are BOTH some FF's Q AND some other FF's D -- functional
        # FF1.Q -> FF2.D direct wires (pipeline stages, synchronisers, retimed FFs).
        # Such a net is a legitimate FF.D injection point even though topologically it
        # is also FF1.Q.  (Not the scan chain: FF1.Q -> FF2.SI is a different pin.)
        # If D nets are excluded too, the overlap cannot help.
        if exclude_d_from_src:
            src_nets -= q_nets
        else:
            src_nets -= (q_nets - stop_nets)
    if exclude_d_from_src:
        src_nets -= stop_nets
    # Non-lib output nets seed the forward region (for complete cones) but are NOT
    # injectable -- they are driven by black-box modules and forcing them would
    # conflict with the driver.
    src_nets -= nonlib_output_nets
    src_list = sorted(src_nets)

    # Invariant: every INJECTABLE net with combinational fanout lies inside the forward
    # region.  If it does not, its declared cone is missing every net reachable only
    # through it, and nothing downstream can tell -- the campaign completes and reports
    # a plausible escape rate against a truncated endpoint set.
    unseeded = {s for s in src_nets if adj1.get(s) and s not in stage_nets}
    if unseeded:
        raise BuildError(
            f"{len(unseeded)} injectable source net(s) with combinational fanout are "
            f"outside the forward region (e.g. {sorted(unseeded)[:5]}); their declared "
            f"cones would be missing every net reachable only through them.")

    # ---- source-only / primary-input audit (informational) ----
    src_only_nets: Set[str] = ({edges[eid].src_net for eids in adj.values() for eid in eids}
                               - lib_dst_nets)
    # flip-flop outputs and primary inputs that drive a data pin directly are sites;
    # anything else not driven by a gate is suspicious
    unexpected_src = (src_nets & src_only_nets) - nonlib_output_nets - q_nets - primary_inputs
    if unexpected_src:
        log.warning("%d injection site(s) are driven by no gate, flip-flop, primary input "
                    "or black box, e.g. %s", len(unexpected_src), sorted(unexpected_src)[:5])

    relevant_nets = set(stage_nets) | set(stop_nets)
    adj_sub = build_adj_subgraph(adj1, edges, relevant_nets)
    topo = topo_sort(relevant_nets, adj_sub, edges)
    topo_rank = {n: i for i, n in enumerate(topo)}

    adj_sub_eids_set: Set[int] = set()
    for eids in adj_sub.values():
        adj_sub_eids_set.update(eids)

    fanin, fanout = compute_fanin_fanout(adj_sub, edges, relevant_nets)
    keep = build_keep_set(relevant_nets, orig_src, stop_nets, fanin, fanout)
    superedges, cadj = build_contracted(adj_sub, edges, keep)
    # Dual-use FF.D: in stop AND has outgoing combinational fanout.  In relevant_keep
    # so build_conedb emits a cone for it -- otherwise any src whose cone passes
    # through one (bypass buses, register-file write-back) gets an empty downstream.
    dual_use_ff_d = {n for n in stop_nets if adj.get(n)}
    relevant_keep = {n for n in keep if n in relevant_nets
                     and (n not in stop_nets or n in dual_use_ff_d)}

    conedb = build_conedb(relevant_keep, topo_rank, cadj, superedges, stop_nets, counts)
    net_entry = build_net_entry(src_list, keep, stop_nets, adj_sub, edges, CHAIN_GUARD)

    n_checked = 0
    if do_self_check:
        n_checked = self_check(src_list, self_check_samples, self_check_seed, net_entry,
                               conedb, superedges, edges, topo_rank, adj_sub, stop_nets,
                               adj_sub_eids_set)

    # ---------- indices ----------
    net_to_idx, idx_to_net = build_net_index(nets)
    inst_to_idx, idx_to_inst = build_inst_index(instances)
    pin_to_net = build_pin_to_net_map(instances)

    ff_index = {
        "ffid_to_info": {
            str(ffid): {**info, "inst_idx": inst_to_idx.get(info["ff_inst"], -1),
                        "d_net_idx": net_to_idx[info["d_net"]]}
            for ffid, info in g.ffid_to_info.items()
        },
        "dnet_to_ffids": {str(net_to_idx[dn]): fids for dn, fids in g.dnet_to_ffids.items()},
        "qnet_to_ffids": {str(net_to_idx[qn]): fids for qn, fids in g.qnet_to_ffids.items()},
        # FF ids number the sequential instances in sorted name order; instances
        # whose data pin is unconnected are listed and get no id.
        "n_sequential": len(g.ff_order),
        "sequential_without_data_pin": g.missing_ff[:200],
    }

    # ---------- expanded skeleton ----------
    with open(skeleton_path, "w", encoding="utf-8") as fsk:
        for src in src_list:
            topo_s, adj_list = materialize_cone_expanded(
                src, net_entry[src], conedb, superedges, edges, topo_rank,
                CHAIN_GUARD, adj_sub_eids_set)
            has_reconv = has_reconvergence_sat2(src, topo_s, adj_list, edges)
            counts["n_sites_reconvergent" if has_reconv else "n_sites_not_reconvergent"] += 1
            dn: Set[str] = set()
            # Self-capture: the loop below walks edges, so it only reaches nets that
            # are some edge's dst and never the src itself.  When the src net IS an
            # FF's D pin, a SET on it is captured by that FF, so add it explicitly.
            if src in stop_nets and src in g.target_stop_dnets:
                dn.add(src)
            for _u, eids in adj_list:
                for eid in eids:
                    v = edges[eid].dst_net
                    if v in stop_nets and v in g.target_stop_dnets:
                        dn.add(v)
            rec = {
                "src_net_idx": net_to_idx[src],
                "cone_topo_net_idxs": [net_to_idx[n] for n in topo_s],
                "cone_adj_idx": [[net_to_idx[u], sorted(set(eids))] for u, eids in adj_list],
                "has_reconv": has_reconv,
                "reachable_dnet_idxs": [net_to_idx[n] for n in sorted(dn)],
            }
            fsk.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts.update({
        "n_nets": len(nets),
        "n_instances": sum(1 for i in instances.values() if i.celltype not in submodules),
        "n_sites": len(src_list),
        "n_flip_flops": len(g.ffid_to_info),
        "n_flip_flops_without_data_pin": len(g.missing_ff),
        "n_gate_arcs": len(edges),
        "n_gate_arcs_between_flip_flops": sum(len(v) for v in adj_sub.values()),
        "n_black_box_instances": sum(g.unknown_types.values()),
        "n_black_box_cell_types": len(g.unknown_types),
        "n_black_box_output_nets": len(nonlib_output_nets),
        "n_ff_data_nets_that_drive_gates": len(dual_use_ff_d),
        "n_primary_inputs": len(primary_inputs),
        "n_black_box_nets": len(g.non_lib_connected_nets),
        "n_nets_driven_by_no_gate": len(src_only_nets),
        "n_sites_driven_by_nothing": len(unexpected_src),
        "n_sites_on_primary_inputs": len(primary_inputs & src_nets),
        "n_sites_on_black_box_nets": len(g.non_lib_connected_nets & src_nets),
        "self_check_samples_ok": n_checked,
    })
    order = sorted(relevant_keep, key=lambda n: (topo_rank.get(n, 1 << 60), n), reverse=True)
    return GraphResult(
        edges=edges, nets=nets, net_to_idx=net_to_idx, idx_to_net=idx_to_net,
        inst_to_idx=inst_to_idx, idx_to_inst=idx_to_inst, pin_to_net=pin_to_net,
        ff_index=ff_index, src_list=src_list,
        non_lib_connected_nets=g.non_lib_connected_nets, counts=dict(counts),
        superedges=superedges, net_entry=net_entry, conedb=conedb, conedb_order=order,
        cone_edge_ids=adj_sub_eids_set,
    )
