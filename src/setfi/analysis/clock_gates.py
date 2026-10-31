"""Rewrite a pattern table so that clock gates act on the flip-flops they clock.

With observe.clock_gates = "expand" a clock gate is observed through its enable: a
pattern (c, F) that holds gate g says that the SET changes what g latches at the edge
T that ends cycle c.  That changes which flip-flops get an edge at T.  From the
recorded state of cycle c, for each flip-flop f:

    E0, E1   does f get an edge at T without / with the SET: the product of the
             enables on its clock path, each enable inverted if F holds that gate

    E0 E1    f flips at T when
    1  1     f in F
    0  0     never: f has no edge
    0  1     (D XOR f in F) != Q      the extra edge captures D, under the SET
    1  0     D != Q                   the edge is lost, so the SET on D does not matter

Here D stands for the value the edge would load: the data pin, passed through what the
cell's model puts in front of it (a scan mux, a synchronous clear).

The result F' holds flip-flops only.  Nothing else of the SET remains after T, because
the gate's latch is transparent again in the low phase of the next cycle.  Grid points
with an empty F' belong to no pattern.  Equal (cycle, F') are merged, with their runs
joined where they touch: the table is the one the definition in
setfi.analysis.patterns gives for F'.

What F' cannot express is counted in the report (see EXCEPTIONS) and left as it is.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict, List, Tuple

import numpy as np

from ..build.clock_gates import STORED_VALUE, TIED_1
from ..campaign.state import read_cycle_bits, read_cycle_unknown
from .patterns import Patterns

REPORT_FORMAT = "setfi-clock-gate-expansion/1"
GATES_FILE = "clock_gates.json"

# Counted in patterns and in grid points.
EXCEPTIONS = {
    "blackbox_clock": "a gate that gains or loses an edge has a black box in its clock fan-out: "
                      "that is a memory operation more or less, not a flip-flop flip",
    "other_edge": "a gate that gains or loses an edge also clocks cells that do not sample its "
                  "rising edge; they are left as they are",
    "unknown_value": "an enable, D or Q that the result depends on was recorded as X or Z, and "
                     "was read as 0",
    "unobserved_flip": "a flip-flop that is not observed would flip, and is not in F'",
    "async_active": "a flip-flop in F' from a gate has its asynchronous pin active in the "
                    "recorded state, so it would not flip",
}


def load_clock_gates(build_dir: str) -> Dict[str, Any]:
    with open(os.path.join(build_dir, GATES_FILE), encoding="utf-8") as f:
        return json.load(f)


def clock_gate_ffs(gates: Dict[str, Any]) -> List[int]:
    """The FF ids of the gates: observed for the expansion, not flip-flops of the results."""
    return [int(r["ffid"]) for r in gates["gates"]]


class _Model:
    """clock_gates.json as arrays.  Gates are in the file's order, parents first."""

    def __init__(self, data: Dict[str, Any], observed: np.ndarray):
        rows = data["gates"]
        index = {r["inst"]: i for i, r in enumerate(rows)}
        self.rows = rows
        self.n = len(rows)
        self.parent = [index[r["parent"]] if r["parent"] else -1 for r in rows]
        self.children = [[index[c] for c in r["gates"]] for r in rows]
        self.enable = np.asarray([r["enable_net_idx"] for r in rows], dtype=np.int64)
        self.test = np.asarray([-1 if r["test_net_idx"] is None else r["test_net_idx"] for r in rows],
                               dtype=np.int64)
        self.ffid = np.asarray([r["ffid"] for r in rows], dtype=np.int64)
        ids, gate, d, q, inv, start = [], [], [], [], [], [0]
        self.async_rows: List[Tuple[int, int, int]] = []
        by_table: Dict[str, Tuple[List[int], List[List[int]]]] = {}
        for i, r in enumerate(rows):
            for f in sorted(r["ffs"]):
                e = data["ffs"][str(f)]
                self.async_rows.extend((len(ids), net, level) for net, level in e["async"])
                if "next_nets" in e:
                    entry = by_table.setdefault(e["celltype"], ([], []))
                    entry[0].append(len(ids))
                    entry[1].append(e["next_nets"])
                ids.append(f)
                gate.append(i)
                d.append(e["d_net_idx"])
                q.append(e["q_net_idx"])
                inv.append(e["q_inverted"])
            start.append(len(ids))
        self.start = start
        # one entry per cell type with a next-value table: (members, their nets [n, pins], table)
        self.tables = [(np.asarray(m, dtype=np.int64), np.asarray(n, dtype=np.int64).reshape(len(m), -1),
                        np.asarray(data["next_tables"][c], dtype=np.uint8)) for c, (m, n) in by_table.items()]
        self.m_id = np.asarray(ids, dtype=np.int64)
        self.m_gate = np.asarray(gate, dtype=np.int64)
        self.m_d = np.asarray(d, dtype=np.int64)
        self.m_q = np.asarray(q, dtype=np.int64)
        self.m_inv = np.asarray(inv, dtype=np.uint8)
        top = int(max([observed.max(initial=-1), self.ffid.max(initial=-1), self.m_id.max(initial=-1)])) + 1
        self.gate_of = np.full(top, -1, dtype=np.int64)
        self.gate_of[self.ffid] = np.arange(self.n)
        self.names = {int(k): v["inst"] for k, v in data["ffs"].items() if v.get("observed", True)}
        self.has_blackbox = [bool(r["blackbox_pins"]) for r in rows]
        self.has_other_edge = [bool(r["other_edge"]) for r in rows]

    def state(self, bits: np.ndarray, unknown: np.ndarray) -> Dict[str, np.ndarray]:
        has_test = self.test >= 0
        test_on = np.zeros(self.n, dtype=bool)
        test_on[has_test] = bits[self.test[has_test]] > 0
        enable = (bits[self.enable] > 0) | test_on
        enable_x = unknown[self.enable] > 0
        enable_x[has_test] |= unknown[self.test[has_test]] > 0
        edge = enable.copy()
        edge_x = enable_x.copy()
        for g, p in enumerate(self.parent):
            if p >= 0:
                edge[g] &= edge[p]
                edge_x[g] |= edge_x[p]
        clocked = np.ones(self.gate_of.size, dtype=bool)
        clocked[self.m_id[~edge[self.m_gate]]] = False
        ff_x = np.zeros(self.gate_of.size, dtype=bool)
        ff_x[self.m_id[edge_x[self.m_gate]]] = True
        d_x = (unknown[self.m_d] > 0) | (unknown[self.m_q] > 0)
        reset = np.zeros(self.m_id.size, dtype=bool)
        for m, net, level in self.async_rows:
            reset[m] |= bits[net] == level
        q = bits[self.m_q] ^ self.m_inv
        # what the edge loads for data pin 0 and 1
        load0 = np.zeros(self.m_id.size, dtype=np.uint8)
        load1 = np.ones(self.m_id.size, dtype=np.uint8)
        for members, nets, table in self.tables:
            index = np.zeros(members.size, dtype=np.int64)
            for pin in range(nets.shape[1]):
                col = nets[:, pin]
                wired = col >= 0
                v = (col == TIED_1).astype(np.int64)
                v[wired] = bits[col[wired]]
                v[col == STORED_VALUE] = q[members[col == STORED_VALUE]]
                d_x[members[wired]] |= unknown[col[wired]] > 0
                index |= v << pin
            load0[members] = table[index * 2]
            load1[members] = table[index * 2 + 1]
        return {"test_on": test_on, "enable": enable, "enable_x": enable_x, "edge": edge, "clocked": clocked,
                "ff_x": ff_x, "d": bits[self.m_d], "load0": load0, "load1": load1, "q": q, "d_x": d_x,
                "reset": reset}


class _Counts:
    def __init__(self):
        self.counts: Dict[str, List[int]] = {}

    def add(self, name: str, grid_points: int) -> None:
        c = self.counts.setdefault(name, [0, 0])
        c[0] += 1
        c[1] += grid_points

    def as_dict(self) -> Dict[str, Dict[str, int]]:
        return {k: {"patterns": v[0], "grid_points": v[1]} for k, v in sorted(self.counts.items())}


def _flips_of_gates(model: _Model, st: Dict[str, np.ndarray], ids: np.ndarray, flipped: List[int],
                    observed: np.ndarray, seen: np.ndarray, mult: int, cnt: _Counts,
                    by_gate: Dict[str, List[int]]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """For a pattern whose F holds the gates `flipped` and the flip-flops `ids`: (flip-flops
    that flip because a gate gained or lost an edge, flip-flops of those gates, whether
    the result depends on an unknown value)."""
    active = set(flipped)
    affected = set()
    todo = list(flipped)
    while todo:
        g = todo.pop()
        if g not in affected:
            affected.add(g)
            todo.extend(model.children[g])
    edge1: Dict[int, bool] = {}
    for g in sorted(affected):                # parents first
        p = model.parent[g]
        above = True if p < 0 else edge1.get(p, bool(st["edge"][p]))
        edge1[g] = bool(st["enable"][g] ^ (g in active)) and above
    changed = [g for g in sorted(affected) if edge1[g] != bool(st["edge"][g])]
    none = np.zeros(0, dtype=np.int64)
    if not changed:
        cnt.add("no_edge_change", mult)
        return none, none, False
    seen[ids] = True
    flips, group = [], []
    unknown = unobserved = resets = blackbox = other = False
    for g in changed:
        a, b = model.start[g], model.start[g + 1]
        members = model.m_id[a:b]
        d = st["d"][a:b]
        if not st["edge"][g]:                 # an extra edge: D under the SET
            d = d ^ seen[members].astype(np.uint8)
        flip = np.where(d > 0, st["load1"][a:b], st["load0"][a:b]) != st["q"][a:b]
        visible = observed[members]
        flips.append(members[flip & visible])
        group.append(members)
        unobserved |= bool((flip & ~visible).any())
        unknown |= bool(st["enable_x"][g]) or bool(st["d_x"][a:b].any())
        resets |= bool((flip & st["reset"][a:b]).any())
        if model.has_blackbox[g]:
            blackbox = True
            c = by_gate.setdefault(model.rows[g]["inst"], [0, 0])
            c[0] += 1
            c[1] += mult
        other |= model.has_other_edge[g]
    seen[ids] = False
    cnt.add("edge_change", mult)
    for name, hit in (("blackbox_clock", blackbox), ("other_edge", other), ("unobserved_flip", unobserved),
                      ("async_active", resets)):
        if hit:
            cnt.add(name, mult)
    return np.concatenate(flips), np.concatenate(group), unknown


def expand_clock_gates(pt: Patterns, gates: Dict[str, Any], state_dir: str,
                       log=print) -> Tuple[Patterns, Dict[str, Any]]:
    """The table of F' (see the module docstring) and the report of what was done."""
    n_nets = int(gates["n_nets"])
    in_table = {int(i): str(n) for i, n in zip(pt.ff_ids.tolist(), pt.ff_instances.tolist())}
    model = _Model(gates, pt.ff_ids)
    gate_ids = set(model.ffid.tolist())
    # flip-flops an edge can flip have no capture window of their own to check, so a
    # flip-flop left out of the records for lack of one can still be in F'
    ff_names = {i: n for i, n in in_table.items() if i not in gate_ids}
    ff_names.update(model.names)
    observed = np.zeros(model.gate_of.size, dtype=bool)
    observed[list(ff_names)] = True
    no_window = [r["inst"] for r in model.rows if r["ffid"] not in in_table]

    cnt = _Counts()
    by_gate: Dict[str, List[int]] = {}
    keys: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    new_of_old = np.full(pt.count, -1, dtype=np.int64)
    seen = np.zeros(model.gate_of.size, dtype=bool)
    in_group = np.zeros(model.gate_of.size, dtype=bool)
    ptr = pt.pattern_ff_ptr
    cycles, first = np.unique(pt.pattern_cycle, return_index=True)
    bounds = first.tolist() + [pt.count]
    for ci, cycle in enumerate(cycles.tolist()):
        a, b = bounds[ci], bounds[ci + 1]
        st = model.state(read_cycle_bits(state_dir, cycle, n_nets),
                         read_cycle_unknown(state_dir, cycle, n_nets))
        ffs = pt.pattern_ffs[ptr[a]:ptr[b]].astype(np.int64)
        off = (ptr[a:b + 1] - ptr[a]).tolist()
        is_gate = model.gate_of[ffs] >= 0
        keep = st["clocked"][ffs] & ~is_gate
        has_gate = np.add.reduceat(is_gate.astype(np.int64), off[:-1]) > 0
        has_x = np.add.reduceat(st["ff_x"][ffs].astype(np.int64), off[:-1]) > 0
        lost = np.add.reduceat((~st["clocked"][ffs] & ~is_gate).astype(np.int64), off[:-1]) > 0
        for i in range(b - a):
            p = a + i
            lo, hi = off[i], off[i + 1]
            mult = int(pt.pattern_multiplicity[p])
            if lost[i]:
                cnt.add("no_fault_free_edge", mult)
            unknown = bool(has_x[i])
            kept = ffs[lo:hi][keep[lo:hi]]
            if has_gate[i]:
                cnt.add("gate_in_F", mult)
                f = ffs[lo:hi]
                named = model.gate_of[f[is_gate[lo:hi]]].tolist()
                flipped = [g for g in named if not st["test_on"][g]]
                if len(flipped) < len(named):
                    cnt.add("test_enable_on", mult)
                unknown |= any(st["enable_x"][g] for g in named)
                if flipped:
                    flips, group, x = _flips_of_gates(model, st, f[~is_gate[lo:hi]], flipped, observed, seen,
                                                      mult, cnt, by_gate)
                    unknown |= x
                    if group.size:
                        in_group[group] = True
                        kept = np.union1d(kept[~in_group[kept]], flips)
                        in_group[group] = False
            if unknown:
                cnt.add("unknown_value", mult)
            if kept.size == 0:
                cnt.add("empty", mult)
                continue
            key = (cycle, tuple(kept.tolist()))
            new_of_old[p] = keys.setdefault(key, len(keys))
    out = _rebuild(pt, keys, new_of_old, ff_names)
    report: Dict[str, Any] = {
        "format": REPORT_FORMAT,
        "patterns_in": pt.count, "grid_points_in": int(pt.pattern_multiplicity.sum()),
        "patterns_out": out.count, "grid_points_out": int(out.pattern_multiplicity.sum()),
        "counts": cnt.as_dict(),
        "exceptions": EXCEPTIONS,
        "blackbox_clock_by_gate": {k: {"patterns": v[0], "grid_points": v[1]}
                                   for k, v in sorted(by_gate.items())},
        "gates_without_capture_window": no_window,
        "build": gates["summary"],
    }
    log(f"[clock gates] {report['patterns_in']:,} patterns -> {report['patterns_out']:,}; "
        + ", ".join(f"{k} {v['patterns']:,}" for k, v in report["counts"].items()))
    return out, report


def _rebuild(pt: Patterns, keys: Dict[Tuple[int, Tuple[int, ...]], int], new_of_old: np.ndarray,
             ff_names: Dict[int, str]) -> Patterns:
    ordered = sorted(keys)
    rank = np.zeros(len(ordered), dtype=np.int64)
    for i, key in enumerate(ordered):
        rank[keys[key]] = i
    old = np.repeat(np.arange(pt.count, dtype=np.int64), np.diff(pt.pattern_run_ptr))
    target = new_of_old[old]
    live = target >= 0
    pat = rank[target[live]]
    site, width = pt.run_site[live], pt.run_width_ps[live]
    lo, hi = pt.run_k_lo[live], pt.run_k_hi[live]
    order = np.lexsort((lo, width, site, pat))
    pat, site, width, lo, hi = pat[order], site[order], width[order], lo[order], hi[order]
    # one trial's ranges that now give one F' and touch are one run
    if pat.size:
        same = (pat[1:] == pat[:-1]) & (site[1:] == site[:-1]) & (width[1:] == width[:-1])
        join = same & (lo[1:] == hi[:-1] + 1)
        start = np.concatenate([[True], ~join])
        last = np.concatenate([~join, [True]])
        pat, site, width, lo, hi = pat[start], site[start], width[start], lo[start], hi[last]
    n = len(ordered)
    length = (hi - lo + 1).astype(np.int64)
    first_site = np.ones(pat.size, dtype=bool)
    first_site[1:] = (pat[1:] != pat[:-1]) | (site[1:] != site[:-1])
    sizes = [len(k[1]) for k in ordered]
    ids = sorted(ff_names)
    return replace(
        pt,
        ff_ids=np.asarray(ids, dtype=np.int32),
        ff_instances=np.asarray([ff_names[i] for i in ids], dtype=str),
        pattern_cycle=np.asarray([k[0] for k in ordered], dtype=np.int32),
        pattern_ff_ptr=np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)]).astype(np.int64),
        pattern_ffs=np.asarray([f for k in ordered for f in k[1]], dtype=np.int32),
        pattern_multiplicity=np.bincount(pat, weights=length, minlength=n).astype(np.int64),
        pattern_n_sites=np.bincount(pat[first_site], minlength=n).astype(np.int32),
        pattern_run_ptr=np.concatenate([[0], np.cumsum(np.bincount(pat, minlength=n),
                                                       dtype=np.int64)]).astype(np.int64),
        run_site=site.astype(np.int32), run_width_ps=width.astype(np.int32),
        run_k_lo=lo.astype(np.int32), run_k_hi=hi.astype(np.int32))
