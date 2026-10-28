"""The capture window of every observed FF in every sampled cycle.

The SDF can give a flip-flop several SETUP (or HOLD) entries for one data edge,
each with a COND on the cell's pins.  In a cycle, the check is the largest value
among the entries whose COND holds in the recorded state of that cycle, and
0 ps when none holds.  An entry without a COND always holds.  Pin values come
from the recorded nets, or from the netlist for a pin tied to 1'b0 / 1'b1.  A
pin without a recorded value satisfies no COND.  A check with a COND that is
not made of pin == value terms (setfi.build.timing.compile_cond_to_conjs) keeps
its first listed entry.  Holds below 0 count as 0, as in load_ff_thresholds.

The result is each FF's most frequent window over the cycles and the (cycle, FF)
pairs whose window differs from it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ..build.timing import compile_cond_to_conjs
from .state import read_cycle_bits
from .substrate import Substrate, _check_value, check_group, load_timing_checks, q_ns_to_ps

# the four checks of an FF: (data edge, check), in the order of FFInfo
SLOTS = (("posedge", "SETUP"), ("posedge", "HOLD"), ("negedge", "SETUP"), ("negedge", "HOLD"))
_NONE = np.iinfo(np.int64).min


@dataclass
class Windows:
    """ffids[i] has the window default[i] in most cycles, and exc_enter[j] /
    exc_leave[j] in cycle exc_cycle[j] (FF exc_ff[j]).  A window is
    (enter[base 0], enter[base 1], leave[base 0], leave[base 1])."""
    ffids: np.ndarray            # int64[n_ff]
    default: np.ndarray          # int64[n_ff, 4]
    exc_cycle: np.ndarray        # int32[n]
    exc_ff: np.ndarray           # int32[n]
    exc_enter: np.ndarray        # int64[n, 2]
    exc_leave: np.ndarray        # int64[n, 2]
    n_unread_checks: int         # checks that keep their first entry: COND not understood


def _window(checks: np.ndarray, capture_edge_ps: int) -> np.ndarray:
    """[..., 4] checks (setup rise, hold rise, setup fall, hold fall) -> windows."""
    t = np.int64(capture_edge_ps)
    su_r, ho_r, su_f, ho_f = (checks[..., i] for i in range(4))
    return np.stack([t + ho_r, t + ho_f, t - su_f, t - su_r], axis=-1)


class _Plan:
    """The checks of the observed FFs, and the COND terms of those that depend on
    the cycle, as flat arrays."""

    def __init__(self, sub: Substrate, checks_by_ff: Dict, clock_pins: List[str], timeprecision_ps: int):
        self.ffids = np.asarray(sorted(sub.ffs), dtype=np.int64)
        n = self.ffids.size
        self.static = np.zeros((n, 4), dtype=np.int64)   # every check's value when it does not vary
        self.n_unread = 0
        dyn_row: List[int] = []       # per varying check: FF row, slot
        dyn_slot: List[int] = []
        var_check: List[int] = []     # per entry of a varying check
        var_val: List[int] = []
        var_always: List[bool] = []
        conj_var: List[int] = []      # per COND conjunction that can hold
        term_conj: List[int] = []     # per pin == value term
        term_src: List[int] = []
        term_want: List[int] = []
        src_of: Dict[int, int] = {}   # net index -> position in src_nets
        q = lambda v: q_ns_to_ps(v, timeprecision_ps)  # noqa: E731

        for row, ffid in enumerate(self.ffids.tolist()):
            ff = sub.ffs[ffid]
            pins = sub.inst_pins.get(ff.inst, {})
            for slot, (edge, kind) in enumerate(SLOTS):
                cp = check_group(checks_by_ff, ff.inst, edge, ff.d_pin, clock_pins)
                entries = [e for e in (cp or {}).get(kind) or [] if _check_value(e) is not None]
                first = (ff.setup_rise_ps, ff.hold_rise_ps, ff.setup_fall_ps, ff.hold_fall_ps)[slot]
                conds = [e.get("enable_cond") for e in entries]
                conjs = [compile_cond_to_conjs(c) if c else None for c in conds]
                if not entries or any(c and cj is None for c, cj in zip(conds, conjs)):
                    self.n_unread += bool(entries)
                    self.static[row, slot] = first
                    continue
                vals = [q(_check_value(e)) for e in entries]
                if not any(conds):
                    v = max(vals)
                    self.static[row, slot] = max(0, v) if kind == "HOLD" else v
                    continue
                check = len(dyn_row)
                dyn_row.append(row)
                dyn_slot.append(slot)
                for cj, v in zip(conjs, vals):
                    var = len(var_val)
                    var_check.append(check)
                    var_val.append(v)
                    var_always.append(cj is None)
                    for conj in cj or ():
                        terms = []
                        possible = True
                        for pin, want in conj.items():
                            net = pins.get(pin, "")
                            if net in ("1'b0", "1'b1"):
                                possible = possible and int(net[-1]) == want
                                continue
                            idx = sub.net_to_idx.get(net)
                            if idx is None:
                                possible = False     # no recorded value: the term never holds
                                continue
                            terms.append((src_of.setdefault(idx, len(src_of)), want))
                        if not possible:
                            continue
                        if not terms:
                            var_always[var] = True
                            continue
                        c = len(conj_var)
                        conj_var.append(var)
                        for s, w in terms:
                            term_conj.append(c)
                            term_src.append(s)
                            term_want.append(w)

        i64 = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
        self.dyn_row, self.dyn_slot = i64(dyn_row), i64(dyn_slot)
        self.dyn_is_hold = np.isin(self.dyn_slot, [1, 3])
        self.var_check, self.var_val = i64(var_check), i64(var_val)
        self.var_always = np.asarray(var_always, dtype=bool)
        self.conj_var = i64(conj_var)
        self.term_conj, self.term_src = i64(term_conj), i64(term_src)
        self.term_want = np.asarray(term_want, dtype=np.uint8)
        self.src_nets = i64(sorted(src_of, key=src_of.get))
        self.dyn_ffs = np.unique(self.dyn_row)              # rows of the FFs whose window varies

    def checks(self, src_vals: np.ndarray) -> np.ndarray:
        """[n_ff, 4] check values (ps) for the recorded values of src_nets."""
        ok = src_vals[self.term_src] == self.term_want
        conj_ok = np.bincount(self.term_conj[~ok], minlength=self.conj_var.size) == 0
        var_ok = self.var_always.copy()
        var_ok[self.conj_var[conj_ok]] = True
        val = np.full(self.dyn_row.size, _NONE, dtype=np.int64)
        np.maximum.at(val, self.var_check[var_ok], self.var_val[var_ok])
        val[val == _NONE] = 0                                # no entry holds: no check
        val = np.where(self.dyn_is_hold, np.maximum(val, 0), val)
        out = self.static.copy()
        out[self.dyn_row, self.dyn_slot] = val
        return out


def campaign_windows(sub: Substrate, stage_dir: str, state_dir: str, cycles: List[int],
                     clock_pins: List[str], capture_edge_ps: int, timeprecision_ps: int = 1) -> Windows:
    checks_by_ff = load_timing_checks(os.path.join(stage_dir, "timing_checks.json"))
    plan = _Plan(sub, checks_by_ff, clock_pins, timeprecision_ps)
    default = _window(plan.static, capture_edge_ps)
    z = np.zeros(0, dtype=np.int32)
    zz = np.zeros((0, 2), dtype=np.int64)
    if plan.dyn_row.size == 0 or not cycles:
        return Windows(plan.ffids, default, z, z, zz, zz, plan.n_unread)

    # the recorded values the CONDs read, one packed row per cycle
    n_nets = len(sub.idx_to_net)
    n_src = plan.src_nets.size
    packed = [np.packbits(read_cycle_bits(state_dir, c, n_nets)[plan.src_nets]) for c in cycles]
    rows = plan.dyn_ffs

    def windows_of(k: int) -> np.ndarray:
        vals = np.unpackbits(packed[k], count=n_src)
        return _window(plan.checks(vals)[rows], capture_edge_ps)

    # each FF's most frequent window (the majority vote finds it when one window
    # holds in more than half of the cycles), then the cycles that differ from it
    cand = windows_of(0)
    count = np.ones(rows.size, dtype=np.int64)
    for k in range(1, len(cycles)):
        w = windows_of(k)
        same = (w == cand).all(axis=1)
        take = (count == 0) & ~same
        cand[take] = w[take]
        count = np.where(same | take, count + 1, count - 1)
    default[rows] = cand
    exc = []
    for k, c in enumerate(cycles):
        w = windows_of(k)
        diff = np.flatnonzero((w != cand).any(axis=1))
        if diff.size:
            exc.append((np.full(diff.size, c, np.int32), plan.ffids[rows[diff]].astype(np.int32), w[diff]))
    if not exc:
        return Windows(plan.ffids, default, z, z, zz, zz, plan.n_unread)
    w = np.concatenate([e[2] for e in exc])
    return Windows(plan.ffids, default, np.concatenate([e[0] for e in exc]),
                   np.concatenate([e[1] for e in exc]), w[:, :2].copy(), w[:, 2:].copy(), plan.n_unread)
