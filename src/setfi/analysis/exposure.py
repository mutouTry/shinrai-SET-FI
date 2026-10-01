"""Capture regions and exposure from pulse records.

For one trial k = (cycle, site, width) and one observed FF f, the capture
region A_kf is the set of SET start times s (integer ps within the cycle) for
which f latches a wrong value: the union, over the pulses recorded at f's D pin,
of [leave - t_exit + 1, enter - t_enter - 1] (see ``setfi.records``).

The exposure of the trial is Omega_k = union_f A_kf: the start times at which
at least one FF is upset.  With the SET start uniform over the cycle, the
probability that the SET of trial k upsets at least one FF is |Omega_k| / T,
and |{f : s in A_kf}| is the number of FFs upset together (multi-bit upset).

Start times are counted on the 1-ps lattice of the phase domain, which is
[0, T) by default ("half_open") or [0, T] ("closed", T + 1 points).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..records import INF_PS, Records


def domain_length(period_ps: int, phase_domain: str) -> int:
    if phase_domain == "half_open":
        return int(period_ps)
    if phase_domain == "closed":
        return int(period_ps) + 1
    raise ValueError(f"phase_domain must be 'half_open' or 'closed', got {phase_domain!r}")


def capture_intervals(cols: Dict[str, np.ndarray], enter: np.ndarray, leave: np.ndarray,
                      dom: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per record: half-open integer interval [lo, hi) of SET start times in
    [0, dom) that capture.  Returns (row_mask, lo, hi) for the non-empty ones."""
    base = cols["base"].astype(np.int64)
    ff = cols["ff"].astype(np.int64)
    te = enter[base, ff]
    tl = leave[base, ff]
    open_end = cols["t_exit"] >= INF_PS
    lo = np.where(open_end, 0, tl - cols["t_exit"] + 1)
    hi = te - cols["t_enter"]            # s_high + 1
    lo = np.clip(lo, 0, dom)
    hi = np.clip(hi, 0, dom)
    keep = hi > lo
    return keep, lo[keep], hi[keep]


def union_by_group(g: np.ndarray, lo: np.ndarray, hi: np.ndarray, span: int):
    """Union of half-open intervals within each group id ``g`` (int64 >= 0),
    all intervals inside [0, span].  Returns (group, lo, hi) of the merged,
    disjoint intervals, sorted by group then lo."""
    if g.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    uniq, gd = np.unique(g, return_inverse=True)   # dense ids keep g * b inside int64
    gd = gd.astype(np.int64).reshape(-1)
    b = np.int64(span + 1)
    lo2 = gd * b + lo
    hi2 = gd * b + hi
    order = np.argsort(lo2, kind="stable")
    lo2, hi2 = lo2[order], hi2[order]
    cm = np.maximum.accumulate(hi2)
    prev = np.empty_like(cm)
    prev[0] = -1
    prev[1:] = cm[:-1]
    start = lo2 > prev
    idx = np.flatnonzero(start)
    m_lo = lo2[idx]
    m_hi = np.maximum.reduceat(hi2, idx)
    m_gd = m_lo // b
    return uniq[m_gd], m_lo - m_gd * b, m_hi - m_gd * b


@dataclass
class ExposureTotals:
    """Sums over the campaign, in ps of start time (weighted by width probability)."""
    n_ff_ids: int
    n_sites: int
    widths: List[int]
    weights: np.ndarray                 # float64[n_w], sums to 1
    exposure_by_width: np.ndarray       # float64[n_w]   sum over (cycle, site) of |Omega|
    ff_exposure: np.ndarray             # float64[n_ff_ids] sum_k p(w_k) |A_kf|
    site_exposure: np.ndarray           # float64[n_sites] sum_k p(w_k) |Omega_k|
    multiplicity: Dict[int, float] = field(default_factory=dict)  # m -> sum_k p(w_k) |{s: m FFs}|
    n_trials_upset_by_width: Optional[np.ndarray] = None           # trials with |Omega| > 0
    trial_table: Optional[List[np.ndarray]] = None


def accumulate(records: Records, weights: Dict[int, float], phase_domain: str = "half_open",
               keep_trials: bool = False) -> ExposureTotals:
    widths = records.pulse_widths_ps
    w_arr = np.asarray([weights[w] for w in widths], dtype=np.float64)
    dom = domain_length(records.period_ps, phase_domain)
    enter, leave = records.thresholds()
    n_ff = enter.shape[1]
    n_sites = len(records.sites)
    n_w = len(widths)
    wsorted = np.asarray(widths, dtype=np.int64)
    if np.any(np.diff(wsorted) <= 0):
        raise ValueError("pulse widths in the index must be strictly increasing")
    cyc_list = records.cycles

    tot = ExposureTotals(n_ff_ids=n_ff, n_sites=n_sites, widths=list(widths), weights=w_arr,
                         exposure_by_width=np.zeros(n_w), ff_exposure=np.zeros(n_ff),
                         site_exposure=np.zeros(n_sites), n_trials_upset_by_width=np.zeros(n_w, np.int64),
                         trial_table=[] if keep_trials else None)
    mult: Dict[int, float] = {}
    for cols in records.shards():
        if cols["cycle"].size == 0:
            continue
        keep, lo, hi = capture_intervals(cols, enter, leave, dom)
        if not keep.any():
            continue
        cyc = cols["cycle"][keep]
        ci = _map_cycles(cyc, cyc_list)
        wi = np.searchsorted(wsorted, cols["width_ps"][keep].astype(np.int64))
        site = cols["site"][keep].astype(np.int64)
        ff = cols["ff"][keep].astype(np.int64)
        trial = (ci * n_sites + site) * n_w + wi

        # per (trial, FF): the FF's capture region
        tf_g, tf_lo, tf_hi = union_by_group(trial * n_ff + ff, lo, hi, dom)
        tf_trial = tf_g // n_ff
        tf_ff = tf_g - tf_trial * n_ff
        tf_len = (tf_hi - tf_lo).astype(np.float64)
        tf_w = w_arr[tf_trial % n_w]
        tot.ff_exposure += np.bincount(tf_ff, weights=tf_len * tf_w, minlength=n_ff)

        # per trial: exposure = union over FFs
        t_g, t_lo, t_hi = union_by_group(tf_trial, tf_lo, tf_hi, dom)
        t_len = np.bincount(t_g, weights=(t_hi - t_lo).astype(np.float64))
        trials = np.flatnonzero(t_len > 0)
        lens = t_len[trials]
        t_w_idx = trials % n_w
        t_site = (trials // n_w) % n_sites
        tot.exposure_by_width += np.bincount(t_w_idx, weights=lens, minlength=n_w)
        tot.n_trials_upset_by_width += np.bincount(t_w_idx, minlength=n_w)
        tot.site_exposure += np.bincount(t_site, weights=lens * w_arr[t_w_idx], minlength=n_sites)
        if keep_trials:
            tot.trial_table.append(np.stack([trials, lens.astype(np.int64)], axis=1))

        # multi-bit upsets: coverage count over start time, per trial
        ev_g = np.concatenate([tf_trial, tf_trial])
        ev_pos = np.concatenate([tf_lo, tf_hi])
        ev_d = np.concatenate([np.ones(tf_lo.size, np.int64), -np.ones(tf_hi.size, np.int64)])
        order = np.lexsort((ev_d, ev_pos, ev_g))
        ev_g, ev_pos, ev_d = ev_g[order], ev_pos[order], ev_d[order]
        count = np.cumsum(ev_d)
        seg = np.zeros(ev_pos.size, dtype=np.int64)
        same = ev_g[1:] == ev_g[:-1]
        seg[:-1] = np.where(same, ev_pos[1:] - ev_pos[:-1], 0)
        ok = (seg > 0) & (count > 0)
        if ok.any():
            m_w = w_arr[ev_g[ok] % n_w]
            h = np.bincount(count[ok], weights=seg[ok] * m_w)
            for m in np.flatnonzero(h):
                mult[int(m)] = mult.get(int(m), 0.0) + float(h[m])
    tot.multiplicity = dict(sorted(mult.items()))
    return tot


def _map_cycles(cyc: np.ndarray, cycles: List[int]) -> np.ndarray:
    arr = np.asarray(cycles, dtype=np.int64)
    order = np.argsort(arr)
    pos = np.searchsorted(arr[order], cyc.astype(np.int64))
    if np.any(pos >= arr.size) or np.any(arr[order][np.minimum(pos, arr.size - 1)] != cyc):
        raise ValueError("records contain a cycle that is not in the index")
    return order[pos]
