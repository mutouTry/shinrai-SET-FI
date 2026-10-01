"""Analysis on hand-made records: capture regions, exposure, multiplicity, widths."""
import json
import math
import os

import numpy as np
import pytest

from setfi.analysis.exposure import accumulate, capture_intervals, union_by_group
from setfi.analysis.report import analyze
from setfi.analysis.widths import WidthModelError, width_weights
from setfi.records import FORMAT, INF_PS, Records, write_shard

T = 100


def _write_records(root, rows, sites=("s0", "s1"), widths=(10, 20), cycles=(7,)):
    os.makedirs(root, exist_ok=True)
    cols = {k: np.asarray([r[i] for r in rows]) for i, k in
            enumerate(("cycle", "site", "width_ps", "ff", "base", "t_enter", "t_exit"))}
    if not rows:
        cols = {k: np.zeros(0) for k in ("cycle", "site", "width_ps", "ff", "base", "t_enter", "t_exit")}
    shard = write_shard(os.path.join(root, "shard_00000.npz"), cols)
    # FF 0: window (T-10, T+5) for both bases; FF 1: (T-20, T+0)
    ffs = [{"id": 0, "inst": "f0", "d_net": "d0", "d_pin": "D", "capture_enter_ps": [T + 5, T + 5],
            "capture_leave_ps": [T - 10, T - 10]},
           {"id": 1, "inst": "f1", "d_net": "d1", "d_pin": "D", "capture_enter_ps": [T, T],
            "capture_leave_ps": [T - 20, T - 20]}]
    index = {"format": FORMAT, "clock_period_ps": T, "capture_edge_ps": T,
             "pulse_widths_ps": list(widths), "cycles": list(cycles),
             "n_trials": len(cycles) * len(sites) * len(widths), "sites": list(sites), "ffs": ffs,
             "shards": [shard]}
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f)
    return Records.open(root)


def brute(rows, dom):
    """Exposure and multiplicity by walking every start time."""
    by = {}
    for c, s, w, f, b, a, e in rows:
        by.setdefault((c, s, w), []).append((f, b, a, e))
    enter = {0: T + 5, 1: T}
    leave = {0: T - 10, 1: T - 20}
    out = {}
    for k, pulses in by.items():
        cov = []
        for s0 in range(dom):
            hit = set()
            for f, b, a, e in pulses:
                if s0 + a < enter[f] and (e >= INF_PS or s0 + e > leave[f]):
                    hit.add(f)
            cov.append(len(hit))
        out[k] = cov
    return out


ROWS = [
    # cycle, site, pw, ff, base, t_enter, t_exit
    (7, 0, 10, 0, 0, 30, 40),        # FF0: s in [T-10-40+1, T+5-30-1] = [51, 74]
    (7, 0, 10, 1, 1, 35, 60),        # FF1: [T-20-60+1, T-35-1] = [21, 64]
    (7, 0, 20, 0, 0, 30, 50),
    (7, 0, 20, 0, 0, 70, 75),        # second pulse on FF0 in the same trial
    (7, 1, 20, 1, 0, 90, INF_PS),    # open-ended: [0, T-90-1]
]


def test_capture_interval_formula(tmp_path):
    rec = _write_records(str(tmp_path), ROWS)
    enter, leave = rec.thresholds()
    cols = rec.load_all()
    keep, lo, hi = capture_intervals(cols, enter, leave, T)
    assert keep.all()
    assert (lo[0], hi[0]) == (51, 75)
    assert (lo[1], hi[1]) == (21, 65)
    assert (lo[4], hi[4]) == (0, 10)


def test_union_by_group():
    g = np.array([3, 3, 3, 9, 9])
    lo = np.array([0, 5, 20, 1, 1])
    hi = np.array([10, 12, 30, 2, 2])
    mg, mlo, mhi = union_by_group(g, lo, hi, 50)
    assert mg.tolist() == [3, 3, 9]
    assert mlo.tolist() == [0, 20, 1]
    assert mhi.tolist() == [12, 30, 2]


@pytest.mark.parametrize("domain,dom", [("half_open", T), ("closed", T + 1)])
def test_exposure_and_multiplicity_match_brute_force(tmp_path, domain, dom):
    rec = _write_records(str(tmp_path), ROWS)
    w = width_weights({"type": "uniform"}, rec.pulse_widths_ps)
    tot = accumulate(rec, w, phase_domain=domain)
    bf = brute(ROWS, dom)
    exp_by_w = {10: 0, 20: 0}
    mult = {}
    for (c, s, pw), cov in bf.items():
        exp_by_w[pw] += sum(1 for x in cov if x > 0)
        for x in cov:
            if x:
                mult[x] = mult.get(x, 0) + 0.5   # uniform over 2 widths
    assert tot.exposure_by_width.tolist() == [exp_by_w[10], exp_by_w[20]]
    assert tot.multiplicity == pytest.approx(mult)


def test_report_probabilities(tmp_path):
    rec_dir = str(tmp_path / "rec")
    _write_records(rec_dir, ROWS)
    s = analyze(rec_dir, str(tmp_path / "an"), {"type": "uniform"}, log=lambda *a: None)
    bf = brute(ROWS, T)
    total = sum(0.5 * sum(1 for x in cov if x > 0) for cov in bf.values())
    assert s["upset_probability"] == pytest.approx(total / (1 * 2 * T))
    assert sum(s["multi_bit_upset_probability"].values()) == pytest.approx(s["upset_probability"])
    for name in ("summary.json", "per_ff.csv", "per_site.csv", "per_width.csv"):
        assert (tmp_path / "an" / name).exists()


def test_width_models():
    ws = [20, 40, 60, 80, 100, 120, 140, 160, 180]
    g = width_weights({"type": "gaussian", "mean_ps": 90, "sd_ps": 40}, ws)
    # N(90, 40) at the injected widths, normalised, rounded to 1e-4
    assert [round(g[w] * 1e4) for w in ws] == [444, 939, 1548, 1988, 1988, 1548, 939, 444, 163]
    e = width_weights({"type": "exponential", "tau_ps": 40}, ws)
    assert [round(e[w] * 1e4) for w in ws] == [3979, 2413, 1464, 888, 538, 327, 198, 120, 73]
    u = width_weights({"type": "uniform"}, ws)
    assert all(math.isclose(v, 1 / 9) for v in u.values())
    t = width_weights({"type": "table", "weights": {str(w): 1 for w in ws}}, ws)
    assert t == pytest.approx(u)
    with pytest.raises(WidthModelError):
        width_weights({"type": "table", "weights": {"20": 1}}, ws)
    # a mean far outside the injected widths: the nearest width takes the weight
    far = width_weights({"type": "gaussian", "mean_ps": 5000, "sd_ps": 20}, ws)
    assert far[180] == pytest.approx(1.0)
    with pytest.raises(WidthModelError):
        width_weights({"type": "gaussian", "mean_ps": 90, "sd_ps": 0}, ws)
    with pytest.raises(WidthModelError):
        width_weights({"type": "gaussian", "mean_ps": 90, "sd_ps": 40, "tau_ps": 3}, ws)
