"""The pattern table: hand-made and random records, the C++ sweep against the
Python reference, the expansion against a brute-force grid, the file formats,
and the records of a campaign."""
import csv
import json
import os
import random
from dataclasses import fields

import numpy as np
import pytest

from setfi import _engine
from setfi.analysis.patterns import FORMAT, Patterns, Source, compute_patterns, phase_grid, reference_patterns
from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.records import COLUMNS, INF_PS, Records, write_shard
from setfi.records import FORMAT as RECORDS_FORMAT
from test_fastpath import write_random_substrate, write_states

T = 100
# FF id -> (capture_enter_ps, capture_leave_ps), for both base values.  FF 0
# captures a pulse [a, b) for start times s in [91 - b, 104 - a], FF 1 for s in
# [81 - b, 99 - a], FF 2 for s in [106 - b, 119 - a].
FFS = {0: ([T + 5] * 2, [T - 10] * 2), 1: ([T] * 2, [T - 20] * 2), 2: ([T + 20] * 2, [T + 5] * 2)}


def write_records(root, files, ffs=FFS, sites=("s0", "s1"), widths=(10, 20), cycles=(7, 9)):
    """A records directory with one record file per list of rows
    (cycle, site, width_ps, ff, base, t_enter, t_exit)."""
    os.makedirs(root, exist_ok=True)
    shards = []
    for i, rows in enumerate(files):
        cols = {k: np.asarray([r[j] for r in rows], dtype=dt) for j, (k, dt) in enumerate(COLUMNS.items())}
        shards.append(write_shard(os.path.join(root, f"records_{i:03d}.npz"), cols))
    index = {"format": RECORDS_FORMAT, "clock_period_ps": T, "capture_edge_ps": T,
             "pulse_widths_ps": list(widths), "cycles": list(cycles),
             "n_trials": len(cycles) * len(sites) * len(widths), "sites": list(sites),
             "ffs": [{"id": f, "inst": f"ff{f}", "capture_enter_ps": e, "capture_leave_ps": lv}
                     for f, (e, lv) in ffs.items()],
             "shards": shards}
    with open(os.path.join(root, "index.json"), "w") as f:
        json.dump(index, f)
    return Records.open(root)


# On the grid 0, 10, .., 100 (steps 10), k = s / 10:
HAND = [
    [(7, 0, 10, 0, 0, 30, 40), (7, 0, 10, 1, 1, 35, 60),        # FF 0 at k 6..7, FF 1 at k 3..6
     (7, 0, 20, 0, 0, 30, 50), (7, 0, 20, 0, 0, 70, 75),        # two pulses on FF 0: k 5..7 and 2..3
     (7, 1, 10, 1, 0, 100, INF_PS), (7, 1, 10, 2, 0, 2, 5),     # no capture: s <= -1, s >= 101
     (7, 1, 20, 1, 0, 99, INF_PS), (7, 1, 20, 2, 1, 2, 6)],     # open-ended, only s = 0; only s = T
    [(9, 0, 10, 0, 0, 30, 40),                                  # FF 0 at k 6..7 ...
     (9, 1, 10, 0, 1, 20, 45),                                  # ... and at k 5..8 from another site
     (9, 1, 20, 1, 0, 10, 50), (9, 1, 20, 1, 0, 40, 70)],       # overlapping pulses on FF 1: k 2..8
]
# cycle, FFs, multiplicity, sites, runs (site, width, k_lo, k_hi)
HAND_TABLE = [
    (7, [0], 6, 1, [(0, 10, 7, 7), (0, 20, 2, 3), (0, 20, 5, 7)]),
    (7, [0, 1], 1, 1, [(0, 10, 6, 6)]),
    (7, [1], 4, 2, [(0, 10, 3, 5), (1, 20, 0, 0)]),
    (7, [2], 1, 1, [(1, 20, 10, 10)]),
    (9, [0], 6, 2, [(0, 10, 6, 7), (1, 10, 5, 8)]),
    (9, [1], 7, 1, [(1, 20, 2, 8)]),
]


def assert_same(a: Patterns, b: Patterns):
    for f in fields(Patterns):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y), f.name
        else:
            assert x == y, f.name


def test_hand_made_records(tmp_path):
    rec = write_records(str(tmp_path), HAND)
    pt = compute_patterns(rec, steps=10)
    assert pt.starts_ps.tolist() == list(range(0, T + 1, 10))
    got = [(pt.cycle(p), pt.ffs(p), pt.multiplicity(p), int(pt.pattern_n_sites[p]),
            [(s.site, s.width_ps, s.k_lo, s.k_hi) for s in pt.sources(p)]) for p in range(pt.count)]
    assert got == [(c, f, m, n, [(f"s{s}", w, lo, hi) for s, w, lo, hi in runs])
                   for c, f, m, n, runs in HAND_TABLE]
    assert pt.sources(3) == [Source("s1", 20, 10, 10, T, T)]
    assert pt.ffs(1, names=True) == ["ff0", "ff1"]
    assert_same(pt, reference_patterns(str(tmp_path), steps=10))
    none = write_records(str(tmp_path / "none"), [[], HAND[0][4:6]])     # no SET upsets anything
    assert compute_patterns(none, steps=10).count == 0
    assert_same(compute_patterns(none, steps=10), reference_patterns(none, steps=10))


def random_campaign(root, seed, n_files=4):
    """Random records in the layout setfi writes (each trial one block of one file)."""
    rng = random.Random(seed)
    ffs = {f: ([T + rng.randint(-5, 30), T + rng.randint(-5, 30)],
               [T - rng.randint(-5, 30), T - rng.randint(-5, 30)]) for f in (0, 1, 2, 4, 5, 7)}
    cycles, widths = [11, 3, 25], [5, 20, 45]
    rows = []
    for c in cycles:
        for s in range(5):
            for w in widths:
                if rng.random() < 0.25:
                    continue                      # no record: the SET reached no FF
                for _ in range(rng.randint(1, 7)):
                    a = rng.randint(0, 130)
                    b = INF_PS if rng.random() < 0.15 else a + rng.randint(1, 70)
                    rows.append((c, s, w, rng.choice(list(ffs)), rng.randint(0, 1), a, b))
    trial_starts = [i for i in range(len(rows)) if i == 0 or rows[i][:3] != rows[i - 1][:3]]
    cuts = sorted(rng.sample(trial_starts[1:], n_files - 1))
    files = [rows[a:b] for a, b in zip([0] + cuts, cuts + [len(rows)])]
    rng.shuffle(files)                            # the order of the files does not matter
    return write_records(root, files, ffs=ffs, sites=[f"n{i}" for i in range(5)], widths=widths,
                         cycles=cycles), rows


@pytest.mark.parametrize("seed,steps", [(1, 1), (2, 7), (3, 20), (4, 50), (5, 150)])
def test_cpp_sweep_matches_the_reference(tmp_path, seed, steps):
    # steps 150 > T: the grid repeats start times
    rec, _ = random_campaign(str(tmp_path), seed)
    ref = reference_patterns(rec, steps)
    assert ref.count > 0
    for jobs in (1, 3):
        assert_same(compute_patterns(rec, steps, jobs=jobs), ref)


def test_trials_split_over_files_or_within_one(tmp_path):
    rec, rows = random_campaign(str(tmp_path / "whole"), 7, n_files=1)
    # cut the records inside trials, and rotate one piece so that a trial also
    # appears in two places of one file
    rng = random.Random(7)
    inside = [i for i in range(1, len(rows)) if rows[i][:3] == rows[i - 1][:3]]
    cuts = sorted(rng.sample(inside, 3))
    pieces = [rows[a:b] for a, b in zip([0] + cuts, cuts + [len(rows)])]
    half = len(pieces[1]) // 2
    pieces[1] = pieces[1][half:] + pieces[1][:half]
    ffs = {f["id"]: (f["capture_enter_ps"], f["capture_leave_ps"]) for f in rec.ffs}
    split = write_records(str(tmp_path / "split"), pieces, ffs=ffs, sites=rec.sites,
                          widths=rec.pulse_widths_ps, cycles=rec.cycles)
    enter, leave = split.thresholds()
    builder = _engine.PatternBuilder(phase_grid(T, 20), enter, leave, np.asarray(split.cycles),
                                     len(split.sites), np.asarray(split.pulse_widths_ps), INF_PS)
    for cols in split.shards():
        builder.add(cols)
    assert builder.n_split_trials > 0
    with pytest.raises(RuntimeError, match="split trials are pending"):
        builder.result()
    for jobs in (1, 2):
        assert_same(compute_patterns(split, 20, jobs=jobs), compute_patterns(rec, 20))
    assert_same(compute_patterns(split, 20), reference_patterns(rec, 20))


def brute_force_grid(rec: Records, steps: int):
    """{(cycle, site, width_ps, k): FFs}: the capture rule at every grid point."""
    enter, leave = rec.thresholds()
    starts = phase_grid(rec.index["capture_edge_ps"], steps).tolist()
    cols = rec.load_all()
    grid = {}
    for c, s, w, f, b, t_enter, t_exit in zip(*(cols[k].tolist() for k in COLUMNS)):
        for k, start in enumerate(starts):
            if start <= enter[b, f] - t_enter - 1 and (t_exit >= INF_PS or start >= leave[b, f] - t_exit + 1):
                grid.setdefault((c, s, w, k), set()).add(f)
    return grid


def assert_expands_to_the_brute_force_grid(pt: Patterns, rec: Records):
    """The expanded runs are the grid points at which a SET upsets flip-flops,
    each once, in order, with those flip-flops."""
    g = pt.grid_points()
    keys = list(zip(g["cycle"].tolist(), g["site"].tolist(), g["width_ps"].tolist(), g["k"].tolist()))
    assert keys == sorted(keys) and len(set(keys)) == len(keys) == int(pt.pattern_multiplicity.sum())
    assert {key: set(pt.ffs(p)) for key, p in zip(keys, g["pattern"].tolist())} == brute_force_grid(rec, pt.steps)


@pytest.mark.parametrize("seed,steps", [(8, 10), (9, 33)])
def test_runs_expand_to_the_brute_force_grid(tmp_path, seed, steps):
    rec, _ = random_campaign(str(tmp_path), seed)
    pt = compute_patterns(rec, steps)
    assert_expands_to_the_brute_force_grid(pt, rec)
    g = pt.grid_points()
    assert np.array_equal(g["start_ps"], pt.starts_ps[g["k"]])
    one = pt.grid_points(cycles=[25])
    assert set(one["cycle"].tolist()) == {25} and one["k"].size == int(np.sum(g["cycle"] == 25))


def test_save_load_and_csv(tmp_path):
    write_records(str(tmp_path / "rec"), HAND)
    pt = compute_patterns(str(tmp_path / "rec"), 10)
    path = str(tmp_path / "patterns.npz")
    pt.save(path)
    assert_same(Patterns.load(path), pt)
    with np.load(path) as z:                     # plain arrays: no pickles needed
        assert str(z["format"]) == FORMAT and all(z[k].dtype != object for k in z.files)
    pt.to_csv(str(tmp_path / "patterns.csv"))
    with open(tmp_path / "patterns.csv", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["pattern", "cycle", "multiplicity", "n_ffs", "n_sites", "ffs"]
    assert rows[1:] == [[str(p), str(c), str(m), str(len(ffs)), str(n), "|".join(map(str, ffs))]
                        for p, (c, ffs, m, n, _) in enumerate(HAND_TABLE)]
    np.savez(str(tmp_path / "other.npz"), x=np.zeros(1))
    with pytest.raises(ValueError, match=FORMAT):
        Patterns.load(str(tmp_path / "other.npz"))


def test_bad_records_and_steps_are_refused(tmp_path):
    with pytest.raises(ValueError, match="steps"):
        compute_patterns(write_records(str(tmp_path / "a"), HAND), 0)
    for i, (bad, message) in enumerate([((8, 0, 10, 0, 0, 1, 2), "cycle 8 is not a sampled cycle"),
                                        ((7, 2, 10, 0, 0, 1, 2), "site 2 is not a site index"),
                                        ((7, 0, 30, 0, 0, 1, 2), "width 30 ps was not injected"),
                                        ((7, 0, 10, 3, 0, 1, 2), "FF id 3")]):
        rec = write_records(str(tmp_path / f"bad{i}"), [HAND[0] + [bad]])
        with pytest.raises(ValueError, match=message):
            compute_patterns(rec, 10)


@pytest.mark.parametrize("seed,opts", [(0, {}), (6, {"capture_edge_ps": 90, "sim_horizon_ps": 140})])
def test_patterns_of_a_campaign(tmp_path, seed, opts):
    names = write_random_substrate(str(tmp_path / "build"), seed)
    write_states(str(tmp_path / "state"), len(names), [3, 5, 8], seed)
    spec = CampaignSpec(stage_dir=str(tmp_path / "build"), state_dir=str(tmp_path / "state"), cycles=[3, 5, 8],
                        out_dir=str(tmp_path / "inject"), clock_period_ps=100, pulse_widths_ps=[5, 15, 30, 60],
                        jobs=2, cycles_per_shard=2, allow_missing_timing_checks=True, **opts)
    run_campaign(spec, log=lambda *a: None)
    rec = Records.open(str(tmp_path / "inject"))
    pt = compute_patterns(rec, 20, jobs=2)
    assert pt.count > 0 and pt.period_ps == opts.get("capture_edge_ps", 100)
    assert_expands_to_the_brute_force_grid(pt, rec)
    assert_same(pt, reference_patterns(rec, 20))
