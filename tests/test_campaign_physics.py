"""Campaign semantics on a hand-written timing graph.

    FF A --a--> INV u1 --n1--> NAND2 u2 --n2--> D of FF C
    FF B --b-----------------> (A2 of u2)

Delays (ps): u1 I->ZN rise 10 / fall 12; u2 A1->ZN rise 20 / fall 25,
A2->ZN rise 22 / fall 27.  FF C: setup 30, hold 5.  T = 100.

cycle 1: a=0, b=1  => n1=1, n2=0.   cycle 2: a=0, b=0  => n1=1, n2=1.
"""
import json
import os

import numpy as np
import pytest

from setfi.analysis.exposure import capture_intervals
from setfi.analysis.patterns import compute_patterns
from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.records import INF_PS, Records

NETS = ["a", "b", "n1", "n2"]          # net_index order (sorted)


def write_substrate(d, checks=None, pins=None, patterns=None, uncond=None):
    """checks: ffC's timing checks; pins: more instance pins; patterns: the COND
    patterns (mask, value_bits, delay_ps, is_rise) of u2:A1->ZN; uncond: whether
    that arc also has an unconditional SDF entry."""
    os.makedirs(d, exist_ok=True)
    idx = {n: i for i, n in enumerate(NETS)}
    j = lambda name, obj: json.dump(obj, open(os.path.join(d, name), "w"))  # noqa: E731
    j("net_index.json", {"net_to_idx": idx, "idx_to_net": {str(i): n for n, i in idx.items()}})
    edges = [
        dict(eid=0, src_net="a", dst_net="n1", inst="u1", celltype="INV", inpin="I", outpin="ZN",
             mask_key="INV:I->ZN", delay_key="u1:I->ZN"),
        dict(eid=1, src_net="n1", dst_net="n2", inst="u2", celltype="ND2", inpin="A1", outpin="ZN",
             mask_key="ND2:A1->ZN", delay_key="u2:A1->ZN"),
        dict(eid=2, src_net="b", dst_net="n2", inst="u2", celltype="ND2", inpin="A2", outpin="ZN",
             mask_key="ND2:A2->ZN", delay_key="u2:A2->ZN"),
    ]
    j("edges.json", edges)
    j("pin_to_net.json", {"u1.I": "a", "u1.ZN": "n1", "u2.A1": "n1", "u2.A2": "b", "u2.ZN": "n2",
                          "ffC.D": "n2", **(pins or {})})
    with open(os.path.join(d, "site_cones.jsonl"), "w") as f:
        f.write(json.dumps({"src_net_idx": idx["n1"], "cone_topo_net_idxs": [idx["n1"], idx["n2"]],
                            "cone_adj_idx": [[idx["n1"], [1]]], "reachable_dnet_idxs": [idx["n2"]]}) + "\n")
        f.write(json.dumps({"src_net_idx": idx["n2"], "cone_topo_net_idxs": [idx["n2"]],
                            "cone_adj_idx": [], "reachable_dnet_idxs": [idx["n2"]]}) + "\n")
    j("ff_index.json", {"ffid_to_info": {"0": {"ff_inst": "ffC", "d_net": "n2", "d_pin": "D",
                                               "celltype": "DFF"}}})
    j("cell_arcs.json", {
        "INV:I->ZN": {"vars_order": ["I"], "truth": [1, 0], "side_pins": []},
        "ND2:A1->ZN": {"vars_order": ["A1", "A2"], "truth": [1, 1, 1, 0], "side_pins": ["A2"]},
        "ND2:A2->ZN": {"vars_order": ["A2", "A1"], "truth": [1, 1, 1, 0], "side_pins": ["A1"]},
    })
    chk = lambda s, h: {"posedge:CP": {"SETUP": [{"value": s}], "HOLD": [{"value": h}]}}  # noqa: E731
    j("timing_checks.json", {"index": {"ffC": {"meta": {"ff_inst": "ffC"}, "checks": checks or {
        "posedge:D": chk(0.030, 0.005), "negedge:D": chk(0.030, 0.005)}}}})
    keys = ["u1:I->ZN", "u2:A1->ZN", "u2:A2->ZN"]
    j("arc_index.json", {"delay_key_to_arc_idx": {k: i for i, k in enumerate(keys)}})
    pt = np.asarray(patterns or np.zeros((0, 4)), dtype=np.int64).reshape(-1, 4)
    np.savez(os.path.join(d, "delay_table.npz"),
             default_rise_ps=np.array([10, 20, 22]), default_fall_ps=np.array([12, 25, 27]),
             patt_ptr=np.array([0, 0, len(pt), len(pt)], np.int32), patt_mask=pt[:, 0],
             patt_valbits=pt[:, 1], patt_dt_ps=pt[:, 2], patt_is_rise=pt[:, 3].astype(np.uint8),
             uncond_rise=np.array([0, int(bool(uncond)), 0], np.uint8),
             uncond_fall=np.array([0, int(bool(uncond)), 0], np.uint8))


def write_state(d):
    os.makedirs(d, exist_ok=True)
    # bit i = value of NETS[i]
    for cyc, vals in {1: dict(a=0, b=1, n1=1, n2=0), 2: dict(a=0, b=0, n1=1, n2=1)}.items():
        v = sum(vals[n] << i for i, n in enumerate(NETS))
        open(os.path.join(d, f"cycle_{cyc}.hex"), "w").write(f"{v:x}\n")


def run(tmp_path, name="out", vcs_align=False, **substrate):
    write_substrate(str(tmp_path / name / "build"), **substrate)
    write_state(str(tmp_path / name / "state"))
    spec = CampaignSpec(stage_dir=str(tmp_path / name / "build"), state_dir=str(tmp_path / name / "state"),
                        cycles=[1, 2], out_dir=str(tmp_path / name / "inject"), clock_period_ps=100,
                        pulse_widths_ps=[10, 20, 40], vcs_align=vcs_align)
    run_campaign(spec, log=lambda *a: None)
    return Records.open(str(tmp_path / name / "inject"))


@pytest.fixture
def campaign(tmp_path):
    return run(tmp_path)


def rows(rec):
    c = rec.load_all()
    sites = rec.sites
    return sorted((int(c["cycle"][i]), sites[int(c["site"][i])], int(c["width_ps"][i]), int(c["base"][i]),
                   int(c["t_enter"][i]), int(c["t_exit"][i])) for i in range(c["cycle"].size))


def test_expected_pulses(campaign):
    rec = campaign
    assert rows(rec) == [
        # cycle 1, SET on n1 (1 -> 0 -> 1): n2 rises after 20 (A1 rise delay), falls
        # after w + 25; a 10-ps pulse dies at u2 (inertial: narrower than its delay)
        (1, "n1", 20, 0, 20, 45),
        (1, "n1", 40, 0, 20, 65),
        # cycle 1, SET directly on the D net
        (1, "n2", 10, 0, 0, 10), (1, "n2", 20, 0, 0, 20), (1, "n2", 40, 0, 0, 40),
        # cycle 2: b = 0 forces n2 = 1, the SET on n1 is logically masked;
        # a SET on n2 is a 1 -> 0 -> 1 excursion (base 1)
        (2, "n2", 10, 1, 0, 10), (2, "n2", 20, 1, 0, 20), (2, "n2", 40, 1, 0, 40),
    ]


def test_capture_window(campaign):
    rec = campaign
    enter, leave = rec.thresholds()
    # base 0: enter = T + hold(rise) = 105, leave = T - setup(fall) = 70
    assert enter[0, 0] == 105 and leave[0, 0] == 70
    cols = rec.load_all()
    keep, lo, hi = capture_intervals(cols, enter, leave, 101)
    got = {(int(cols["cycle"][i]), rec.sites[int(cols["site"][i])], int(cols["width_ps"][i])):
           (int(a), int(b) - 1) for i, a, b in zip(np.flatnonzero(keep), lo, hi)}
    # SET on n1, width 20: pulse [s+20, s+45) overlaps (70, 105) iff 26 <= s <= 84
    assert got[(1, "n1", 20)] == (26, 84)
    # SET on n2, width 10: [s, s+10) overlaps (70, 105) iff 61 <= s <= 100 (clipped to T)
    assert got[(1, "n2", 10)] == (61, 100)


def test_pattern_grid_matches_the_capture_intervals(campaign):
    rec = campaign
    enter, leave = rec.thresholds()
    cols = rec.load_all()
    keep, lo, hi = capture_intervals(cols, enter, leave, 101)
    from_records = set()
    for i, a, b in zip(np.flatnonzero(keep), lo, hi):
        for s in range(a, b):
            from_records.add((int(cols["cycle"][i]), rec.sites[int(cols["site"][i])], s,
                              int(cols["width_ps"][i])))
    pt = compute_patterns(rec, steps=100)          # start times 0, 1, .., 100 ps
    assert [pt.ffs(p) for p in range(pt.count)] == [[0]] * pt.count
    g = pt.grid_points()
    grid = {(c, rec.sites[s], start, w) for c, s, start, w in
            zip(g["cycle"].tolist(), g["site"].tolist(), g["start_ps"].tolist(), g["width_ps"].tolist())}
    assert grid == from_records


def cond_checks(setup, hold):
    """ffC's checks: [(COND, ns), ...] entries for SETUP and HOLD, both data edges."""
    entry = lambda c, v: {"value": v, "enable_cond": c}  # noqa: E731
    cp = {"posedge:CP": {"SETUP": [entry(c, v) for c, v in setup], "HOLD": [entry(c, v) for c, v in hold]}}
    return {"posedge:D": cp, "negedge:D": cp}


def test_setup_hold_follow_the_cond_of_each_cycle(tmp_path):
    # ffC.SE is net b: 1 in cycle 1, 0 in cycle 2.  TE is not connected, so its
    # COND never holds.
    rec = run(tmp_path, checks=cond_checks(setup=[("SE == 1'b1", 0.030), ("SE == 1'b0", 0.050)],
                                           hold=[("SE == 1'b1", 0.005), ("TE == 1'b1", 0.015)]),
              pins={"ffC.SE": "b"})
    enter, leave = rec.thresholds()
    exc = rec.window_exceptions()
    # cycle 1: setup 30, hold 5.  cycle 2: setup 50, and no HOLD entry holds: 0
    assert (enter[0, 0], leave[0, 0]) == (105, 70)
    assert exc["cycle"].tolist() == [2] and exc["ff"].tolist() == [0]
    assert exc["enter"].tolist() == [[100, 100]] and exc["leave"].tolist() == [[50, 50]]
    cols = rec.load_all()
    keep, lo, hi = capture_intervals(cols, enter, leave, 101, exc)
    got = {(int(cols["cycle"][i]), rec.sites[int(cols["site"][i])], int(cols["width_ps"][i])):
           (int(a), int(b) - 1) for i, a, b in zip(np.flatnonzero(keep), lo, hi)}
    # cycle 2, SET on n2, width 10: [s, s+10) overlaps (50, 100) iff 41 <= s <= 99
    assert got[(2, "n2", 10)] == (41, 99)
    assert got[(1, "n2", 10)] == (61, 100)
    g = compute_patterns(rec, 100).grid_points()      # starts_ps[k] = k
    n2 = rec.sites.index("n2")
    for c, lo_hi in ((1, (61, 100)), (2, (41, 99))):
        ks = g["k"][(g["cycle"] == c) & (g["site"] == n2) & (g["width_ps"] == 10)]
        assert (ks.min(), ks.max()) == lo_hi


def test_a_cond_setfi_cannot_read_keeps_the_first_entry(tmp_path):
    rec = run(tmp_path, checks=cond_checks(setup=[("!SE", 0.030), ("SE", 0.050)],
                                           hold=[("!SE", 0.005), ("SE", 0.015)]),
              pins={"ffC.SE": "b"})
    enter, leave = rec.thresholds()
    assert (enter[0, 0], leave[0, 0]) == (105, 70)
    assert rec.window_exceptions()["cycle"].size == 0


def test_vcs_align_zero_delay_when_no_cond_path_is_active(tmp_path):
    # u2:A1->ZN is conditional on A2 == 0 (side bit 0); in cycle 1 A2 = b = 1, so
    # the path from n1 is inactive
    pt = [(1, 0, 99, 1), (1, 0, 99, 0)]
    key = (1, "n1", 20, 0)
    pulses = lambda r: {x[4:] for x in rows(r) if x[:4] == key}  # noqa: E731
    # default delays (rise 20 at n1's fall, fall 25 at its rise), or none with vcs_align
    assert pulses(run(tmp_path, "off", patterns=pt)) == {(20, 45)}
    assert pulses(run(tmp_path, "on", vcs_align=True, patterns=pt)) == {(0, 20)}
    # an unconditional SDF entry keeps the path active
    assert pulses(run(tmp_path, "uncond", vcs_align=True, patterns=pt, uncond=True)) == {(20, 45)}


def test_vcs_align_captures_a_pulse_at_d_from_the_set_start_at_the_capture_edge(tmp_path):
    from setfi.analysis.exposure import accumulate
    from setfi.analysis.patterns import reference_patterns
    hold0 = {"posedge:D": {"posedge:CP": {"SETUP": [{"value": 0.030}], "HOLD": [{"value": 0.0}]}},
             "negedge:D": {"posedge:CP": {"SETUP": [{"value": 0.030}], "HOLD": [{"value": 0.0}]}}}
    off = run(tmp_path, "off", checks=hold0)
    on = run(tmp_path, "on", vcs_align=True, checks=hold0)
    assert rows(on) == rows(off)           # the records of this cone do not change
    at_T = lambda r: {(int(c), r.sites[int(s)], int(w)) for c, s, w, k in zip(  # noqa: E731
        *(compute_patterns(r, 4).grid_points()[x] for x in ("cycle", "site", "width_ps", "k"))) if k == 4}
    # hold 0: enter = T, so a pulse at D from s = T is not inside the window ...
    assert at_T(off) == set()
    # ... except with vcs_align, for the records with t_enter = 0 (the SETs on n2)
    assert at_T(on) == {(c, "n2", w) for c in (1, 2) for w in (10, 20, 40)}
    for r in (off, on):
        ref, fast = reference_patterns(r, 4), compute_patterns(r, 4)
        assert ref.pattern_ffs.tolist() == fast.pattern_ffs.tolist()
        assert ref.run_k_hi.tolist() == fast.run_k_hi.tolist()
    w = {10: 1 / 3, 20: 1 / 3, 40: 1 / 3}
    for dom, extra in (("half_open", 0), ("closed", 2)):
        a, b = accumulate(off, w, dom), accumulate(on, w, dom)
        assert (b.exposure_by_width - a.exposure_by_width).tolist() == [extra] * 3
