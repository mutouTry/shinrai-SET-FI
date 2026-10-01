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
from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.records import INF_PS, Records

NETS = ["a", "b", "n1", "n2"]          # net_index order (sorted)


def write_substrate(d):
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
                          "ffC.D": "n2"})
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
    j("timing_checks.json", {"index": {"ffC": {"meta": {"ff_inst": "ffC"}, "checks": {
        "posedge:D": chk(0.030, 0.005), "negedge:D": chk(0.030, 0.005)}}}})
    keys = ["u1:I->ZN", "u2:A1->ZN", "u2:A2->ZN"]
    j("arc_index.json", {"delay_key_to_arc_idx": {k: i for i, k in enumerate(keys)}})
    np.savez(os.path.join(d, "delay_table.npz"),
             default_rise_ps=np.array([10, 20, 22]), default_fall_ps=np.array([12, 25, 27]),
             patt_ptr=np.zeros(4, np.int32), patt_mask=np.zeros(0, np.int64),
             patt_valbits=np.zeros(0, np.int64), patt_dt_ps=np.zeros(0, np.int64),
             patt_is_rise=np.zeros(0, np.uint8))


def write_state(d):
    os.makedirs(d, exist_ok=True)
    # bit i = value of NETS[i]
    for cyc, vals in {1: dict(a=0, b=1, n1=1, n2=0), 2: dict(a=0, b=0, n1=1, n2=1)}.items():
        v = sum(vals[n] << i for i, n in enumerate(NETS))
        open(os.path.join(d, f"cycle_{cyc}.hex"), "w").write(f"{v:x}\n")


@pytest.fixture
def campaign(tmp_path):
    write_substrate(str(tmp_path / "build"))
    write_state(str(tmp_path / "state"))
    spec = CampaignSpec(stage_dir=str(tmp_path / "build"), state_dir=str(tmp_path / "state"),
                        cycles=[1, 2], out_dir=str(tmp_path / "out"), clock_period_ps=100,
                        pulse_widths_ps=[10, 20, 40], phase_grid_steps=100)
    run_campaign(spec, log=lambda *a: None)
    return Records.open(str(tmp_path / "out")), tmp_path


def rows(rec):
    c = rec.load_all()
    sites = rec.sites
    return sorted((int(c["cycle"][i]), sites[int(c["site"][i])], int(c["width_ps"][i]), int(c["base"][i]),
                   int(c["t_enter"][i]), int(c["t_exit"][i])) for i in range(c["cycle"].size))


def test_expected_pulses(campaign):
    rec, _ = campaign
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
    rec, _ = campaign
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


def test_phase_grid_matches_records(campaign):
    rec, tmp = campaign
    enter, leave = rec.thresholds()
    cols = rec.load_all()
    keep, lo, hi = capture_intervals(cols, enter, leave, 101)
    from_records = set()
    for i, a, b in zip(np.flatnonzero(keep), lo, hi):
        for s in range(a, b):
            from_records.add((int(cols["cycle"][i]), rec.sites[int(cols["site"][i])], s,
                              int(cols["width_ps"][i])))
    grid = set()
    for line in open(tmp / "out" / "phase_grid_captures.csv").read().splitlines()[1:]:
        c, s, st, pw, ffs = line.split(",")
        assert ffs == "0"
        grid.add((int(c), s, int(st), int(pw)))
    assert grid == from_records
