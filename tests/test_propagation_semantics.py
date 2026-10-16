"""Propagation semantics on hand-written timing graphs (docs/MODEL.md,
Propagation): inertial filtering, and the delay of simultaneous input changes.

Each graph is written as build output (see test_campaign_physics.py) from a list
of gates; the gate under test drives the D pin of FF "ff", so its output pulse
is what the campaign records.  Delays in ps; SET widths are chosen so that the
buffers/inverters in front of the gate under test do not filter the pulse.
"""
import json
import os
from typing import Dict, List

import numpy as np
import pytest

from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.records import Records

T = 1000                                  # clock period; horizon 1250 ps


def gate(inst, cell, out, ins, truth, arcs):
    """ins: [(pin, net)] in truth-table bit order; arcs: {pin: dict(rise, fall, side=[pins],
    patt=[(mask, valbits, delay, "R"|"F")])}, mask bit i = side pin i."""
    return dict(inst=inst, cell=cell, out=out, ins=ins, truth=truth, arcs=arcs)


def buf(inst, a, z, rise, fall):
    return gate(inst, "BUF", ("Z", z), [("I", a)], [0, 1], {"I": dict(rise=rise, fall=fall)})


def inv(inst, a, z, rise, fall):
    return gate(inst, "INV", ("ZN", z), [("I", a)], [1, 0], {"I": dict(rise=rise, fall=fall)})


def truth_of(fn, n):
    return [int(bool(fn(*[(i >> j) & 1 for j in range(n)]))) for i in range(1 << n)]


def settle(gates, src_vals: Dict[str, int]) -> Dict[str, int]:
    v = dict(src_vals)
    for g in gates:                                  # gates are listed in topological order
        idx = sum(v[n] << j for j, (_p, n) in enumerate(g["ins"]))
        v[g["out"][1]] = g["truth"][idx]
    return v


def write_graph(d, gates, src_vals, site, dnet):
    """Build output for `gates`, one injection site, one observed FF on `dnet`."""
    os.makedirs(d, exist_ok=True)
    vals = settle(gates, src_vals)
    nets = sorted(vals)
    idx = {n: i for i, n in enumerate(nets)}
    j = lambda name, obj: json.dump(obj, open(os.path.join(d, name), "w"))  # noqa: E731
    j("net_index.json", {"net_to_idx": idx, "idx_to_net": {str(i): n for n, i in idx.items()}})
    edges, cell_arcs, pin_to_net, keys = [], {}, {"ff.D": dnet}, []
    rise, fall, ptr, pm, pv, pdt, pr = [], [], [0], [], [], [], []
    for g in gates:
        opin, onet = g["out"]
        pin_to_net[f"{g['inst']}.{opin}"] = onet
        order = [p for p, _n in g["ins"]]
        for p, n in g["ins"]:
            pin_to_net[f"{g['inst']}.{p}"] = n
            a = g["arcs"].get(p)
            if a is None:
                continue
            mk, dk = f"{g['cell']}:{p}->{opin}", f"{g['inst']}:{p}->{opin}"
            edges.append(dict(eid=len(edges), src_net=n, dst_net=onet, inst=g["inst"],
                              celltype=g["cell"], inpin=p, outpin=opin, mask_key=mk, delay_key=dk))
            cell_arcs[mk] = {"vars_order": order, "truth": g["truth"], "side_pins": a.get("side", [])}
            keys.append(dk)
            rise.append(a["rise"])
            fall.append(a["fall"])
            for m, bits, dt, rf in a.get("patt", []):
                pm.append(m), pv.append(bits), pdt.append(dt), pr.append(1 if rf == "R" else 0)
            ptr.append(len(pm))
    j("edges.json", edges)
    j("pin_to_net.json", pin_to_net)
    j("cell_arcs.json", cell_arcs)
    j("arc_index.json", {"delay_key_to_arc_idx": {k: i for i, k in enumerate(keys)}})
    np.savez(os.path.join(d, "delay_table.npz"), default_rise_ps=np.array(rise, np.int64),
             default_fall_ps=np.array(fall, np.int64), patt_ptr=np.array(ptr, np.int32),
             patt_mask=np.array(pm, np.int64), patt_valbits=np.array(pv, np.int64),
             patt_dt_ps=np.array(pdt, np.int64), patt_is_rise=np.array(pr, np.uint8))
    # the site's cone: every net reachable from it, in gate (topological) order
    cone, adj = [site], {}
    for e in edges:                                  # edges follow the gate order
        if e["src_net"] in cone:
            adj.setdefault(e["src_net"], []).append(e["eid"])
            if e["dst_net"] not in cone:
                cone.append(e["dst_net"])
    with open(os.path.join(d, "site_cones.jsonl"), "w") as f:
        f.write(json.dumps({"src_net_idx": idx[site], "cone_topo_net_idxs": [idx[n] for n in cone],
                            "cone_adj_idx": [[idx[n], e] for n, e in adj.items()],
                            "reachable_dnet_idxs": [idx[dnet]]}) + "\n")
    j("ff_index.json", {"ffid_to_info": {"0": {"ff_inst": "ff", "d_net": dnet, "d_pin": "D",
                                               "celltype": "DFF"}}})
    chk = {"posedge:CP": {"SETUP": [{"value": 0.03}], "HOLD": [{"value": 0.005}]}}
    j("timing_checks.json", {"index": {"ff": {"meta": {"ff_inst": "ff"},
                                              "checks": {"posedge:D": chk, "negedge:D": chk}}}})
    return vals, idx


def pulses(tmp_path, gates, src_vals, site, dnet, widths, **spec_kw):
    """D-pin pulses {width: [(base, t_enter, t_exit), ...]} of one campaign."""
    b, s, o = (str(tmp_path / x) for x in ("build", "state", "out"))
    vals, idx = write_graph(b, gates, src_vals, site, dnet)
    os.makedirs(s, exist_ok=True)
    with open(os.path.join(s, "cycle_1.hex"), "w") as f:
        f.write(f"{sum(v << idx[n] for n, v in vals.items()):x}\n")
    run_campaign(CampaignSpec(stage_dir=b, state_dir=s, cycles=[1], out_dir=o, clock_period_ps=T,
                              pulse_widths_ps=list(widths), **spec_kw), log=lambda *a: None)
    c = Records.open(o).load_all()
    out: Dict[int, List] = {w: [] for w in widths}
    for i in range(c["cycle"].size):
        out[int(c["width_ps"][i])].append((int(c["base"][i]), int(c["t_enter"][i]),
                                           int(c["t_exit"][i])))
    return {w: sorted(v) for w, v in out.items()}


# --------------------------------------------------------------- inertial filtering
# An OR4 sees A2 = [66, 119) and A4 = [100, 172): a 110-ps SET on s reaches A2
# through a buffer of 66/9 ps and A4 through one of 100/62 ps.
OR4 = truth_of(lambda a1, a2, a3, a4: a1 | a2 | a3 | a4, 4)
OR4_ARCS = {"A1": dict(rise=58, fall=74), "A2": dict(rise=56, fall=72),
            "A3": dict(rise=40, fall=60), "A4": dict(rise=41, fall=57)}


def or4_graph():
    return [buf("b2", "s", "x2", 66, 9), buf("b4", "s", "x4", 100, 62),
            gate("u", "OR4", ("Z", "z"), [("A1", "a1"), ("A2", "x2"), ("A3", "a3"), ("A4", "x4")],
                 OR4, OR4_ARCS)]


def test_a_pulse_that_another_input_still_drives_passes(tmp_path):
    got = pulses(tmp_path, or4_graph(), dict(s=0, a1=0, a3=0), "s", "z", [110])
    # A2's rise schedules Z at 66 + 56 = 122.  A4 rises at 100 and keeps Z's new
    # value when A2 falls at 119, so the rise stands; Z falls 57 ps after A4 does.
    assert got == {110: [(0, 122, 229)]}


def test_a_pulse_narrower_than_the_gate_delay_dies(tmp_path):
    # only A4 sees the pulse (A2's buffer filters it: 30, 60 < 66); a 30-ps pulse
    # dies at the OR4 (30 < 41), a 60-ps one passes
    g = or4_graph()
    g[1] = buf("b4", "s", "x4", 20, 20)
    got = pulses(tmp_path, g, dict(s=0, a1=0, a3=0), "s", "z", [30, 60])
    assert got == {30: [], 60: [(0, 20 + 41, 60 + 20 + 57)]}


# ----------------------------------------------------- simultaneous input changes
# AOI211, ZN = !(A1 & A2 | B | C), A1 = C = 0; A2 and B fall together (a SET on s
# through two inverters of 20 ps) and rise together again.  Only B alone explains
# ZN's rise (A1 = 0), but A2's path is active too.
AOI211 = truth_of(lambda a1, a2, b, c: not ((a1 & a2) | b | c), 4)


def aoi211_graph(a2_patt, b_patt, a2_def=(31, 25), b_def=(37, 23)):
    # A2 -> ZN side pins (A1, B, C), B -> ZN side pins (A1, A2, C); bit i = side pin i
    arcs = {"A2": dict(rise=a2_def[0], fall=a2_def[1], side=["A1", "B", "C"], patt=a2_patt),
            "B": dict(rise=b_def[0], fall=b_def[1], side=["A1", "A2", "C"], patt=b_patt)}
    return [inv("ia", "s", "x2", 20, 20), inv("ib", "s", "xb", 20, 20),
            gate("u", "AOI211", ("ZN", "zn"), [("A1", "a1"), ("A2", "x2"), ("B", "xb"), ("C", "c")],
                 AOI211, arcs)]


# both paths conditional, both CONDs true after the change, different delays:
# rise A2 32 (A1=0,B=0,C=0), B 34 (A1=0,A2=0,C=0); fall A2 26, B 21 (A2/B high again)
A2_PATT = [(7, 0, 32, "R"), (7, 1, 30, "R"), (7, 2, 26, "F"), (7, 1, 24, "F")]
B_PATT = [(7, 0, 34, "R"), (7, 2, 37, "R"), (7, 1, 38, "R"),
          (7, 0, 23, "F"), (7, 2, 21, "F"), (7, 1, 23, "F")]
SRC_B = dict(s=0, a1=0, c=0)                         # x2 = xb = 1, zn = 0


@pytest.mark.parametrize("delay_choice", ["max", "min"])
def test_simultaneous_changes_take_the_smallest_active_delay(tmp_path, delay_choice):
    w = 200                                          # A2, B fall at 20 and rise at 220
    got = pulses(tmp_path, aoi211_graph(A2_PATT, B_PATT), SRC_B, "s", "zn", [w],
                 delay_choice=delay_choice)
    assert got == {w: [(0, 20 + 32, w + 20 + 21)]}  # rise via A2 (32 < 34), fall via B (21 < 26)


def test_inactive_paths_do_not_compete(tmp_path):
    # A2's rise CONDs need A1 = 1: inactive after the change, so its (smaller)
    # default rise delay 10 is not used and B (34) alone sets the rise
    a2 = [(7, 1, 30, "R"), (7, 2, 26, "F")]
    got = pulses(tmp_path, aoi211_graph(a2, B_PATT, a2_def=(10, 25)), SRC_B, "s", "zn", [200])
    assert got == {200: [(0, 20 + 34, 200 + 20 + 21)]}


@pytest.mark.parametrize("delay_choice", ["max", "min"])
def test_without_an_active_path_the_inputs_that_explain_the_change_compete(tmp_path, delay_choice):
    # no rise COND of A2 or B holds: B, the only input that alone explains the
    # rise, takes its default delay 37 (delay_choice has one candidate)
    a2 = [(7, 1, 30, "R"), (7, 2, 26, "F")]
    b = [(7, 1, 38, "R"), (7, 2, 21, "F")]
    g = aoi211_graph(a2, b, a2_def=(10, 25), b_def=(37, 23))
    got = pulses(tmp_path, g, SRC_B, "s", "zn", [200], delay_choice=delay_choice)
    assert got == {200: [(0, 20 + 37, 200 + 20 + 21)]}


NOR2 = truth_of(lambda a1, a2: not (a1 | a2), 2)
# fall CONDs need the other input low (false once both have risen); rise CONDs hold
NOR2_ARCS = {"A1": dict(rise=35, fall=40, side=["A2"], patt=[(1, 0, 30, "F"), (1, 0, 25, "R")]),
             "A2": dict(rise=38, fall=45, side=["A1"], patt=[(1, 0, 33, "F"), (1, 0, 28, "R")])}


@pytest.mark.parametrize("delay_choice, fall", [("max", 45), ("min", 40)])
def test_without_an_active_path_delay_choice_picks_a_default_delay(tmp_path, delay_choice, fall):
    # A1 and A2 rise together at 20 and each alone explains ZN's fall; no fall
    # COND holds, so their default delays 40 and 45 compete.  They fall together
    # at 220, where both rise CONDs hold: the smaller one (25) is used.
    g = [inv("i1", "s", "x1", 20, 20), inv("i2", "s", "x2", 20, 20),
         gate("u", "NOR2", ("ZN", "zn"), [("A1", "x1"), ("A2", "x2")], NOR2, NOR2_ARCS)]
    got = pulses(tmp_path, g, dict(s=1), "s", "zn", [200], delay_choice=delay_choice)
    assert got == {200: [(1, 20 + fall, 200 + 20 + 25)]}


# ------------------------------------------------------------ single input changes
# One input changes at a time: the delay of its COND entry that holds, or else its
# default delay, whatever delay_choice says; inertial filtering as above.
ND2 = truth_of(lambda a1, a2: not (a1 & a2), 2)
ND2_ARCS = {"A1": dict(rise=20, fall=25, side=["A2"], patt=[(1, 1, 18, "R"), (1, 1, 23, "F")]),
            "A2": dict(rise=22, fall=27, side=["A1"], patt=[(1, 0, 40, "R")])}


@pytest.mark.parametrize("delay_choice", ["max", "min"])
def test_single_input_changes(tmp_path, delay_choice):
    g = [inv("u1", "a", "n1", 10, 12),
         gate("u2", "ND2", ("ZN", "n2"), [("A1", "n1"), ("A2", "b")], ND2, ND2_ARCS),
         buf("u3", "n2", "z", 15, 15)]
    widths = [10, 20, 30, 60]
    # SET on n1 (1 -> 0 -> 1) with b = 1: A1's CONDs hold (A2 = 1): rise 18, fall 23;
    # a 10-ps pulse dies at u2 (10 < 18)
    got = pulses(tmp_path / "n1", g, dict(a=0, b=1), "n1", "z", widths, delay_choice=delay_choice)
    assert got == {10: [], 20: [(0, 18 + 15, 20 + 23 + 15)], 30: [(0, 33, 68)], 60: [(0, 33, 98)]}
    # SET on b (1 -> 0 -> 1) with n1 = 1: A2's only COND (A1 = 0) is false: default
    # delays rise 22 / fall 27; a 20-ps pulse dies (20 < 22)
    got = pulses(tmp_path / "b", g, dict(a=0, b=1), "b", "z", widths, delay_choice=delay_choice)
    assert got == {10: [], 20: [], 30: [(0, 22 + 15, 30 + 27 + 15)], 60: [(0, 37, 102)]}
