"""The logic-only translation (setfi.campaign.logic, `setfi logic`) against a
whole-netlist Boolean evaluation in plain Python, on random timing graphs and
on a hand-written netlist that goes through the build stage; the pattern-table
format; the stage bookkeeping."""
import json
import os
import random
import signal
import time
from collections import defaultdict, deque

import numpy as np
import pytest

from setfi import _engine, cli
from setfi.analysis.patterns import Patterns, compute_patterns
from setfi.build import BuildSpec, LibrarySpec, run_build
from setfi.campaign import fastpath
from setfi.campaign.logic import LogicSpec, run_logic
from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.campaign.state import read_cycle_bits
from setfi.campaign.substrate import SubstrateError, load_substrate
from test_fastpath import CLOCK, write_random_substrate
from test_patterns import assert_same


# --------------------------------------------------------------------------- reference
def netlist(build_dir):
    """The net names (by index) and every combinational gate of a build as
    (output, inputs, truth table), in topological order, from edges.json,
    pin_to_net.json and cell_arcs.json."""
    load = lambda name: json.load(open(os.path.join(build_dir, name)))  # noqa: E731
    names = [n for _, n in sorted((int(i), n) for i, n in load("net_index.json")["idx_to_net"].items())]
    index = {n: i for i, n in enumerate(names)}
    pins, arcs = load("pin_to_net.json"), load("cell_arcs.json")
    gates = {}
    for e in load("edges.json"):
        key, ent = (e["inst"], e["outpin"]), arcs.get(e["mask_key"])
        if key not in gates and isinstance(ent, dict) and ent["truth"]:
            ins = [index[pins[f"{e['inst']}.{p}"]] for p in ent["vars_order"]]
            gates[key] = (index[e["dst_net"]], ins, ent["truth"])
    driver = {out: key for key, (out, _, _) in gates.items()}
    readers, waiting = defaultdict(list), {}
    for key, (_, ins, _) in gates.items():
        deps = {n for n in ins if n in driver}
        waiting[key] = len(deps)
        for n in deps:
            readers[n].append(key)
    ready, order = deque(sorted(k for k, w in waiting.items() if w == 0)), []
    while ready:
        key = ready.popleft()
        order.append(gates[key])
        for r in readers[gates[key][0]]:
            waiting[r] -= 1
            if waiting[r] == 0:
                ready.append(r)
    assert len(order) == len(gates), "combinational loop"
    return names, order


def settle(gates, values, hold=None):
    """The values with every gate output recomputed in order; net `hold` keeps its value."""
    v = list(values)
    for out, ins, truth in gates:
        if out != hold:
            v[out] = truth[sum(v[n] << j for j, n in enumerate(ins))]
    return v


def reference_upsets(build_dir, state_dir, cycles, sites, ff_ids):
    """{(cycle, site): FF ids} for every pair with a SET that changes a data pin:
    the whole netlist evaluated with and without the site inverted."""
    names, gates = netlist(build_dir)
    index = {n: i for i, n in enumerate(names)}
    info = json.load(open(os.path.join(build_dir, "ff_index.json")))["ffid_to_info"]
    dnet = {f: index[info[str(f)]["d_net"]] for f in ff_ids}
    out = {}
    for c in cycles:
        good = read_cycle_bits(state_dir, c, len(names)).tolist()
        assert settle(gates, good) == good, "the recorded state must be consistent"
        for s in sites:
            bad = list(good)
            bad[index[s]] ^= 1
            bad = settle(gates, bad, hold=index[s])
            ffs = [f for f in ff_ids if bad[dnet[f]] != good[dnet[f]]]
            if ffs:
                out[(c, s)] = ffs
    return out


def write_consistent_states(build_dir, state_dir, sources, seed=0):
    """cycle_<N>.hex for every cycle of `sources` ({cycle: {net: value}}): nets that
    no gate drives get the given value (else a random one), gate outputs are settled."""
    names, gates = netlist(build_dir)
    driven = {out for out, _, _ in gates}
    rng = random.Random(seed)
    os.makedirs(state_dir, exist_ok=True)
    for c, given in sources.items():
        v = [0 if i in driven else int(given.get(n, rng.getrandbits(1))) for i, n in enumerate(names)]
        v = settle(gates, v)
        with open(os.path.join(state_dir, f"cycle_{c}.hex"), "w") as f:
            f.write(f"{sum(b << i for i, b in enumerate(v)):x}\n")


def upsets_of(pt: Patterns):
    """{(cycle, site): FF ids} from the runs of a logic-only table."""
    out = {}
    for p in range(pt.count):
        for src in pt.sources(p):
            assert (src.width_ps, src.k_lo, src.k_hi, src.start_lo_ps) == (0, 0, 0, 0)
            assert (pt.cycle(p), src.site) not in out
            out[(pt.cycle(p), src.site)] = pt.ffs(p)
    return out


# --------------------------------------------------------------------------- random graphs
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("jobs", [1, 3])
def test_random_graphs_match_the_whole_netlist_evaluation(tmp_path, seed, jobs):
    # gate outputs missing from net_to_idx (settled from their inputs), pins that
    # share a net, interconnect delays (ignored here)
    b, s = str(tmp_path / "build"), str(tmp_path / "state")
    write_random_substrate(b, seed)
    cycles = [3, 5, 8, 13]
    write_consistent_states(b, s, {c: {} for c in cycles}, seed)
    pt = run_logic(LogicSpec(stage_dir=b, state_dir=s, cycles=cycles, jobs=jobs), log=lambda *a: None)
    assert pt.count > 0
    assert upsets_of(pt) == reference_upsets(b, s, cycles, pt.sites.tolist(), pt.ff_ids.tolist())


def test_same_sites_cycles_and_flip_flops_as_the_patterns_stage(tmp_path):
    b, s = str(tmp_path / "build"), str(tmp_path / "state")
    write_random_substrate(b, 4)
    cycles = [9, 2, 7]                                    # in campaign order, not sorted
    write_consistent_states(b, s, {c: {} for c in cycles}, 4)
    sites = sorted(load_substrate(b, clock_pins=CLOCK).skeleton)[2:9]
    run_campaign(CampaignSpec(stage_dir=b, state_dir=s, cycles=cycles, out_dir=str(tmp_path / "inject"),
                              clock_period_ps=100, pulse_widths_ps=[20, 60], sites=sites), log=lambda *a: None)
    timed = compute_patterns(str(tmp_path / "inject"), steps=10)
    logic = run_logic(LogicSpec(stage_dir=b, state_dir=s, cycles=cycles, sites=sites), log=lambda *a: None)
    for name in ("cycles", "sites", "ff_ids", "ff_instances"):
        a, z = getattr(logic, name), getattr(timed, name)
        assert a.dtype == z.dtype and a.tolist() == z.tolist(), name
    with pytest.raises(SubstrateError, match=r"1 name\(s\) are not injection sites"):
        run_logic(LogicSpec(stage_dir=b, state_dir=s, cycles=cycles, sites=sites + ["nope"]), log=lambda *a: None)


def test_logic_block_does_not_depend_on_the_thread_count(tmp_path):
    b, s = str(tmp_path / "build"), str(tmp_path / "state")
    names = write_random_substrate(b, 11)
    write_consistent_states(b, s, {1: {}, 2: {}}, 11)
    sub = load_substrate(b, clock_pins=CLOCK, n_threads=2)
    cones = fastpath.compile_sites(sub, sorted(sub.skeleton))
    bits = np.stack([read_cycle_bits(s, c, len(names)) for c in (1, 2)])
    ids = np.arange(len(cones), dtype=np.int32) + 100
    runs = [_engine.logic_block(cones, bits, np.array([1, 2]), ids, nt) for nt in (1, 2, 5)]
    assert runs[0]["ff"].size > 0 and set(runs[0]["site"].tolist()) <= set(ids.tolist())
    for r in runs[1:]:
        for k, v in runs[0].items():
            assert v.dtype == r[k].dtype and np.array_equal(v, r[k]), k


def test_cones_with_wires_are_refused(tmp_path):
    write_random_substrate(str(tmp_path), 3)
    sub = load_substrate(str(tmp_path), clock_pins=CLOCK, interconnect="transport", n_threads=2)
    cones = fastpath.compile_sites(sub, sorted(sub.skeleton))
    bits = np.zeros((1, len(sub.idx_to_net)), np.uint8)
    with pytest.raises(RuntimeError, match="without interconnect wires"):
        _engine.logic_block(cones, bits, np.array([1]), np.arange(len(cones), dtype=np.int32), 1)


def chain_cone(n):
    """A cone of n inverters in a row, nets 0 .. n (net i recorded as global net i);
    FF 0 on net n.  Its consistent states: chain_state(n)."""
    i32 = lambda x: np.asarray(x, dtype=np.int32)  # noqa: E731
    empty = i32([])
    return _engine.CompiledCone.from_arrays(dict(
        fanout_ptr=i32(list(range(n + 1)) + [n]), fanout_idx=i32(range(n)), gate_out=i32(range(1, n + 1)),
        gate_var_ptr=i32(range(n + 1)), gate_var_nets=i32(range(n)), gate_truth_ptr=i32(range(0, 2 * n + 1, 2)),
        gate_truth=np.tile(np.asarray([1, 0], np.uint8), n), arc_dkey_id=i32([-1] * n), arc_cause_i=i32([-1] * n),
        arc_side_mode=np.zeros(n, np.int8), arc_side_ptr=i32([0] * (n + 1)), arc_side_nets=empty,
        rec_idx=i32([-1] * n + [0]), wire_sink=empty, ffids=i32([0]), dp_rec_slot=i32([0]), dp_ptr=i32([0, 1]),
        dp_idx=i32([0]), src_i=0, gidx=i32(range(n + 1)), settle_gate=empty, settle_in_ptr=i32([0]),
        settle_in=empty, wp_drv=empty, wp_sink=empty, dp_g=np.asarray([-1], np.int64)))


def chain_state(n, n_cycles):
    return np.tile((np.arange(n + 1) % 2).astype(np.uint8), (n_cycles, 1))


def test_a_long_chain():
    out = _engine.logic_block([chain_cone(5000)], chain_state(5000, 3), np.array([4, 5, 6]), np.array([7]), 1)
    assert out["cycle"].tolist() == [4, 5, 6] and out["site"].tolist() == [7] * 3 and out["ff"].tolist() == [0] * 3


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
def test_logic_block_stops_on_a_signal():
    # as run_block: Ctrl-C (here a SIGALRM handler that raises) does not wait for the
    # end of the block; uninterrupted, the block below takes several seconds
    cones = [chain_cone(100_000)] * 2000
    bits = chain_state(100_000, 16)

    class Interrupted(Exception):
        pass

    def handler(signum, frame):
        raise Interrupted

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        t = time.time()
        with pytest.raises(Interrupted):
            _engine.logic_block(cones, bits, np.arange(16), np.arange(len(cones), dtype=np.int32), 2)
        assert time.time() - t < 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# --------------------------------------------------------------------------- a netlist
CELLS = """
module INV (input A, output ZN);
  assign ZN = !A;
endmodule
module BUF (input A, output Z);
  assign Z = A;
endmodule
module AN2 (input A1, input A2, output Z);
  assign Z = A1 & A2;
endmodule
module XOR2 (input A1, input A2, output Z);
  assign Z = A1 ^ A2;
endmodule
module HA (input A, input B, output S, output CO);
  assign S = A ^ B;
  assign CO = A & B;
endmodule
module DFF (input D, input CP, output reg Q);
  always @(posedge CP) Q <= D;
endmodule
"""

# instance: (cell, {pin: net}); f<i> is FF id i (ids follow the instance names)
GATES = {
    "u0": ("AN2", {"A1": "q0", "A2": "a", "Z": "d0"}),      # d0 is a data net itself
    "u1": ("INV", {"A": "q1", "ZN": "s1"}),                 # s1 drives f1 and f2 ...
    "u2": ("BUF", {"A": "s1", "Z": "b1"}),                  # ... and f3 through a buffer
    "u3": ("INV", {"A": "q3", "ZN": "x1"}),                 # x1 reaches the XOR twice
    "u4": ("BUF", {"A": "x1", "Z": "x2"}),
    "u5": ("INV", {"A": "x1", "ZN": "x3"}),
    "u6": ("XOR2", {"A1": "x2", "A2": "x3", "Z": "xo"}),
    "u7": ("INV", {"A": "q4", "ZN": "m"}),
    "u8": ("AN2", {"A1": "m", "A2": "q5", "Z": "n"}),       # q5 = 0 blocks m
    "u9": ("INV", {"A": "q6", "ZN": "h"}),
    "u10": ("HA", {"A": "h", "B": "q7", "S": "s", "CO": "co"}),   # two outputs
}
FF_D = {"f0": "d0", "f1": "s1", "f2": "s1", "f3": "b1", "f4": "xo", "f5": "n", "f6": "s", "f7": "co"}
ARCS = {"INV": [("A", "ZN")], "BUF": [("A", "Z")], "AN2": [("A1", "Z"), ("A2", "Z")],
        "XOR2": [("A1", "Z"), ("A2", "Z")], "HA": [("A", "S"), ("B", "S"), ("A", "CO"), ("B", "CO")]}
# source values of two cycles: q5 = q7 = 0 in cycle 10, 1 in cycle 11
SOURCES = {10: {"a": 0, **{f"q{i}": 0 for i in range(8)}},
           11: {"a": 1, **{f"q{i}": int(i in (1, 5, 7)) for i in range(8)}}}
# cycle -> {FF ids: the sites that upset exactly these FFs}
EXPECTED = {
    10: {(0,): ["d0"], (1, 2, 3): ["s1"], (3,): ["b1"], (4,): ["x2", "x3", "xo"], (5,): ["n"],
         (6,): ["h", "s"], (7,): ["co"]},
    11: {(0,): ["d0"], (1, 2, 3): ["s1"], (3,): ["b1"], (4,): ["x2", "x3", "xo"], (5,): ["m", "n"],
         (6,): ["s"], (6, 7): ["h"], (7,): ["co"]},
}


def write_design(d):
    """Cells, netlist and SDF of the design above in directory d."""
    nets = {n for _, pins in GATES.values() for n in pins.values()} | {f"q{i}" for i in range(len(FF_D))}
    body = [f"  DFF {f} (.D({dn}), .CP(clk), .Q(q{f[1:]}));" for f, dn in FF_D.items()]
    body += [f"  {cell} {inst} (" + ", ".join(f".{p}({n})" for p, n in pins.items()) + ");"
             for inst, (cell, pins) in GATES.items()]
    (d / "cells.v").write_text(CELLS)
    (d / "top.v").write_text(f"module top (clk, a);\n  input clk, a;\n  wire {', '.join(sorted(nets - {'a'}))};\n"
                             + "\n".join(body)
                             + "\nendmodule\n")
    sdf = ['(DELAYFILE (SDFVERSION "3.0") (DESIGN "top") (TIMESCALE 1ns) (DIVIDER /)']
    for f in FF_D:
        sdf.append(f'  (CELL (CELLTYPE "DFF") (INSTANCE {f})'
                   ' (DELAY (ABSOLUTE (IOPATH (posedge CP) Q (0.05) (0.05))))'
                   ' (TIMINGCHECK (SETUP (posedge D) (posedge CP) (0.02)) (SETUP (negedge D) (posedge CP) (0.02))'
                   ' (HOLD (posedge D) (posedge CP) (0.005)) (HOLD (negedge D) (posedge CP) (0.005))))')
    for inst, (cell, _) in GATES.items():
        paths = " ".join(f"(IOPATH {i} {o} (0.02) (0.03))" for i, o in ARCS[cell])
        sdf.append(f'  (CELL (CELLTYPE "{cell}") (INSTANCE {inst}) (DELAY (ABSOLUTE {paths})))')
    (d / "top.sdf").write_text("\n".join(sdf) + "\n)\n")


@pytest.fixture
def design(tmp_path):
    write_design(tmp_path)
    b, s = str(tmp_path / "build"), str(tmp_path / "state")
    run_build(BuildSpec(netlist=str(tmp_path / "top.v"), sdf=str(tmp_path / "top.sdf"), top="top", out_dir=b,
                        library=LibrarySpec(behavioral_verilog=str(tmp_path / "cells.v"))))
    write_consistent_states(b, s, SOURCES)
    return b, s


def test_the_netlist(design):
    b, s = design
    pt = run_logic(LogicSpec(stage_dir=b, state_dir=s, cycles=[11, 10], jobs=2), log=lambda *a: None)
    assert pt.sites.tolist() == ["b1", "co", "d0", "h", "m", "n", "s", "s1", "x1", "x2", "x3", "xo"]
    assert pt.ff_instances.tolist() == list(FF_D) and pt.ff_ids.tolist() == list(range(8))
    got = {c: {} for c in EXPECTED}
    for p in range(pt.count):
        got[pt.cycle(p)][tuple(pt.ffs(p))] = [src.site for src in pt.sources(p)]
    assert got == EXPECTED            # x1 upsets nothing: its two paths cancel in the XOR
    assert upsets_of(pt) == reference_upsets(b, s, [10, 11], pt.sites.tolist(), list(range(8)))
    # patterns sorted by cycle, then FF ids; multiplicity = number of sites
    assert [pt.cycle(p) for p in range(pt.count)] == [10] * 7 + [11] * 8
    assert pt.pattern_multiplicity.tolist() == pt.pattern_n_sites.tolist() == np.diff(pt.pattern_run_ptr).tolist()


def test_the_table_format(design, tmp_path):
    b, s = design
    pt = run_logic(LogicSpec(stage_dir=b, state_dir=s, cycles=[10, 11]), log=lambda *a: None)
    assert (pt.steps, pt.period_ps, pt.starts_ps.tolist(), pt.pulse_widths_ps.tolist()) == (0, 0, [0], [])
    assert pt.cycles.tolist() == [10, 11]
    path = str(tmp_path / "patterns.npz")
    pt.save(path)
    back = Patterns.load(path)
    assert_same(back, pt)
    with np.load(path) as z:
        assert all(z[k].dtype != object for k in z.files)
    g = back.grid_points()
    assert set(g["width_ps"].tolist()) == set(g["k"].tolist()) == set(g["start_ps"].tolist()) == {0}
    assert sorted(zip(g["cycle"].tolist(), back.sites[g["site"]].tolist())) == sorted(upsets_of(pt))
    pt.to_csv(str(tmp_path / "patterns.csv"))
    rows = open(tmp_path / "patterns.csv").read().splitlines()
    assert rows[0] == "pattern,cycle,multiplicity,n_ffs,n_sites,ffs"
    assert rows[1:4] == ["0,10,1,1,1,0", "1,10,1,3,1,1|2|3", "2,10,1,1,1,3"]


# --------------------------------------------------------------------------- the stage
def _config(d, extra=""):
    (d / "tb.sv").write_text("module tb; endmodule\n")
    (d / "setfi.toml").write_text(f"""
[design]
netlist = "top.v"
sdf = "top.sdf"
top = "top"
clock_period_ns = 1.0
[output]
dir = "out"
[library]
cell_functions = ["cells.v"]
[workload]
testbench = ["tb.sv"]
tb_top = "tb"
cell_sim_models = ["cells.v"]
{extra}""")
    return str(d / "setfi.toml")


def _record(path, sources):
    """The record stage, without a simulator: states written by the reference."""
    from setfi import config as C
    cfg = C.load(path)
    rec = cli.paths(cfg)["record"]
    write_consistent_states(cli.paths(cfg)["build"], os.path.join(rec, "states"), sources)
    with open(os.path.join(rec, "cycles.json"), "w") as f:
        json.dump({"format": "setfi-cycles/1", "cycles": list(sources)}, f)
    fp = cli._begin(cfg, "record")
    cli._finish(cfg, "record", fp, {"cycles": len(sources)})


def test_setfi_logic(tmp_path, capsys):
    write_design(tmp_path)
    path = _config(tmp_path, "[logic]\njobs = 2\n")
    assert cli.main(["logic", "-c", path]) == 1
    assert "the 'build' stage is not up to date" in capsys.readouterr().err
    assert cli.main(["build", "-c", path]) == 0
    _record(path, SOURCES)
    capsys.readouterr()
    assert cli.main(["status", "-c", path]) == 0
    assert "\nlogic " not in capsys.readouterr().out       # listed once it has run
    assert cli.main(["logic", "-c", path]) == 0
    assert "[logic] 2 cycles x 12 sites = 24 SETs" in capsys.readouterr().out
    out = tmp_path / "out" / "logic"
    assert sorted(os.listdir(out)) == ["patterns.csv", "patterns.npz"]
    pt = Patterns.load(str(out / "patterns.npz"))
    assert pt.count == 15 and pt.steps == 0
    assert cli.main(["logic", "-c", path]) == 0
    assert "[logic] up to date" in capsys.readouterr().out
    assert cli.main(["status", "-c", path]) == 0
    assert "logic    up to date" in capsys.readouterr().out

    # the keys it reads: jobs only changes how it runs; inject's keys do not matter
    text = open(path).read()
    open(path, "w").write(text.replace("jobs = 2", "jobs = 1") + "[fault_model]\npulse_widths_ps = [50]\n")
    assert cli.main(["logic", "-c", path]) == 0
    assert "[logic] up to date" in capsys.readouterr().out
    open(path, "a").write('sites = ["h", "x1"]\n')
    assert cli.main(["status", "-c", path]) == 0
    assert "logic    to run  (changed: fault_model.sites)" in capsys.readouterr().out
    assert cli.main(["logic", "-c", path]) == 0
    pt = Patterns.load(str(out / "patterns.npz"))
    assert pt.sites.tolist() == ["h", "x1"] and pt.count == 2          # x1 upsets nothing
    _record(path, {10: SOURCES[10]})
    ok, why = cli.stage_status(cli.C.load(path), "logic", cli.read_stamps(cli.C.load(path)))
    assert not ok and "the result of record" in why
