"""observe.clock_gates = "expand": what the build finds, how a pattern table is rewritten
(small hand-made cases, and against a brute-force table), and that "ignore" changes nothing."""
import json
import os

import numpy as np
import pytest

from setfi import cli
from setfi import config as C
from setfi.analysis.clock_gates import clock_gate_ffs, expand_clock_gates
from setfi.analysis.patterns import Patterns, _table, compute_patterns, phase_grid
from setfi.analysis.report import analyze
from setfi.build import BuildError, BuildSpec, LibrarySpec, run_build
from setfi.campaign.run import CampaignSpec, run_campaign
from setfi.campaign.state import read_cycle_bits, read_cycle_unknown
from setfi.library.liberty import liberty_to_verilog
from setfi.records import Records
from test_patterns import assert_same

# ---- the rewrite, on a hand-written clock_gates.json ----
#
# nets:  0 enable of g0   1 enable of g1   2 test enable of both
#        3.. 3 + 2n: D and Q of flip-flop n (D of ff n = 3 + 2n, Q = 4 + 2n)
# ids:   ff 0, 1 behind g0   ff 2 behind g1   ff 3 ungated   gates 10 (g0), 11 (g1)
N_NETS = 3 + 2 * 4
G0, G1 = 10, 11


def ff_row(n, gate, q_inverted=False, async_pins=(), observed=True, **extra):
    return {"inst": f"ff{n}", "gate": gate, "d_net_idx": 3 + 2 * n, "q_net_idx": 4 + 2 * n,
            "q_inverted": q_inverted, "async": [list(a) for a in async_pins], "observed": observed, **extra}


def gate_row(name, ffid, enable, ffs, parent=None, gates=(), blackbox=(), other=()):
    return {"inst": name, "celltype": "ICG", "type": "latch_posedge_precontrol", "ffid": ffid,
            "enable_net_idx": enable, "test_net_idx": 2, "parent": parent, "gates": list(gates), "ffs": list(ffs),
            "blackbox_pins": list(blackbox), "other_edge": list(other), "through_logic": 0}


def gates_json(series=False, blackbox=(), other=(), ff_kw=None):
    ff_kw = ff_kw or {}
    rows = [gate_row("g0", G0, 0, [0, 1], gates=["g1"] if series else [], blackbox=blackbox, other=other),
            gate_row("g1", G1, 1, [2], parent="g0" if series else None)]
    ffs = {str(n): ff_row(n, g, **ff_kw.get(n, {})) for n, g in ((0, "g0"), (1, "g0"), (2, "g1"))}
    return {"format": "setfi-clock-gates/1", "n_nets": N_NETS, "gates": rows, "ffs": ffs, "several_gates": [],
            "unmodelled_gates": [], "other_clock_sources": {}, "summary": {}}


def write_state(d, cycle, nets):
    """A state file: `nets` maps a net to 0, 1 or "x" (default 0)."""
    digits = ["0"] * ((N_NETS + 3) // 4)
    values = [0] * N_NETS
    unknown = set()
    for n, v in nets.items():
        if v == "x":
            unknown.add(n // 4)
        else:
            values[n] = v
    for i in range(len(digits)):
        digits[i] = "x" if i in unknown else format(sum(values[i * 4 + b] << b for b in range(4) if i * 4 + b < N_NETS), "x")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"cycle_{cycle}.hex"), "w") as f:
        f.write("".join(reversed(digits)) + "\n")


FF_IDS = [0, 1, 2, 3, G0, G1]


def table(rows, steps=10, ff_ids=FF_IDS):
    """A Patterns from {(cycle, FF ids): [(site, width, k_lo, k_hi), ...]}."""
    return Patterns(period_ps=100, steps=steps, starts_ps=phase_grid(100, steps),
                    cycles=np.asarray(sorted({c for c, _ in rows}), dtype=np.int64),
                    pulse_widths_ps=np.asarray([10, 20], dtype=np.int64), sites=np.asarray(["s0", "s1"], dtype=str),
                    ff_ids=np.asarray(ff_ids, dtype=np.int32),
                    ff_instances=np.asarray([f"x{i}" for i in ff_ids], dtype=str),
                    **_table({k: list(v) for k, v in rows.items()}))


def expand(tmp_path, nets, F, gates=None, steps=10, ff_ids=FF_IDS):
    state = str(tmp_path / "states")
    write_state(state, 7, nets)
    pt = table({(7, tuple(F)): [(0, 10, 2, 4)]}, steps=steps, ff_ids=ff_ids)
    return expand_clock_gates(pt, gates or gates_json(), state, log=lambda s: None)


def flips(pt):
    return [pt.ffs(p) for p in range(pt.count)]


def count(report, name):
    return report["counts"].get(name, {"patterns": 0})["patterns"]


def d(n):
    return 3 + 2 * n


def q(n):
    return 4 + 2 * n


def test_extra_edge(tmp_path):
    # g0 is off and the SET turns it on: ff 0 has D != Q and flips, ff 1 has D = Q and does not
    pt, rep = expand(tmp_path, {d(0): 1}, [G0])
    assert flips(pt) == [[0]]
    assert count(rep, "gate_in_F") == 1 and count(rep, "edge_change") == 1
    assert pt.multiplicity(0) == 3 and int(pt.pattern_n_sites[0]) == 1


def test_missing_edge(tmp_path):
    # g0 is on and the SET turns it off: the flip-flops keep Q, so those with D != Q differ
    pt, _ = expand(tmp_path, {0: 1, d(0): 1, q(1): 1}, [G0])
    assert flips(pt) == [[0, 1]]


def test_set_on_d_at_the_same_time(tmp_path):
    # extra edge: the edge captures D under the SET, so D = Q and f in F flips
    pt, _ = expand(tmp_path, {}, [0, G0])
    assert flips(pt) == [[0]]
    # extra edge with D != Q and f in F: the SET restores the old value
    pt, _ = expand(tmp_path, {d(0): 1}, [0, G0])
    assert pt.count == 0
    # missing edge: this edge is gone, so the SET on D does not act
    pt, _ = expand(tmp_path, {0: 1}, [0, G0])
    assert pt.count == 0


def test_d_equal_q_does_not_flip(tmp_path):
    pt, rep = expand(tmp_path, {0: 1, d(0): 1, q(0): 1}, [G0])
    assert pt.count == 0
    assert count(rep, "edge_change") == 1


def test_empty_result_is_counted_not_a_pattern(tmp_path):
    pt, rep = expand(tmp_path, {}, [G0])
    assert pt.count == 0 and pt.run_site.size == 0
    assert rep["counts"]["empty"] == {"patterns": 1, "grid_points": 3}
    assert rep["patterns_in"] == 1 and rep["patterns_out"] == 0
    assert rep["grid_points_in"] == 3 and rep["grid_points_out"] == 0


def test_flip_flop_without_a_fault_free_edge_does_not_capture(tmp_path):
    # no gate in F: g0 is off, so ff 0 has no edge; g1 is on and ff 3 is not gated
    pt, rep = expand(tmp_path, {1: 1}, [0, 2, 3])
    assert flips(pt) == [[2, 3]]
    assert count(rep, "no_fault_free_edge") == 1


def test_next_value_of_a_synchronous_clear(tmp_path):
    # ff 0 loads clear-pin AND data pin; the clear pin is net d(3)
    gates = gates_json(ff_kw={0: {"celltype": "SC", "next_nets": [d(3)]}})
    gates["next_tables"] = {"SC": [0, 0, 0, 1]}
    # data pin 1, clear active (0): the extra edge loads 0, equal to Q
    pt, _ = expand(tmp_path, {d(0): 1}, [G0], gates=gates)
    assert pt.count == 0
    # clear inactive: it loads the data pin
    pt, _ = expand(tmp_path, {d(0): 1, d(3): 1}, [G0], gates=gates)
    assert flips(pt) == [[0]]
    # Q = 1, data pin 1, clear active: a lost edge keeps 1 where the edge would have cleared it
    pt, _ = expand(tmp_path, {0: 1, d(0): 1, q(0): 1}, [G0], gates=gates)
    assert flips(pt) == [[0]]
    # an unknown clear pin is reported
    _, rep = expand(tmp_path, {d(0): 1, d(3): "x"}, [G0], gates=gates)
    assert count(rep, "unknown_value") == 1


def test_next_value_can_hold_the_stored_value(tmp_path):
    # an enable flip-flop: enable pin net d(3), table over (data pin, enable, stored value)
    gates = gates_json(ff_kw={0: {"celltype": "EN", "next_nets": [d(3), -3]}})
    gates["next_tables"] = {"EN": [0, 0, 0, 1, 1, 1, 0, 1]}
    # enable 0, Q = 1: it holds, no flip although D != Q
    pt, _ = expand(tmp_path, {q(0): 1}, [G0], gates=gates)
    assert pt.count == 0
    # enable 1: it loads the data pin
    pt, _ = expand(tmp_path, {q(0): 1, d(3): 1}, [G0], gates=gates)
    assert flips(pt) == [[0]]


def test_other_gate_unchanged(tmp_path):
    # g0 flips on, g1 stays on: ff 2 in F keeps its capture
    pt, _ = expand(tmp_path, {1: 1, d(0): 1}, [2, 3, G0])
    assert flips(pt) == [[0, 2, 3]]


def test_q_inverted_output(tmp_path):
    gates = gates_json(ff_kw={0: {"q_inverted": True}})
    # net Q = 1 means the stored value is 0, and D = 0: no flip
    pt, _ = expand(tmp_path, {q(0): 1}, [G0], gates=gates)
    assert pt.count == 0
    # net Q = 0 means the stored value is 1, and D = 0: it flips
    pt, _ = expand(tmp_path, {}, [G0], gates=gates)
    assert flips(pt) == [[0]]


def test_test_enable_on_masks_the_enable(tmp_path):
    # with the test enable on, the latch holds 1 whatever the enable does
    pt, rep = expand(tmp_path, {2: 1, d(0): 1}, [3, G0])
    assert flips(pt) == [[3]]
    assert count(rep, "test_enable_on") == 1 and count(rep, "edge_change") == 0


def test_series_upstream_flip(tmp_path):
    gates = gates_json(series=True)
    # g0 on -> off, g1 on: g1's group loses its edge too
    pt, _ = expand(tmp_path, {0: 1, 1: 1, d(0): 1, d(2): 1}, [G0], gates=gates)
    assert flips(pt) == [[0, 2]]
    # g0 on -> off, g1 off: g1's group had no edge and has none
    pt, _ = expand(tmp_path, {0: 1, d(0): 1, d(2): 1}, [2, G0], gates=gates)
    assert flips(pt) == [[0]]
    # g0 off -> on, g1 on: g1's group gains the edge
    pt, _ = expand(tmp_path, {1: 1, d(2): 1}, [G0], gates=gates)
    assert flips(pt) == [[2]]
    # g0 off -> on, g1 off: g1's group still has no edge
    pt, _ = expand(tmp_path, {d(2): 1}, [G0], gates=gates)
    assert pt.count == 0


def test_series_downstream_flip_and_both(tmp_path):
    gates = gates_json(series=True)
    # g0 off, g1 flips: the chain stays closed, nothing changes
    pt, rep = expand(tmp_path, {d(2): 1}, [G1], gates=gates)
    assert pt.count == 0
    assert count(rep, "no_edge_change") == 1 and count(rep, "edge_change") == 0
    # g0 on, g1 off -> on: only g1's group changes
    pt, _ = expand(tmp_path, {0: 1, d(2): 1}, [G1], gates=gates)
    assert flips(pt) == [[2]]
    # both flip, off and on: the product stays 0 for ff 2 (no edge), g0's group gains one
    pt, _ = expand(tmp_path, {1: 1, d(0): 1, d(2): 1}, [G0, G1], gates=gates)
    assert flips(pt) == [[0]]
    # both flip, on and off: g1's group loses its edge and g0's loses its own
    pt, _ = expand(tmp_path, {0: 1, d(0): 1, d(2): 1}, [G0, G1], gates=gates)
    assert flips(pt) == [[0]]


def test_blackbox_in_fanout_is_reported(tmp_path):
    gates = gates_json(blackbox=["ram.clk"])
    pt, rep = expand(tmp_path, {d(0): 1}, [G0], gates=gates)
    assert flips(pt) == [[0]]
    assert rep["counts"]["blackbox_clock"] == {"patterns": 1, "grid_points": 3}
    assert rep["blackbox_clock_by_gate"] == {"g0": {"patterns": 1, "grid_points": 3}}
    # no edge change, nothing to report
    _, rep = expand(tmp_path, {}, [3], gates=gates)
    assert count(rep, "blackbox_clock") == 0


def test_other_exceptions_are_reported(tmp_path):
    _, rep = expand(tmp_path, {d(0): 1}, [G0], gates=gates_json(other=["lat0"]))
    assert count(rep, "other_edge") == 1
    pt, rep = expand(tmp_path, {d(0): 1, d(1): 1}, [G0], gates=gates_json(ff_kw={0: {"observed": False}}),
                     ff_ids=[1, 2, 3, G0, G1])
    assert flips(pt) == [[1]] and count(rep, "unobserved_flip") == 1
    assert 0 not in pt.ff_ids.tolist()
    # an active low reset (net d(3), not used by another flip-flop here)
    _, rep = expand(tmp_path, {d(0): 1}, [G0], gates=gates_json(ff_kw={0: {"async_pins": [(d(3), 0)]}}))
    assert count(rep, "async_active") == 1


@pytest.mark.parametrize("nets", [{0: "x"}, {d(0): "x"}, {q(1): "x"}])
def test_unknown_value_is_reported(tmp_path, nets):
    _, rep = expand(tmp_path, nets, [G0])
    assert count(rep, "unknown_value") == 1


def test_unknown_digits(tmp_path):
    write_state(str(tmp_path), 7, {1: 1, 5: "x"})
    bits, unknown = read_cycle_bits(str(tmp_path), 7, N_NETS), read_cycle_unknown(str(tmp_path), 7, N_NETS)
    assert bits.tolist()[:2] == [0, 1]
    assert np.flatnonzero(unknown).tolist() == [4, 5, 6, 7]


def test_gates_leave_the_table(tmp_path):
    pt, _ = expand(tmp_path, {d(0): 1}, [G0])
    assert set(pt.ff_ids.tolist()) == {0, 1, 2, 3}
    assert pt.ffs(0, names=True) == ["ff0"]


def test_merged_runs_touch_and_sites_are_recounted(tmp_path):
    state = str(tmp_path / "states")
    write_state(state, 7, {d(0): 1})
    write_state(state, 9, {d(0): 1, 0: 1, d(1): 1})
    pt = table({(7, (0,)): [(0, 10, 1, 2)],                     # g0 off: no edge, gone
                (7, (G0,)): [(0, 10, 3, 4), (1, 20, 6, 6)],     # gives {0}
                (7, (3,)): [(0, 10, 5, 5)],
                (7, (0, G0)): [(0, 20, 0, 1)],                  # D ^ 1 = Q: empty
                (9, (0, 1)): [(1, 10, 0, 1)],                   # g0 on: captures {0, 1}
                (9, (G0,)): [(1, 10, 2, 3), (0, 10, 2, 3)]})    # missing edge: {0, 1}, touches
    out, rep = expand_clock_gates(pt, gates_json(), state, log=lambda s: None)
    got = [(out.cycle(p), out.ffs(p), out.multiplicity(p), int(out.pattern_n_sites[p]),
            [(s.site, s.width_ps, s.k_lo, s.k_hi) for s in out.sources(p)]) for p in range(out.count)]
    assert got == [(7, [0], 3, 2, [("s0", 10, 3, 4), ("s1", 20, 6, 6)]),
                   (7, [3], 1, 1, [("s0", 10, 5, 5)]),
                   (9, [0, 1], 6, 2, [("s0", 10, 2, 3), ("s1", 10, 0, 3)])]
    assert rep["grid_points_in"] == 2 + 3 + 1 + 2 + 2 + 4
    assert rep["grid_points_out"] == 10
    assert rep["counts"]["empty"]["grid_points"] == 2 + 2


def test_random_tables_against_brute_force(tmp_path):
    """The rewrite equals: F' of every grid point from the definition, then its table."""
    rng = np.random.default_rng(3)
    gates = gates_json(series=True)
    state = str(tmp_path / "states")
    cycles = [1, 2, 3, 4]
    ids_all = [0, 1, 2, 3, G0, G1]

    def f_prime(F, nets):
        v = lambda n: nets.get(n, 0)  # noqa: E731
        e0, e1 = v(0), v(1)
        edge0 = {0: e0, 1: e0, 2: e0 and e1, 3: 1}
        a = e0 ^ (G0 in F)
        edge1 = {0: a, 1: a, 2: a and (e1 ^ (G1 in F)), 3: 1}
        out = set()
        for f in range(4):
            in_f = f in F
            if edge0[f] and edge1[f]:
                hit = in_f
            elif not edge0[f] and not edge1[f]:
                hit = False
            elif edge1[f]:
                hit = (v(d(f)) ^ in_f) != v(q(f)) if f != 3 else in_f
            else:
                hit = v(d(f)) != v(q(f))
            if hit:
                out.add(f)
        return tuple(sorted(out))

    for trial in range(30):
        states = {}
        for c in cycles:
            nets = {n: int(rng.integers(0, 2)) for n in [0, 1] + [d(i) for i in range(3)] + [q(i) for i in range(3)]}
            states[c] = nets
            write_state(state, c, nets)
        # per trial (cycle, site, width): a random F for each grid index
        by_key, expected = {}, {}
        for c in cycles:
            for s, w in ((0, 10), (0, 20), (1, 10)):
                k = 0
                while k <= 10:
                    run = int(rng.integers(1, 4))
                    hi = min(10, k + run - 1)
                    F = tuple(sorted(rng.choice(ids_all, size=int(rng.integers(0, 4)), replace=False).tolist()))
                    if F:
                        by_key.setdefault((c, F), []).append((s, w, k, hi))
                    k = hi + 1
        # the input may hold one F in adjacent ranges of a trial: merge as compute_patterns does
        for key, runs in by_key.items():
            runs.sort()
            merged = []
            for r in runs:
                if merged and merged[-1][:2] == r[:2] and merged[-1][3] + 1 == r[2]:
                    merged[-1] = (r[0], r[1], merged[-1][2], r[3])
                else:
                    merged.append(r)
            by_key[key] = merged
        grid = {}
        for (c, F), runs in by_key.items():
            Fp = f_prime(set(F), states[c])
            for s, w, lo, hi in runs:
                for k in range(lo, hi + 1):
                    grid[(c, s, w, k)] = Fp
        for (c, s, w, k), Fp in sorted(grid.items()):
            if not Fp:
                continue
            runs = expected.setdefault((c, Fp), [])
            if runs and runs[-1][:2] == (s, w) and runs[-1][3] + 1 == k:
                runs[-1] = (s, w, runs[-1][2], k)
            else:
                runs.append((s, w, k, k))
        pt = table(by_key)
        out, _ = expand_clock_gates(pt, gates, state, log=lambda s: None)
        want = table(expected, ff_ids=[0, 1, 2, 3]) if expected else None
        got = {(out.cycle(p), tuple(out.ffs(p))): out for p in range(out.count)}
        assert set(got) == set(expected), trial
        if want is not None:
            for name in ("pattern_cycle", "pattern_ff_ptr", "pattern_ffs", "pattern_multiplicity",
                         "pattern_n_sites", "pattern_run_ptr", "run_site", "run_width_ps", "run_k_lo", "run_k_hi"):
                a, b = getattr(out, name), getattr(want, name)
                assert a.dtype == b.dtype and np.array_equal(a, b), (trial, name)


# ---- the build, on a small netlist ----
LIBERTY = """
library (t) {
  cell (INV) { pin (A) { direction : input; } pin (Z) { direction : output; function : "!A"; } }
  cell (BUF) { pin (A) { direction : input; } pin (Z) { direction : output; function : "A"; } }
  cell (AND) { pin (A) { direction : input; } pin (B) { direction : input; }
    pin (Z) { direction : output; function : "A * B"; } }
  cell (TIE0) { pin (Z) { direction : output; function : "0"; } }
  cell (DFF) { ff (IQ, IQN) { next_state : "D"; clocked_on : "CK"; }
    pin (D) { direction : input; } pin (CK) { direction : input; clock : true; }
    pin (Q) { direction : output; function : "IQ"; } }
  cell (DFFN) { ff (IQ, IQN) { next_state : "D"; clocked_on : "CK"; }
    pin (D) { direction : input; } pin (CK) { direction : input; clock : true; }
    pin (QN) { direction : output; function : "IQN"; } }
  cell (DFFNE) { ff (IQ, IQN) { next_state : "D"; clocked_on : "!CK"; }
    pin (D) { direction : input; } pin (CK) { direction : input; clock : true; }
    pin (Q) { direction : output; function : "IQ"; } }
  cell (DFFR) { ff (IQ, IQN) { next_state : "D"; clocked_on : "CK"; clear : "!RN"; }
    pin (D) { direction : input; } pin (CK) { direction : input; clock : true; } pin (RN) { direction : input; }
    pin (Q) { direction : output; function : "IQ"; } }
  cell (DFFC) { ff (IQ, IQN) { next_state : "CN * D"; clocked_on : "CK"; }
    pin (D) { direction : input; } pin (CK) { direction : input; clock : true; } pin (CN) { direction : input; }
    pin (Q) { direction : output; function : "IQ"; } }
  cell (LAT) { latch (IQ, IQN) { enable : "CK"; data_in : "D"; }
    pin (D) { direction : input; } pin (CK) { direction : input; }
    pin (Q) { direction : output; function : "IQ"; } }
  cell (ICG) { clock_gating_integrated_cell : latch_posedge_precontrol;
    pin (CK) { direction : input; clock_gate_clock_pin : true; }
    pin (E) { direction : input; clock_gate_enable_pin : true; }
    pin (TE) { direction : input; clock_gate_test_pin : true; }
    pin (GCK) { direction : output; clock_gate_out_pin : true; state_function : "CK * IQ"; } }
  cell (ICGN) { clock_gating_integrated_cell : latch_negedge_precontrol;
    pin (CK) { direction : input; clock_gate_clock_pin : true; }
    pin (E) { direction : input; clock_gate_enable_pin : true; }
    pin (TE) { direction : input; clock_gate_test_pin : true; }
    pin (GCK) { direction : output; clock_gate_out_pin : true; state_function : "CK + !IQ"; } }
}
"""

NETLIST = """
module top (clk, a, b, out);
  input clk, a, b;
  output out;
  wire en0, en1, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, g0, g0b, g0i, g1, gn, tie, l0;
  TIE0 t0 ( .Z(tie) );
  INV u0 ( .A(q0), .Z(en0) );
  INV u1 ( .A(q1), .Z(en1) );
  DFF f0 ( .D(a), .CK(clk), .Q(q0) );
  DFF f1 ( .D(b), .CK(clk), .Q(q1) );
  ICG g_top ( .CK(clk), .E(en0), .TE(tie), .GCK(g0) );
  BUF ub ( .A(g0), .Z(g0b) );
  INV ui ( .A(g0), .Z(g0i) );
  DFF f2 ( .D(en1), .CK(g0b), .Q(q2) );
  DFFN f3 ( .D(en1), .CK(g0b), .QN(q3) );
  DFFR f4 ( .D(en1), .CK(g0b), .RN(q1), .Q(q4) );
  DFFNE f5 ( .D(en1), .CK(g0b), .Q(q5) );
  DFFNE f6 ( .D(en1), .CK(g0i), .Q(q6) );
  DFFC f9 ( .D(en1), .CK(g0b), .CN(q1), .Q(q9) );
  LAT l1 ( .D(en1), .CK(g0b), .Q(l0) );
  ram mem ( .clk(g0b), .d(q2) );
  ICG g_low ( .CK(g0b), .E(en1), .TE(tie), .GCK(g1) );
  DFF f7 ( .D(en0), .CK(g1), .Q(q7) );
  ICGN g_neg ( .CK(clk), .E(en0), .TE(tie), .GCK(gn) );
  DFF f8 ( .D(en0), .CK(gn), .Q(q8) );
  INV u2 ( .A(q8), .Z(out) );
endmodule
"""


def sdf_text():
    tc = """(TIMINGCHECK
      (SETUP (posedge {d}) (posedge CK) (0.02)) (SETUP (negedge {d}) (posedge CK) (0.02))
      (HOLD (posedge {d}) (posedge CK) (0.01)) (HOLD (negedge {d}) (posedge CK) (0.01)))"""
    inv = '(DELAY (ABSOLUTE (IOPATH A Z (0.03) (0.03))))'
    ff = '(DELAY (ABSOLUTE (IOPATH CK Q (0.05) (0.05))))\n    ' + tc.format(d="D")
    icg = '(DELAY (ABSOLUTE (IOPATH CK GCK (0.03) (0.03))))\n    ' + tc.format(d="E")
    cells = {"u0": ("INV", inv), "u1": ("INV", inv), "u2": ("INV", inv), "ui": ("INV", inv), "ub": ("BUF", inv),
             "f0": ("DFF", ff), "f1": ("DFF", ff), "f2": ("DFF", ff),
             "f3": ("DFFN", '(DELAY (ABSOLUTE (IOPATH CK QN (0.05) (0.05))))\n    ' + tc.format(d="D")),
             "f4": ("DFFR", ff), "f5": ("DFFNE", ff), "f6": ("DFFNE", ff), "f7": ("DFF", ff), "f8": ("DFF", ff),
             "f9": ("DFFC", ff),
             "l1": ("LAT", '(DELAY (ABSOLUTE (IOPATH D Q (0.03) (0.03))))'),
             "g_top": ("ICG", icg), "g_low": ("ICG", icg), "g_neg": ("ICGN", icg)}
    return ('(DELAYFILE (SDFVERSION "3.0") (DESIGN "top") (TIMESCALE 1ns) (DIVIDER /)\n'
            + "".join(f'  (CELL (CELLTYPE "{c}") (INSTANCE {i})\n    {b})\n' for i, (c, b) in cells.items()) + ")\n")


def build(tmp_path, gate_cells="roles", **lib_kw):
    verilog, models = liberty_to_verilog_text(tmp_path)
    roles = {m.name: m.clock_gate for m in models.values() if m.clock_gate} if gate_cells == "roles" else gate_cells
    (tmp_path / "top.v").write_text(NETLIST)
    (tmp_path / "top.sdf").write_text(sdf_text())
    spec = BuildSpec(netlist=str(tmp_path / "top.v"), sdf=str(tmp_path / "top.sdf"), top="top",
                     out_dir=str(tmp_path / "out"), clock_gate_cells=roles, allow_missing_delays=True,
                     library=LibrarySpec(behavioral_verilog=verilog, blackbox_cells=("ram",),
                                         untimed_cells=("TIE0",), clock_pin_names=("CK",), **lib_kw))
    return spec, run_build(spec)


def liberty_to_verilog_text(tmp_path):
    (tmp_path / "cells.lib").write_text(LIBERTY)
    text, models = liberty_to_verilog([str(tmp_path / "cells.lib")])
    (tmp_path / "cells.v").write_text(text)
    return str(tmp_path / "cells.v"), models


def load(spec, name):
    with open(os.path.join(spec.out_dir, name)) as f:
        return json.load(f)


def test_build_finds_gates_and_their_fanout(tmp_path):
    spec, man = build(tmp_path)
    cg = load(spec, "clock_gates.json")
    ff = load(spec, "ff_index.json")
    name = {int(k): v["ff_inst"] for k, v in ff["ffid_to_info"].items()}
    rows = {r["inst"]: r for r in cg["gates"]}
    assert [r["inst"] for r in cg["gates"]] == ["g_top", "g_low"]          # parents first
    top, low = rows["g_top"], rows["g_low"]
    assert sorted(name[f] for f in top["ffs"]) == ["f2", "f3", "f4", "f6", "f9"]  # through buffer and inverter
    assert sorted(top["other_edge"]) == ["f5", "l1"]                        # the falling edge, a latch
    assert top["blackbox_pins"] == ["mem.clk"]
    assert top["gates"] == ["g_low"] and low["parent"] == "g_top"
    assert [name[f] for f in low["ffs"]] == ["f7"]
    net = load(spec, "net_index.json")["net_to_idx"]
    assert top["enable_net_idx"] == net["en0"] and low["enable_net_idx"] == net["en1"]
    assert top["test_net_idx"] == net["tie"]
    e3 = cg["ffs"][str(next(f for f in top["ffs"] if name[f] == "f3"))]
    assert e3["q_inverted"] is True and e3["q_net_idx"] == net["q3"] and e3["d_net_idx"] == net["en1"]
    e4 = cg["ffs"][str(next(f for f in top["ffs"] if name[f] == "f4"))]
    assert e4["q_inverted"] is False and e4["async"] == [[net["q1"], 0]] and e4["observed"] is True
    e9 = cg["ffs"][str(next(f for f in top["ffs"] if name[f] == "f9"))]
    assert e9["next_nets"] == [net["q1"]] and cg["next_tables"] == {"DFFC": [0, 0, 0, 1]}
    assert "next_nets" not in e3 and "next_nets" not in e4
    assert [u["inst"] for u in cg["unmodelled_gates"]] == ["g_neg"]
    assert set(cg["other_clock_sources"]) == {"g_neg"}
    assert man["clock_gates"]["gates_in_series"] == 1
    assert "clock_gates.json" in man["artifacts"]
    # modelled gates are observed through the enable, the other type is not
    assert {"g_top", "g_low"} <= set(name.values()) and "g_neg" not in name.values()
    assert ff["_meta"]["observe_filter"]["n_clock_gates_kept"] == 2


def test_gates_are_observed_despite_the_exclusions(tmp_path):
    spec, _ = build(tmp_path, non_observed_cells=("ICG.*",))
    name = {v["ff_inst"] for v in load(spec, "ff_index.json")["ffid_to_info"].values()}
    assert {"g_top", "g_low"} <= name


def test_excluded_flip_flop_behind_a_gate_is_marked(tmp_path):
    spec, _ = build(tmp_path, non_observed_cells=("DFFN",))
    cg = load(spec, "clock_gates.json")
    assert [e["inst"] for e in cg["ffs"].values() if not e["observed"]] == ["f3"]


def test_enable_must_be_the_data_pin(tmp_path):
    with pytest.raises(BuildError, match=r"clock gate ICG: Liberty names clock pin CK and enable pin E"):
        build(tmp_path, d_pin_names=("TE",))


def test_ignore_is_the_build_without_gates(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    spec_a, _ = build(a, gate_cells=None)
    spec_b, _ = build(b)
    assert not os.path.exists(os.path.join(spec_a.out_dir, "clock_gates.json"))
    ffa, ffb = load(spec_a, "ff_index.json"), load(spec_b, "ff_index.json")
    assert {v["ff_inst"] for v in ffa["ffid_to_info"].values()} == \
        {v["ff_inst"] for v in ffb["ffid_to_info"].values()} - {"g_top", "g_low"}
    for name in sorted(os.listdir(spec_a.out_dir)):
        if name in ("ff_index.json", "build_manifest.json"):
            continue
        with open(os.path.join(spec_a.out_dir, name), "rb") as f, open(os.path.join(spec_b.out_dir, name), "rb") as g:
            assert f.read() == g.read(), name


# ---- build, campaign, patterns, analyze ----
def campaign(tmp_path, spec, nets, name):
    index = load(spec, "net_index.json")["net_to_idx"]
    n = len(index)
    state = tmp_path / "states"
    state.mkdir(exist_ok=True)
    val = 0
    for net, v in nets.items():
        val |= v << index[net]
    (state / "cycle_5.hex").write_text(f"{val:0{(n + 3) // 4}x}\n")
    out = str(tmp_path / name)
    run_campaign(CampaignSpec(stage_dir=spec.out_dir, state_dir=str(state), cycles=[5], out_dir=out,
                              clock_period_ps=1000, pulse_widths_ps=[100], clock_pins=["CK"],
                              allow_missing_timing_checks=True), log=lambda s: None)
    return out, str(state), n


def test_campaign_to_expanded_patterns(tmp_path):
    spec, _ = build(tmp_path)
    # q0 = 1: en0 = 0, so g_top is off; q1 = 0: en1 = 1, D of f2..f6 is 1
    nets = {"q0": 1, "en0": 0, "q1": 0, "en1": 1, "g0": 0}
    out, state, _ = campaign(tmp_path, spec, nets, "inject")
    pt = compute_patterns(out, 100)
    cg = load(spec, "clock_gates.json")
    ids = {r["inst"]: r["ffid"] for r in cg["gates"]}
    assert any(ids["g_top"] in pt.ffs(p) for p in range(pt.count))
    expanded, rep = expand_clock_gates(pt, cg, state, log=lambda s: None)
    assert rep["counts"]["gate_in_F"]["patterns"] > 0
    assert not set(expanded.ff_ids.tolist()) & set(ids.values())
    assert not set(expanded.pattern_ffs.tolist()) & set(ids.values())
    name = {int(k): v["ff_inst"] for k, v in load(spec, "ff_index.json")["ffid_to_info"].items()}
    flipped = {name[f] for f in expanded.pattern_ffs.tolist()}
    # the extra edge flips f2 (D 1, Q 0) and the others of the group
    assert "f2" in flipped
    # the table still follows the format: run lengths add up, runs are ordered
    for p in range(expanded.count):
        src = expanded.sources(p)
        assert sum(s.k_hi - s.k_lo + 1 for s in src) == expanded.multiplicity(p)
    path = str(tmp_path / "patterns.npz")
    expanded.save(path, expansion=json.dumps(rep))
    assert_same(Patterns.load(path), expanded)
    with np.load(path) as z:
        assert json.loads(str(z["expansion"]))["format"] == "setfi-clock-gate-expansion/1"


def test_analyze_without_gates_equals_the_ignore_campaign(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    spec_a, _ = build(a, gate_cells=None)
    spec_b, _ = build(b)
    nets = {"q0": 1, "en0": 0, "q1": 0, "en1": 1}
    rec_a = campaign(a, spec_a, nets, "inject")[0]
    rec_b = campaign(b, spec_b, nets, "inject")[0]
    gate_ffs = clock_gate_ffs(load(spec_b, "clock_gates.json"))
    assert Records.open(rec_b, exclude_ffs=gate_ffs).ffs == Records.open(rec_a).ffs
    uniform = {"type": "uniform"}
    sa = analyze(rec_a, str(a / "an"), uniform, log=lambda s: None)
    sb = analyze(rec_b, str(b / "an"), uniform, log=lambda s: None, exclude_ffs=gate_ffs)
    assert sa == sb
    for name in ("per_ff.csv", "per_site.csv", "per_width.csv"):
        assert (a / "an" / name).read_bytes() == (b / "an" / name).read_bytes(), name


# ---- the stages, with the recording replaced by a hand-written state ----
def _stage_project(tmp_path, mode):
    (tmp_path / "cells.lib").write_text(LIBERTY)
    (tmp_path / "top.v").write_text(NETLIST)
    (tmp_path / "top.sdf").write_text(sdf_text())
    (tmp_path / "tb.sv").write_text("")
    path = tmp_path / "setfi.toml"
    path.write_text(f"""
[design]
netlist = "top.v"
sdf = "top.sdf"
top = "top"
clock_period_ns = 1.0
[output]
dir = "out"
[library]
cell_functions = ["cells.lib"]
clock_pins = ["CK"]
blackbox_cells = ["ram"]
untimed_cells = ["TIE0"]
[workload]
testbench = ["tb.sv"]
cell_sim_models = ["cells.lib"]
[fault_model]
pulse_widths_ps = [100, 200]
[observe]
allow_missing_timing_checks = true
clock_gates = "{mode}"
[patterns]
steps = 100
[build]
allow_missing_delays = true
""")
    return C.load(str(path), check_files=False)


def _record_by_hand(cfg):
    p = cli.paths(cfg)
    with open(os.path.join(p["build"], "net_index.json")) as f:
        index = json.load(f)["net_to_idx"]
    os.makedirs(os.path.join(p["record"], "states"))
    val = sum(v << index[n] for n, v in {"q0": 1, "q1": 0, "en1": 1}.items())
    with open(os.path.join(p["record"], "states", "cycle_5.hex"), "w") as f:
        f.write(f"{val:0{(len(index) + 3) // 4}x}\n")
    with open(os.path.join(p["record"], "cycles.json"), "w") as f:
        json.dump({"format": "setfi-cycles/1", "cycles": [5]}, f)
    cli._finish(cfg, "record", cli.fingerprint(cfg, "record"), {"cycles": 1})


def test_stages_expand(tmp_path, capsys):
    cfg = _stage_project(tmp_path, "expand")
    p = cli.paths(cfg)
    cli.do_build(cfg)
    _record_by_hand(cfg)
    cli.do_inject(cfg)
    cli.do_patterns(cfg)
    assert sorted(os.listdir(p["patterns"])) == ["clock_gates.json", "patterns.csv", "patterns.npz"]
    with open(os.path.join(p["patterns"], "clock_gates.json")) as f:
        rep = json.load(f)
    assert rep["counts"]["gate_in_F"]["patterns"] > 0 and rep["counts"]["blackbox_clock"]["patterns"] > 0
    pt = Patterns.load(os.path.join(p["patterns"], "patterns.npz"))
    with np.load(os.path.join(p["patterns"], "patterns.npz")) as z:
        assert json.loads(str(z["expansion"])) == rep
    cg = load_gates(p)
    assert not set(pt.ff_ids.tolist()) & set(clock_gate_ffs(cg))
    stamps = cli.read_stamps(cfg)
    assert stamps["patterns"]["summary"]["clock_gates"] == rep["counts"]
    assert set(stamps["patterns"]["upstream"]) == {"inject.pulses", "build.graph", "record.states"}
    capsys.readouterr()
    cli.do_patterns(cfg, force=False)
    assert "[patterns] up to date" in capsys.readouterr().out

    cli.do_logic(cfg)
    assert os.path.exists(os.path.join(p["logic"], "clock_gates.json"))
    lg = Patterns.load(os.path.join(p["logic"], "patterns.npz"))
    assert not set(lg.ff_ids.tolist()) & set(clock_gate_ffs(cg)) and lg.steps == 0

    out = cli.do_analyze(cfg)[0]
    with open(os.path.join(out, "summary.json")) as f:
        summary = json.load(f)
    assert summary["provenance"]["clock_gates_excluded"] == 2
    names = {r["inst"] for r in cg["gates"]}
    with open(os.path.join(out, "per_ff.csv")) as f:
        assert not names & {line.split(",")[1] for line in f.read().splitlines()[1:]}


def load_gates(p):
    with open(os.path.join(p["build"], "clock_gates.json")) as f:
        return json.load(f)


def test_stages_ignore_stay_as_they_were(tmp_path):
    cfg = _stage_project(tmp_path, "ignore")
    p = cli.paths(cfg)
    cli.do_build(cfg)
    _record_by_hand(cfg)
    cli.do_inject(cfg)
    cli.do_patterns(cfg)
    cli.do_logic(cfg)
    assert sorted(os.listdir(p["patterns"])) == ["patterns.csv", "patterns.npz"]
    assert sorted(os.listdir(p["logic"])) == ["patterns.csv", "patterns.npz"]
    assert not os.path.exists(os.path.join(p["build"], "clock_gates.json"))
    with np.load(os.path.join(p["patterns"], "patterns.npz")) as z:
        assert "expansion" not in z.files


def test_switching_the_mode_reruns_build(tmp_path, capsys):
    cfg = _stage_project(tmp_path, "ignore")
    cli.do_build(cfg)
    on = _stage_project(tmp_path, "expand")
    ok, why = cli.stage_status(on, "build", cli.read_stamps(on))
    assert not ok and "observe.clock_gates" in why
    cli.do_build(on)
    assert os.path.exists(os.path.join(cli.paths(on)["build"], "clock_gates.json"))
    off = _stage_project(tmp_path, "ignore")
    ok, why = cli.stage_status(off, "build", cli.read_stamps(off))
    assert not ok
    cli.do_build(off)
    assert not os.path.exists(os.path.join(cli.paths(off)["build"], "clock_gates.json"))
    capsys.readouterr()


# ---- configuration ----
def test_config_key_and_its_stages():
    assert C.schema()["observe"]["clock_gates"].default == "ignore"
    assert set(C.used_by("observe", "clock_gates")) == {"build", "patterns", "logic", "analyze"}


def _cfg(tmp_path, mode):
    for f in ("net.v", "net.sdf", "cells.v", "cells.lib", "tb.sv"):
        (tmp_path / f).write_text("")
    path = tmp_path / "setfi.toml"
    path.write_text(f"""
[design]
netlist = "net.v"
sdf = "net.sdf"
top = "t"
clock_period_ns = 1.0
[output]
dir = "out"
[library]
cell_functions = ["cells.lib"]
[workload]
testbench = ["tb.sv"]
[observe]
clock_gates = "{mode}"
""")
    return C.load(str(path), check_files=False)


def test_fingerprints_are_those_without_the_key_when_ignored(tmp_path):
    cfg = _cfg(tmp_path, "ignore")
    for stage in ("build", "patterns", "logic"):
        assert "observe.clock_gates" not in cli.fingerprint(cfg, stage, {})["config"]
    assert list(cli.fingerprint(cfg, "patterns", {})["upstream"]) == ["inject.pulses"]
    on = _cfg(tmp_path, "expand")
    assert cli.fingerprint(on, "patterns", {})["config"]["observe.clock_gates"] == "expand"
    assert cli.fingerprint(on, "build", {})["fingerprint"] != cli.fingerprint(cfg, "build", {})["fingerprint"]
    assert set(cli.fingerprint(on, "patterns", {})["upstream"]) == {"inject.pulses", "build.graph", "record.states"}


def test_expand_needs_liberty(tmp_path):
    from setfi.pipeline import build_spec_from_config
    cfg = _cfg(tmp_path, "expand")
    cfg["library"]["cell_functions"] = [str(tmp_path / "cells.v")]
    with pytest.raises(C.ConfigError, match="finds clock gates in Liberty"):
        build_spec_from_config(cfg, cli.paths(cfg))
