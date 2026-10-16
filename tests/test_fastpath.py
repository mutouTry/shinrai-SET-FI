"""The C++ fast path (skeleton reader, cone compiler, run_block) against the
Python reference: same compiled arrays, same campaign output files.

Random timing graphs exercise multi-input gates, conditional delays, shared
pins, wires (interconnect = transport), nets missing from net_to_idx (state
settling) and every propagation option.

With SETFI_REFERENCE_ENGINE=1 in the environment, run_campaign uses the Python
reference engine (setfi.campaign.run), so that the other campaign tests can be
run on it as well.
"""
import json
import os
import random
import signal
import time

import numpy as np
import pytest

from setfi import _engine
from setfi.campaign import fastpath
from setfi.campaign.run import CampaignSpec, engine_options, run_campaign
from setfi.campaign.substrate import (SiteCones, SubstrateError, load_net_index, load_site_cones,
                                      load_skeleton, load_substrate)

CLOCK = ["CP", "CK", "CLK"]


def _dump(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f)


def write_random_substrate(d, seed, n_src=5, n_gates=40, wires=True, drop_from_net_to_idx=True, pin="A"):
    """A random combinational DAG between flip-flops, written as a build directory."""
    rng = random.Random(seed)
    os.makedirs(d, exist_ok=True)
    nets = [f"s{i}" for i in range(n_src)]
    gates = []                                  # (inst, cell, in_nets, out_net)
    for g in range(n_gates):
        k = rng.choice([1, 2, 2, 3, 3, 4])
        ins = [rng.choice(nets[max(0, len(nets) - 12):]) for _ in range(k)]
        if k >= 2 and rng.random() < 0.15:
            ins[1] = ins[0]                     # two pins on one net
        out = f"n{g}"
        gates.append((f"u{g}", f"C{g}", ins, out))
        nets.append(out)
    n_ff = 8
    d_nets = rng.sample(nets[n_src:], n_ff)
    names = sorted(nets) + ["unused_net"]
    idx = {n: i for i, n in enumerate(names)}
    net_to_idx = dict(idx)
    if drop_from_net_to_idx:
        # a gate output known to idx_to_net but not to net_to_idx: its value is settled
        net_to_idx.pop(rng.choice([g[3] for g in gates[n_gates // 2:]]))
    _dump(d, "net_index.json", {"net_to_idx": net_to_idx, "idx_to_net": {str(i): n for n, i in idx.items()}})

    edges, pin_to_net, cell_arcs, dkeys = [], {}, {}, []
    rise, fall, patt_ptr, patt_mask, patt_val, patt_dt, patt_rise = [], [], [0], [], [], [], []
    for inst, cell, ins, out in gates:
        k = len(ins)
        pins = [f"{pin}{j}" for j in range(k)]
        truth = [rng.randint(0, 1) for _ in range(2 ** k)]
        if len(set(truth)) == 1:
            truth[rng.randrange(len(truth))] ^= 1
        for j, p in enumerate(pins):
            pin_to_net[f"{inst}.{p}"] = ins[j]
            side = [q for q in pins if q != p]
            cell_arcs[f"{cell}:{p}->Z"] = {"vars_order": pins, "truth": truth, "side_pins": side}
            dk = f"{inst}:{p}->Z"
            dkeys.append(dk)
            rise.append(rng.randint(5, 60))
            fall.append(rng.randint(5, 60))
            if side and rng.random() < 0.5:     # conditional delays on the side pins
                for _ in range(rng.randint(1, 3)):
                    m = rng.randint(1, 2 ** len(side) - 1)
                    patt_mask.append(m)
                    patt_val.append(rng.randint(0, 2 ** len(side) - 1) & m)
                    patt_dt.append(rng.choice([-1, rng.randint(3, 70)]))
                    patt_rise.append(rng.randint(0, 1))
            patt_ptr.append(len(patt_mask))
            edges.append(dict(eid=len(edges), src_net=ins[j], dst_net=out, inst=inst, celltype=cell,
                              inpin=p, outpin="Z", mask_key=f"{cell}:{p}->Z", delay_key=dk))
        pin_to_net[f"{inst}.Z"] = out
    _dump(d, "edges.json", edges)
    ffs = {}
    for i, dn in enumerate(d_nets):
        pin_to_net[f"ff{i}.D"] = dn
        ffs[str(i)] = {"ff_inst": f"ff{i}", "d_net": dn, "d_pin": "D", "celltype": "DFF"}
    _dump(d, "pin_to_net.json", pin_to_net)
    _dump(d, "cell_arcs.json", cell_arcs)
    _dump(d, "ff_index.json", {"ffid_to_info": ffs})
    chk = lambda s, h: {"posedge:CP": {"SETUP": [{"value": s}], "HOLD": [{"value": h}]}}  # noqa: E731
    _dump(d, "timing_checks.json", {"index": {f"ff{i}": {"meta": {"ff_inst": f"ff{i}"}, "checks": {
        "posedge:D": chk(rng.choice([0.02, 0.03]), 0.005),
        "negedge:D": chk(rng.choice([0.02, 0.03]), 0.004)}} for i in range(n_ff)}})
    _dump(d, "arc_index.json", {"delay_key_to_arc_idx": {k: i for i, k in enumerate(dkeys)}})
    np.savez(os.path.join(d, "delay_table.npz"), default_rise_ps=np.array(rise), default_fall_ps=np.array(fall),
             patt_ptr=np.array(patt_ptr, np.int32), patt_mask=np.array(patt_mask, np.int64),
             patt_valbits=np.array(patt_val, np.int64), patt_dt_ps=np.array(patt_dt, np.int64),
             patt_is_rise=np.array(patt_rise, np.uint8))
    if wires:
        ents = []
        for inst, _cell, ins, _out in gates:
            for j in range(len(ins)):
                if rng.random() < 0.6:
                    r = rng.choice([0, 0, rng.randint(1, 20)])
                    ents.append({"sink_inst": inst, "sink_pin": f"{pin}{j}", "rise_ps": r,
                                 "fall_ps": r if rng.random() < 0.7 else rng.randint(0, 20)})
        _dump(d, "interconnect.json", {"entries": ents})

    # site cones: fan-out cone of every source and of some gate outputs
    fan = {}
    for e in edges:
        fan.setdefault(e["src_net"], []).append(e)
    order = {n: i for i, n in enumerate(nets)}
    sites = nets[:n_src] + rng.sample(nets[n_src:], n_gates // 3)
    with open(os.path.join(d, "site_cones.jsonl"), "w") as f:
        for s in sites:
            seen, todo = {s}, [s]
            while todo:
                u = todo.pop()
                for e in fan.get(u, []):
                    if e["dst_net"] not in seen:
                        seen.add(e["dst_net"])
                        todo.append(e["dst_net"])
            topo = sorted(seen, key=order.get)
            adj = [[idx[u], [e["eid"] for e in fan.get(u, []) if e["dst_net"] in seen]] for u in topo]
            adj = [a for a in adj if a[1]]
            f.write(json.dumps({"src_net_idx": idx[s], "cone_topo_net_idxs": [idx[n] for n in topo],
                                "cone_adj_idx": adj, "has_reconv": rng.choice([True, False]),
                                "reachable_dnet_idxs": [idx[n] for n in topo if n in d_nets]}) + "\n")
    return names


def write_states(d, n_nets, cycles, seed):
    rng = random.Random(seed)
    os.makedirs(d, exist_ok=True)
    for c in cycles:
        v = rng.getrandbits(n_nets)
        with open(os.path.join(d, f"cycle_{c}.hex"), "w") as f:
            f.write(f"{v:x}\n")


def output_files(out_dir):
    """Every output file's content; index.json without elapsed_s."""
    res = {}
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if fn == "index.json":
            with open(p) as f:
                idx = json.load(f)
            idx.pop("elapsed_s")
            res[fn] = idx
        elif fn.endswith(".npz"):
            with np.load(p) as z:
                res[fn] = {k: z[k].copy() for k in z.files}
        else:
            with open(p, "rb") as f:
                res[fn] = f.read()
    return res


def assert_same_outputs(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        if isinstance(a[k], dict) and k.endswith(".npz"):
            assert sorted(a[k]) == sorted(b[k])
            for c in a[k]:
                assert a[k][c].dtype == b[k][c].dtype and np.array_equal(a[k][c], b[k][c]), (k, c)
        else:
            assert a[k] == b[k], k


def compare_cones(sub):
    sites = sorted(sub.skeleton)
    cones = fastpath.compile_sites(sub, sites, n_threads=2, keep_names=True)
    names = fastpath.compiled_names(sub)
    assert names is not None, "the compiler tables were not built"
    for s, c in zip(sites, cones):
        ref = fastpath.reference_arrays(sub, s)
        got = c.arrays()
        for k, v in got.items():
            if k in ("net_nid", "dp_nid"):
                continue
            if isinstance(v, np.ndarray):
                r = np.asarray(ref[k])
                assert v.dtype == r.dtype and v.shape == r.shape and np.array_equal(v, r), (s, k)
            else:
                assert v == ref[k], (s, k)
        i2 = [names[x] if x >= 0 else None for x in got["net_nid"].tolist()]
        assert [n if n is not None and "\x01" not in n else None for n in ref["i2net"]] == i2, s
    return len(sites)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("interconnect", ["ignore", "transport"])
def test_compiled_cones_match_the_python_reference(tmp_path, seed, interconnect):
    write_random_substrate(str(tmp_path), seed)
    sub = load_substrate(str(tmp_path), clock_pins=CLOCK, interconnect=interconnect, n_threads=2)
    assert isinstance(sub.skeleton, SiteCones)
    assert compare_cones(sub) > 0


def test_site_cones_reader_matches_load_skeleton(tmp_path):
    write_random_substrate(str(tmp_path), 7)
    _, idx_to_net = load_net_index(str(tmp_path / "net_index.json"))
    path = str(tmp_path / "site_cones.jsonl")
    fast = load_site_cones(path, idx_to_net)
    ref = load_skeleton(path, idx_to_net)
    assert list(fast) == list(ref) and len(fast) == len(ref)
    assert all(fast[s] == ref[s] for s in ref)
    assert isinstance(load_substrate(str(tmp_path), clock_pins=CLOCK).skeleton, dict)   # the default reader


@pytest.mark.parametrize("line", [
    '{"src_net": "s0", "cone_topo_nets": ["s0"], "cone_adj": [], "reachable_dnets": []}',   # name form
    '{"src_net_idx": 0, "cone_topo_net_idxs": [0.0], "cone_adj_idx": [], "reachable_dnet_idxs": []}',
    '{"src_net_idx": 0, "cone_topo_net_idxs": [99999], "cone_adj_idx": [], "reachable_dnet_idxs": []}',
])
def test_unusual_site_cones_use_the_python_reader(tmp_path, line):
    write_random_substrate(str(tmp_path), 3)
    with open(tmp_path / "site_cones.jsonl", "a") as f:
        f.write(line + "\n")
    _, idx_to_net = load_net_index(str(tmp_path / "net_index.json"))
    assert load_site_cones(str(tmp_path / "site_cones.jsonl"), idx_to_net) is None
    if "99999" in line:          # the reference reader reports it
        with pytest.raises(SubstrateError):
            load_substrate(str(tmp_path), clock_pins=CLOCK, n_threads=2)
        return
    sub = load_substrate(str(tmp_path), clock_pins=CLOCK, n_threads=2)
    assert isinstance(sub.skeleton, dict)
    compare_cones(sub)           # the compiler also runs on a dict skeleton


OPTION_SETS = [
    dict(),
    dict(delay_choice="min"),
    dict(coalesce_window_ps=1),
    dict(coalesce_window_ps=3),
    dict(delay_choice="min", coalesce_window_ps=2),
    dict(capture_edge_ps=90, sim_horizon_ps=140, timeprecision_ps=2),
]


@pytest.mark.parametrize("seed,opts,interconnect,jobs", [
    (0, OPTION_SETS[0], "ignore", 1), (1, OPTION_SETS[1], "transport", 2), (2, OPTION_SETS[2], "transport", 3),
    (3, OPTION_SETS[3], "ignore", 2), (4, OPTION_SETS[4], "transport", 1), (5, OPTION_SETS[0], "transport", 4),
    (6, OPTION_SETS[5], "transport", 3),
])
def test_fast_engine_writes_what_the_reference_engine_writes(tmp_path, monkeypatch, seed, opts, interconnect, jobs):
    names = write_random_substrate(str(tmp_path / "build"), seed)
    cycles = [3, 5, 8]
    write_states(str(tmp_path / "state"), len(names), cycles, seed)
    outs = {}
    for engine in ("reference", "fast"):
        if engine == "reference":
            monkeypatch.setenv("SETFI_REFERENCE_ENGINE", "1")
        else:
            monkeypatch.delenv("SETFI_REFERENCE_ENGINE", raising=False)
        spec = CampaignSpec(stage_dir=str(tmp_path / "build"), state_dir=str(tmp_path / "state"), cycles=cycles,
                            out_dir=str(tmp_path / engine), clock_period_ps=100, pulse_widths_ps=[5, 15, 30, 60],
                            interconnect=interconnect, jobs=jobs, cycles_per_shard=2,
                            allow_missing_timing_checks=True, **opts)
        idx = run_campaign(spec, log=lambda *a: None)
        outs[engine] = output_files(str(tmp_path / engine))
        assert idx["counts"]["trials"] > 0
    assert_same_outputs(outs["reference"], outs["fast"])
    assert any(v["n_records"] for v in outs["fast"]["index.json"]["shards"])


def test_run_block_does_not_depend_on_the_thread_count(tmp_path):
    names = write_random_substrate(str(tmp_path / "build"), 11)
    write_states(str(tmp_path / "state"), len(names), [1, 2], 11)
    sub = load_substrate(str(tmp_path / "build"), clock_pins=CLOCK, interconnect="transport", n_threads=2)
    sites = sorted(sub.skeleton)
    cones = fastpath.compile_sites(sub, sites)
    from setfi.campaign.state import read_cycle_bits
    bits = np.stack([read_cycle_bits(str(tmp_path / "state"), c, len(names)) for c in (1, 2)])
    spec = CampaignSpec(stage_dir="", state_dir="", cycles=[1, 2], out_dir="", clock_period_ps=100,
                        pulse_widths_ps=[10, 40]).resolved()
    runs = [_engine.run_block(cones, bits, np.array([1, 2]), np.arange(len(sites), dtype=np.int32),
                              np.array([10, 40]), sub.delaytab, engine_options(spec), 2**62, nt)
            for nt in (1, 2, 5)]
    for r in runs[1:]:
        for k, v in runs[0].items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(v, r[k]), k
            else:
                assert v == r[k], k
    assert runs[0]["cycle"].size > 0


def test_engine_options_are_checked(tmp_path):
    write_random_substrate(str(tmp_path / "build"), 1)
    sub = load_substrate(str(tmp_path / "build"), clock_pins=CLOCK)
    cones = fastpath.compile_sites(sub, sorted(sub.skeleton)[:1])
    spec = CampaignSpec(stage_dir="", state_dir="", cycles=[1], out_dir="", clock_period_ps=100,
                        pulse_widths_ps=[10]).resolved()
    bits = np.zeros((1, len(sub.idx_to_net)), np.uint8)
    args = (cones, bits, np.array([1]), np.zeros(1, np.int32), np.array([10]), sub.delaytab)
    _engine.run_block(*args, engine_options(spec), 2**62, 1)
    with pytest.raises(RuntimeError, match="unknown engine option 'bogus'"):
        _engine.run_block(*args, dict(engine_options(spec), bogus=1), 2**62, 1)
    opts = engine_options(spec)
    opts.pop("delay_policy_is_max")
    with pytest.raises(RuntimeError, match="'delay_policy_is_max' is missing"):
        _engine.run_block(*args, opts, 2**62, 1)


def test_int32_record_columns_are_checked_only_when_a_record_holds_the_value(tmp_path):
    # cycle and width_ps are int32 in the records; the Python loop fails on a value
    # out of range only when it writes a record (np.full(..., np.int32) raises in
    # NumPy 2.2 and later; NumPy 2.0 and earlier wrap the value)
    names = write_random_substrate(str(tmp_path / "build"), 11)
    sub = load_substrate(str(tmp_path / "build"), clock_pins=CLOCK)
    sites = sorted(sub.skeleton)
    cones = fastpath.compile_sites(sub, sites)
    spec = CampaignSpec(stage_dir="", state_dir="", cycles=[1], out_dir="", clock_period_ps=100,
                        pulse_widths_ps=[10]).resolved()
    bits = np.random.default_rng(11).integers(0, 2, (1, len(names)), dtype=np.uint8)

    def run(cycle, widths):
        return _engine.run_block(cones, bits, np.array([cycle]), np.arange(len(sites), dtype=np.int32),
                                 np.array(widths), sub.delaytab, engine_options(spec), 2**62, 2)

    assert run(5, [10, 40])["cycle"].size > 0
    assert run(2**31 + 7, [0])["cycle"].size == 0           # a zero-width SET reaches nothing
    with pytest.raises(OverflowError, match=r"2147483655 out of bounds for int32 \(cycle\)"):
        run(2**31 + 7, [10, 40])
    with pytest.raises(OverflowError, match=r"2147483648 out of bounds for int32 \(pulse width\)"):
        run(5, [10, 2**31])


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="needs signal.setitimer")
def test_run_block_stops_on_a_signal(tmp_path):
    # Ctrl-C (here: a SIGALRM handler that raises) must not wait for the end of the
    # block.  With no observed FF the trials run but record nothing, so the block
    # below needs little memory; uninterrupted it takes over 15 s on a desktop CPU.
    names = write_random_substrate(str(tmp_path / "build"), 11)
    _dump(str(tmp_path / "build"), "timing_checks.json", {"index": {}})
    sub = load_substrate(str(tmp_path / "build"), clock_pins=CLOCK, allow_missing_timing_checks=True)
    cones = fastpath.compile_sites(sub, sorted(sub.skeleton)) * 1500
    spec = CampaignSpec(stage_dir="", state_dir="", cycles=[1], out_dir="", clock_period_ps=100,
                        pulse_widths_ps=[10]).resolved()
    n_cyc = 20
    bits = np.random.default_rng(1).integers(0, 2, (n_cyc, len(names)), dtype=np.uint8)

    class Interrupted(Exception):
        pass

    def handler(signum, frame):
        raise Interrupted

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        t = time.time()
        with pytest.raises(Interrupted):
            _engine.run_block(cones, bits, np.arange(n_cyc), np.arange(len(cones), dtype=np.int32),
                              np.arange(5, 305, 5), sub.delaytab, engine_options(spec), 2**62, 2)
        assert time.time() - t < 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def test_pin_names_with_a_slash_use_the_python_cone_builder(tmp_path):
    # the Python code names a wire pin node "<net>\x01<inst>/<pin>"; a "/" in a pin
    # name could give two pin nodes one name there, so such builds are not compiled in C++
    write_random_substrate(str(tmp_path), 5, pin="A/")
    sub = load_substrate(str(tmp_path), clock_pins=CLOCK, interconnect="transport", n_threads=2)
    assert fastpath.compiled_names(sub) is None
    sub = load_substrate(str(tmp_path), clock_pins=CLOCK, interconnect="ignore", n_threads=2)
    assert compare_cones(sub) > 0


def test_a_site_cones_file_that_cannot_be_read_goes_to_the_python_reader(tmp_path):
    write_random_substrate(str(tmp_path), 3)
    _, idx_to_net = load_net_index(str(tmp_path / "net_index.json"))
    assert load_site_cones(str(tmp_path / "missing.jsonl"), idx_to_net) is None
    assert load_site_cones(str(tmp_path), idx_to_net) is None           # a directory
    assert _engine.parse_site_cones(str(tmp_path / "missing.jsonl")) is None


def test_rejected_sites_raise_the_reference_error(tmp_path):
    write_random_substrate(str(tmp_path / "build"), 2)
    arcs = json.load(open(tmp_path / "build" / "cell_arcs.json"))
    for k in [k for k in arcs if k.startswith("C0:")]:       # gate u0 loses its cell function
        del arcs[k]
    _dump(str(tmp_path / "build"), "cell_arcs.json", arcs)
    sub = load_substrate(str(tmp_path / "build"), clock_pins=CLOCK)
    sites = sorted(sub.skeleton)
    with pytest.raises(SubstrateError, match="cell function missing for instance u0"):
        fastpath.compile_sites(sub, sites)
    ok = [s for s in sites if "u0" not in {e.inst for e in sub.edges.values()
                                           if e.dst_net in set(sub.skeleton[s]["cone_topo_nets"])}]
    assert len(fastpath.compile_sites(sub, ok)) == len(ok)


def test_compiled_cone_round_trip(tmp_path):
    write_random_substrate(str(tmp_path), 4)
    sub = load_substrate(str(tmp_path), clock_pins=CLOCK, interconnect="transport", n_threads=2)
    for c in fastpath.compile_sites(sub, sorted(sub.skeleton)):
        a = c.arrays()
        b = _engine.CompiledCone.from_arrays(a).arrays()
        for k in a:
            if k not in ("net_nid",):
                assert np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k


def test_a_campaign_without_cycles_compiles_no_cone(tmp_path, monkeypatch):
    write_random_substrate(str(tmp_path / "build"), 2)
    arcs = json.load(open(tmp_path / "build" / "cell_arcs.json"))
    for k in [k for k in arcs if k.startswith("C0:")]:
        del arcs[k]
    _dump(str(tmp_path / "build"), "cell_arcs.json", arcs)
    for engine in ("reference", "fast"):
        if engine == "reference":
            monkeypatch.setenv("SETFI_REFERENCE_ENGINE", "1")
        else:
            monkeypatch.delenv("SETFI_REFERENCE_ENGINE", raising=False)
        spec = CampaignSpec(stage_dir=str(tmp_path / "build"), state_dir=str(tmp_path / "state"), cycles=[],
                            out_dir=str(tmp_path / engine), clock_period_ps=100, pulse_widths_ps=[10], jobs=2)
        idx = run_campaign(spec, log=lambda *a: None)
        assert idx["shards"] == [] and idx["counts"]["trials"] == 0


def test_unknown_sites_are_refused(tmp_path):
    write_random_substrate(str(tmp_path / "build"), 2)
    sites = sorted(load_substrate(str(tmp_path / "build"), clock_pins=CLOCK).skeleton)
    spec = CampaignSpec(stage_dir=str(tmp_path / "build"), state_dir=str(tmp_path / "state"), cycles=[1],
                        out_dir=str(tmp_path / "out"), clock_period_ps=100, pulse_widths_ps=[10],
                        sites=[sites[0], sites[1] + "x"])
    with pytest.raises(SubstrateError, match=r"1 name\(s\) are not injection sites"):
        run_campaign(spec, log=lambda *a: None)
    from setfi.campaign.run import check_sites
    with pytest.raises(SubstrateError, match=r"1 name\(s\) are not injection sites"):
        check_sites([sites[1] + "x"], str(tmp_path / "build"))
    check_sites(sites[:3], str(tmp_path / "build"))
