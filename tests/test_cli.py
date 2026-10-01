"""Stage bookkeeping: what makes a stage out of date."""
import json
import os

import pytest

from setfi import cli
from setfi import config as C


def _cfg(tmp_path, tb='// tb.sv\n`include "defs.vh"\n'):
    for name in ("net.v", "net.sdf", "cells.v", "sim.v", "defs.vh"):
        (tmp_path / name).write_text(f"// {name}\n")
    (tmp_path / "tb.sv").write_text(tb)
    (tmp_path / "setfi.toml").write_text("""
[design]
netlist = "net.v"
sdf = "net.sdf"
top = "top"
clock_period_ns = 1.0
[output]
dir = "out"
[library]
cell_functions = ["cells.v"]
[workload]
testbench = ["tb.sv"]
cell_sim_models = ["sim.v"]
""")
    return C.load(str(tmp_path / "setfi.toml"))


OUTPUT_FILES = {  # a minimal complete output of each stage
    "build": {"build_manifest.json": {"artifacts": ["net_index.json"]}, "net_index.json": {}},
    "record": {"cycles.json": {"format": "setfi-cycles/1", "cycles": [1]}, "states/cycle_1.hex": "0"},
    "inject": {"index.json": {"shards": [{"file": "records_c000_s000.npz"}]},
               "records_c000_s000.npz": ""},
}


def _stamp(cfg, stage, **outputs):
    """Record `stage` as run now, with the given output digests and minimal output files."""
    stamps = cli.read_stamps(cfg)
    stamps[stage] = {**cli.fingerprint(cfg, stage, stamps), "outputs": outputs}
    cli._write_stamps(cfg, stamps)
    for name, content in OUTPUT_FILES[stage].items():
        p = os.path.join(cli.paths(cfg)[stage], name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content if isinstance(content, str) else json.dumps(content))


def _current(cfg, stage, require_output=True):
    return cli.stage_status(cfg, stage, cli.read_stamps(cfg), require_output)


def test_input_file_change_makes_stage_stale(tmp_path):
    cfg = _cfg(tmp_path)
    _stamp(cfg, "build", nets="n", graph="g")
    assert _current(cfg, "build")[0]
    (tmp_path / "net.sdf").write_text("// changed delays\n")
    ok, why = _current(cfg, "build")
    assert not ok and "design.sdf" in why


def test_a_stage_depends_on_the_results_it_reads(tmp_path):
    cfg = _cfg(tmp_path)
    _stamp(cfg, "build", nets="n1", graph="g1")
    _stamp(cfg, "record", states="s1")
    _stamp(cfg, "inject", pulses="p1")
    # build re-run with other delays: same nets, other timing graph
    _stamp(cfg, "build", nets="n1", graph="g2")
    assert _current(cfg, "record")[0]
    ok, why = _current(cfg, "inject")
    assert not ok and "the result of build" in why
    # record re-run (e.g. after a comment was added to the testbench): same states
    _stamp(cfg, "inject", pulses="p2")
    _stamp(cfg, "record", states="s1")
    assert _current(cfg, "inject")[0]


def test_only_keys_a_stage_reads_matter(tmp_path):
    cfg = _cfg(tmp_path)
    for st in ("build", "record"):
        _stamp(cfg, st)
    cfg["fault_model"]["pulse_widths_ps"] = [10, 20]      # read by inject only
    assert _current(cfg, "build")[0] and _current(cfg, "record")[0]
    cfg["observe"]["exclude_instances"] = ["u_dbg"]        # read by build only
    assert not _current(cfg, "build")[0] and _current(cfg, "record")[0]


def test_observed_flip_flops_do_not_change_the_recorded_nets(tmp_path):
    cfg = _cfg(tmp_path)
    b = cli.paths(cfg)["build"]
    os.makedirs(b)
    with open(os.path.join(b, "net_index.json"), "w") as f:
        json.dump({"idx_to_net": {"0": "d", "1": "q"}}, f)
    ffx = {"ffid_to_info": {"0": {"d_net": "d"}}, "dnet_to_ffids": {"0": [0]},
           "qnet_to_ffids": {"1": [0]}}
    with open(os.path.join(b, "ff_index.json"), "w") as f:
        json.dump(ffx, f)
    before = cli.output_digests(cfg, "build")["nets"]
    ffx["ffid_to_info"] = {}                               # the flip-flop is no longer observed
    with open(os.path.join(b, "ff_index.json"), "w") as f:
        json.dump(ffx, f)
    assert cli.output_digests(cfg, "build")["nets"] == before


def test_included_files_count_other_files_do_not(tmp_path):
    cfg = _cfg(tmp_path)
    _stamp(cfg, "build")
    _stamp(cfg, "record")
    (tmp_path / "notes.v").write_text("// unrelated\n")
    assert _current(cfg, "record")[0]
    (tmp_path / "defs.vh").write_text("`define DEPTH 8\n")
    ok, why = _current(cfg, "record")
    assert not ok and "workload.include_files" in why


def test_execution_settings_and_moving_the_project_do_not_matter(tmp_path):
    import shutil
    cfg = _cfg(tmp_path)
    fp = cli.fingerprint(cfg, "inject")["fingerprint"]
    cfg["inject"]["jobs"] = 8
    assert cli.fingerprint(cfg, "inject")["fingerprint"] == fp
    fb = cli.fingerprint(cfg, "record")["fingerprint"]
    shutil.copytree(tmp_path, tmp_path.parent / (tmp_path.name + "_moved"))
    moved = C.load(str(tmp_path.parent / (tmp_path.name + "_moved") / "setfi.toml"))
    assert cli.fingerprint(moved, "record")["fingerprint"] == fb


def test_deleted_stage_directories(tmp_path):
    import shutil
    cfg = _cfg(tmp_path)
    _stamp(cfg, "build")
    _stamp(cfg, "record")
    _stamp(cfg, "inject")
    shutil.rmtree(cli.paths(cfg)["build"])
    shutil.rmtree(cli.paths(cfg)["record"])
    # the pulse records still match the inputs; build and record have to run again
    assert _current(cfg, "build", require_output=False)[0]
    assert not _current(cfg, "build")[0]
    assert _current(cfg, "inject")[0]


def test_analysis_names(tmp_path):
    cfg = _cfg(tmp_path)
    _stamp(cfg, "inject")
    for bad in ("..", "../x", "a/b", ""):
        with pytest.raises(cli.SetfiCLIError, match="--name"):
            cli.do_analyze(cfg, bad)
    cfg["analyze"]["width_models"] = [{"type": "uniform"}, {"type": "exponential", "tau_ps": 40}]
    with pytest.raises(cli.SetfiCLIError, match="exactly one width model"):
        cli.do_analyze(cfg, "mine")


def test_config_errors_are_reported_without_traceback(tmp_path, capsys):
    _cfg(tmp_path)
    text = (tmp_path / "setfi.toml").read_text() + \
        "[analyze]\nwidth_models = [{type = \"gaussian\", mean_ps = 90}]\n"
    (tmp_path / "setfi.toml").write_text(text)
    rc = cli.main(["config", "-c", str(tmp_path / "setfi.toml")])
    err = capsys.readouterr().err
    assert rc == 1 and "width_models" in err and "sd_ps" in err and "Traceback" not in err


def test_partly_deleted_outputs(tmp_path):
    cfg = _cfg(tmp_path)
    for st in ("build", "record", "inject"):
        _stamp(cfg, st)
    os.remove(os.path.join(cli.paths(cfg)["record"], "states", "cycle_1.hex"))
    ok, why = _current(cfg, "record")
    assert not ok and "cycle_1.hex" in why
    os.remove(os.path.join(cli.paths(cfg)["inject"], "records_c000_s000.npz"))
    with pytest.raises(cli.SetfiCLIError, match="setfi inject --force"):
        cli.do_analyze(cfg)


def test_a_simulator_key_at_its_default_changes_nothing(tmp_path):
    cfg = _cfg(tmp_path)
    fp = cli.fingerprint(cfg, "record")["fingerprint"]
    cfg["simulator"]["iverilog"] = "iverilog"
    assert cli.fingerprint(cfg, "record")["fingerprint"] == fp


def test_status_and_config_commands(tmp_path, capsys):
    _cfg(tmp_path)
    os.remove(tmp_path / "net.sdf")
    assert cli.main(["config", "-c", str(tmp_path / "setfi.toml")]) == 1
    out = capsys.readouterr().out
    assert "[design]" in out and "# not found: [design] sdf:" in out
    assert cli.main(["status", "-c", str(tmp_path / "setfi.toml")]) == 0
    out = capsys.readouterr().out
    assert "build    to run  (it has not run)" in out and "analyze  no results" in out
