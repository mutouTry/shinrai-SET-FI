"""Command templates, simulator presets, spec construction, env scrubbing."""
import pytest

from setfi.workload import spec as SP
from setfi.workload.errors import WorkloadError
from setfi.workload.simulate import clean_env

ARTS = "/work/out/record/sim"
SIF = "/opt/containers/vcs.sif"


def _scalars(prefix="pass1", work=ARTS, out=ARTS):
    return {"work_dir": work, "output_dir": out, "sim_bin": f"{work}/simv_{prefix}",
            "filelist": f"{work}/{prefix}_filelist.f", "top": "setfi_tb_wrapper"}


def test_vcs_preset_commands_in_a_container():
    """The full compile and run commands, token for token, binds in the given order."""
    sim = SP.vcs_preset(container=SIF,
                        binds=["/tools:/tools", "{work_dir}:{work_dir}", "/data:/data"],
                        bind_output_dir=False, extra_args=["-full64"])
    cmds = sim.commands(scalars=_scalars(), defines=["SYNTHESIS", "PERIOD=2"], plusargs=[])
    prefix = ["singularity", "exec", "--cleanenv", "-B", "/tools:/tools",
              "-B", f"{ARTS}:{ARTS}", "-B", "/data:/data", SIF]
    assert cmds["compile"] == prefix + [
        "vcs", "-full64", "-sverilog", "-debug_access",
        "-timescale=1ns/1ps", "+define+SYNTHESIS", "+define+PERIOD=2",
        "+notimingchecks", "+nospecify", "-o", f"{ARTS}/simv_pass1", "-f", f"{ARTS}/pass1_filelist.f"]
    assert cmds["run"] == prefix + [f"{ARTS}/simv_pass1"]
    assert sim.recorder == "program"
    assert sim.required_files == [SIF]


def test_vcs_preset_default_output_bind_and_plusargs():
    sim = SP.vcs_preset(container=SIF)
    cmds = sim.commands(scalars=_scalars("pass2", work="/o/sim", out="/o"), defines=[],
                        plusargs=["+setfi_outdir=/o/cycle_state"])
    assert cmds["run"] == ["singularity", "exec", "--cleanenv", "-B", "/o:/o", SIF,
                           "/o/sim/simv_pass2", "+setfi_outdir=/o/cycle_state"]


def test_vcs_preset_native():
    sim = SP.vcs_preset(vcs_binary="/tools/vcs/bin/vcs")
    cmds = sim.commands(scalars=_scalars(), defines=["A"], plusargs=[])
    assert cmds["compile"][0] == "/tools/vcs/bin/vcs"
    assert cmds["run"] == [f"{ARTS}/simv_pass1"]
    assert sim.required_files == []


def test_icarus_preset():
    sim = SP.icarus_preset(extra_args=["-Wno-timescale"])
    cmds = sim.commands(scalars=_scalars(), defines=["SYNTHESIS", "P=1"], plusargs=["+x=1"])
    assert cmds["compile"] == ["iverilog", "-g2012", "-Wno-timescale", "-DSYNTHESIS", "-DP=1",
                               "-s", "setfi_tb_wrapper", "-o", f"{ARTS}/simv_pass1",
                               "-c", f"{ARTS}/pass1_filelist.f"]
    assert cmds["run"] == ["vvp", "-n", f"{ARTS}/simv_pass1", "+x=1"]
    assert sim.recorder == "strobe" and sim.timescale_preamble == "1ns/1ps"


def test_render_command_rules():
    lists = {"xs": ["a", "b"], "none": []}
    scal = {"w": "/w"}
    assert SP.render_command(["t", "{xs}", "{none}", "-M{w}/c", "{{lit}}"], scal, lists) == \
        ["t", "a", "b", "-M/w/c", "{lit}"]
    with pytest.raises(WorkloadError):
        SP.render_command(["--x={xs}"], scal, lists)          # list inside a token
    with pytest.raises(WorkloadError):
        SP.render_command(["{nope}"], scal, lists)            # unknown placeholder


def test_extra_args_may_use_scalars():
    sim = SP.vcs_preset(extra_args=["-Mdir={work_dir}/csrc"])
    assert f"-Mdir={ARTS}/csrc" in sim.commands(scalars=_scalars(), defines=[], plusargs=[])["compile"]


def test_simulator_from_dict():
    s = SP.simulator_from_dict({"preset": "vcs", "container": SIF, "binds": ["/tools"],
                                "extra_args": ["-full64"], "overrides": {"name": "my-vcs"}})
    assert s.name == "my-vcs" and s.binds == ["/tools", "{output_dir}:{output_dir}"]
    s2 = SP.simulator_from_dict({"name": "custom", "compile": ["mysim", "-f", "{filelist}"],
                                 "run": ["{sim_bin}"], "recorder": "strobe"})
    assert s2.recorder == "strobe"
    for bad in ({"preset": "nosuch"}, {"preset": "vcs", "bogus": 1},
                {"name": "c", "compile": ["x"], "run": ["y"], "recorder": "always"}):
        with pytest.raises(WorkloadError):
            SP.simulator_from_dict(bad)


def _min_spec(**kw):
    d = dict(netlist="n.v", top_module="top", net_index="ni.json", testbench="tb.sv",
             cell_models=["cells.v"], output_dir="out", simulator={"preset": "icarus"})
    d.update(kw)
    return d


def test_workload_spec_from_dict_defaults():
    s = SP.WorkloadSpec.from_dict(_min_spec())
    assert [str(p) for p in s.testbench] == ["tb.sv"]
    assert s.simulator.name == "icarus"
    assert (s.n_sample_cycles, s.sample_seed, s.padding_cycles, s.warmup_cycles) == (30, 12345, 5, 0)
    assert s.max_x_frac == 0.02 and s.reserve_factor == 3
    assert s.expected_pass_string == "RESULT: PASS"
    assert s.cycle_window_csv is None and s.ff_index is None
    d = s.to_dict()
    assert d["netlist"] == "n.v" and d["simulator"]["recorder"] == "strobe"


@pytest.mark.parametrize("bad", [
    {"nosuch_key": 1},
    {"reset_polarity": "low"},
    {"n_sample_cycles": 0},
    {"max_x_frac": 1.5},
    {"testbench": []},
])
def test_workload_spec_rejects(bad):
    with pytest.raises(WorkloadError):
        SP.WorkloadSpec.from_dict(_min_spec(**bad))


def test_clean_env():
    sim = SP.vcs_preset(env_unset=["LD_LIBRARY_PATH"])
    env = clean_env(sim, {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/c/lib", "HOME": "/home/u"})
    assert env == {"PATH": "/usr/bin", "HOME": "/home/u"}
    assert clean_env(SP.icarus_preset(), {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/c"}) == \
        {"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/c"}
