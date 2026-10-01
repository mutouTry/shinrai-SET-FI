import os

import pytest

from setfi import config as C


def _write(tmp_path, text):
    p = tmp_path / "setfi.toml"
    p.write_text(text)
    return str(p)


MINIMAL = """
[design]
netlist = "d/net.v"
sdf = "d/net.sdf"
top = "top"
clock_period_ns = 1.22
[library]
cell_functions = ["lib/cells.v"]
[output]
dir = "runs/x"
"""


def test_defaults_and_relative_paths(tmp_path):
    cfg = C.load(_write(tmp_path, MINIMAL), sections=["design", "library", "output"])
    assert cfg["design"]["netlist"] == os.path.join(str(tmp_path), "d/net.v")
    assert cfg["library"]["cell_functions"] == [os.path.join(str(tmp_path), "lib/cells.v")]
    assert cfg["fault_model"]["pulse_widths_ps"] == [20, 40, 60, 80, 100, 120, 140, 160, 180]
    assert cfg["propagation"]["delay_choice"] == "max"
    assert cfg["analyze"]["phase_domain"] == "half_open"


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(C.ConfigError, match="unknown key"):
        C.load(_write(tmp_path, MINIMAL + "\n[propagation]\ndelay_polcy = 'min'\n"),
               sections=["design"])


def test_unknown_section_is_an_error(tmp_path):
    with pytest.raises(C.ConfigError, match="unknown section"):
        C.load(_write(tmp_path, MINIMAL + "\n[simulaton]\n"), sections=["design"])


def test_missing_required(tmp_path):
    with pytest.raises(C.ConfigError, match="clock_period_ns is required"):
        C.load(_write(tmp_path, MINIMAL.replace("clock_period_ns = 1.22", "")),
               sections=["design"])


def test_type_and_choice_errors(tmp_path):
    with pytest.raises(C.ConfigError, match="integers"):
        C.load(_write(tmp_path, MINIMAL + "[fault_model]\npulse_widths_ps = [20, 'x']\n"),
               sections=["design"])
    with pytest.raises(C.ConfigError, match="one of"):
        C.load(_write(tmp_path, MINIMAL + "[propagation]\nsdf_rail = 'typ'\n"),
               sections=["design"])


def test_template_parses_and_names_every_key(tmp_path):
    text = C.template()
    for sec, keys in C.schema().items():
        assert f"[{sec}]" in text
        for name in keys:
            assert f"\n{name} = " in text or f"\n# {name} =" in text
    cfg = C.load(_write(tmp_path, text), check_files=False)
    assert cfg["design"]["top"] == "my_top"


def _full(tmp_path, extra=""):
    for f in ("d/net.v", "d/net.sdf", "lib/cells.v", "tb.sv", "sim.v"):
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("")
    return _write(tmp_path, MINIMAL + '[workload]\ntestbench = ["tb.sv"]\ncell_sim_models = ["sim.v"]\n'
                  + extra)


@pytest.mark.parametrize("extra, message", [
    ("tb_substitutions = [{regex = 'a('}]", "tb_substitutions, entry 1: needs the keys"),
    ("tb_substitutions = [{regex = 'a(', replacement = 'b'}]", "bad regex"),
    ("macro_rtl_substitutions = [{macro_module = 'm', rtl_file = 'no.v'}]",
     "not found:\n  \\[workload\\] macro_rtl_substitutions, entry 1"),
])
def test_table_entries_are_checked(tmp_path, extra, message):
    with pytest.raises(C.ConfigError, match=message):
        C.load(_full(tmp_path, extra + "\n"))


def test_scan_pin_pairs_are_checked(tmp_path):
    path = _full(tmp_path)
    text = open(path).read().replace('cell_functions = ["lib/cells.v"]',
                                     'cell_functions = ["lib/cells.v"]\nscan_data_pin_pairs = [{d = "D"}]')
    with pytest.raises(C.ConfigError, match="scan_data_pin_pairs, entry 1"):
        C.load(_write(tmp_path, text))


@pytest.mark.parametrize("models, message", [
    ('[{type = "gaussian", mean_ps = "abc", sd_ps = 40}]', "mean_ps must be a number"),
    ('[{type = "uniform"}, {type = "uniform"}]', "same distribution twice"),
    ('[]', "at least one"),
])
def test_width_models_are_checked(tmp_path, models, message):
    _full(tmp_path)
    with pytest.raises(C.ConfigError, match=message):
        C.load(_full(tmp_path, f"[analyze]\nwidth_models = {models}\n"))


def test_used_by():
    assert C.used_by("design", "clock_period_ns") == ["inject"]
    assert C.used_by("design", "netlist") == ["build", "record"]
    assert C.used_by("inject", "jobs") == ["inject"]
