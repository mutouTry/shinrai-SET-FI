"""Map the resolved configuration onto each stage's own spec."""
from __future__ import annotations

import os
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List

from .campaign.substrate import q_ns_to_ps


def clock_period_ps(cfg) -> int:
    return q_ns_to_ps(cfg["design"]["clock_period_ns"])


def campaign_spec_from_config(cfg, paths: Dict[str, str]):
    from .campaign.run import CampaignSpec
    from .campaign.state import read_sampled_cycles
    T = clock_period_ps(cfg)
    prop = cfg["propagation"]
    inj = cfg["inject"]
    state_dir, meta = state_paths(paths["record"])
    horizon = int((Decimal(str(prop["horizon_factor"])) * T).to_integral_value(rounding=ROUND_HALF_UP))
    return CampaignSpec(
        stage_dir=paths["build"], state_dir=state_dir, cycles=read_sampled_cycles(meta),
        out_dir=paths["inject"], clock_period_ps=T,
        pulse_widths_ps=sorted(set(int(w) for w in cfg["fault_model"]["pulse_widths_ps"])),
        sim_horizon_ps=horizon, delay_choice=prop["delay_choice"],
        electrical_masking=prop["electrical_masking"], em_margin_ps=prop["em_margin_ps"],
        interconnect=prop["interconnect"], clock_pins=cfg["library"]["timing_check_clock_pins"],
        allow_missing_timing_checks=cfg["observe"]["allow_missing_timing_checks"],
        sites=cfg["fault_model"]["sites"] or None, cycles_per_shard=inj["cycles_per_shard"],
        jobs=inj["jobs"], phase_grid_steps=inj["phase_grid_steps"] or None)


def state_paths(record_dir: str):
    """(states directory, cycles.json) written by the record stage."""
    return os.path.join(record_dir, "states"), os.path.join(record_dir, "cycles.json")


SIM_KEYS = {
    "icarus": {"iverilog", "vvp", "generation", "timescale", "extra_args", "run_args", "env_unset"},
    "vcs": {"vcs_binary", "container", "container_runtime", "container_exec_args", "binds",
            "bind_output_dir", "extra_args", "run_args", "env_unset"},
    "custom": {"compile", "run", "wrapper", "binds", "bind_flag", "define_format", "recorder",
               "timescale", "env_unset", "extra_args", "run_args", "include_format"},
}


def simulator_defaults(cfg) -> Dict[str, Any]:
    """The values the preset uses for the [simulator] keys that are not set."""
    import dataclasses
    import inspect
    from .workload.spec import PRESETS, SimulatorSpec
    preset = cfg["simulator"]["preset"]
    if preset in PRESETS:
        sig = inspect.signature(PRESETS[preset])
        out = {k: p.default for k, p in sig.parameters.items() if k in SIM_KEYS[preset]}
    else:
        fields = {f.name: f for f in dataclasses.fields(SimulatorSpec)}
        out = {}
        for k in SIM_KEYS["custom"]:
            name = "timescale_preamble" if k == "timescale" else k
            f = fields.get(name)
            if f is not None and f.default is not dataclasses.MISSING:
                out[k] = f.default
        out["include_format"] = "+incdir+{dir}"
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}
    return {k: v for k, v in out.items() if v is not None}


def simulator_dict(cfg) -> Dict[str, Any]:
    from .config import ConfigError
    sim = cfg["simulator"]
    preset = sim["preset"]
    given = {k: v for k, v in sim.items() if k != "preset" and v is not None}
    bad = sorted(set(given) - SIM_KEYS[preset])
    if bad:
        raise ConfigError(f"[simulator] {bad} do not apply to preset {preset!r} "
                          f"(it takes {sorted(SIM_KEYS[preset])})")
    include_format = given.pop("include_format", None) or {"icarus": "-I{dir}"}.get(preset, "+incdir+{dir}")
    incs = [include_format.format(dir=d) for d in include_dirs(cfg)]
    if incs:
        given["extra_args"] = list(given.get("extra_args") or []) + incs
    if preset == "custom":
        missing = [k for k in ("compile", "run") if k not in given]
        if missing:
            raise ConfigError(f"[simulator] preset 'custom' needs {missing}")
        if "timescale" in given:
            given["timescale_preamble"] = given.pop("timescale")
        return {"name": "custom", **given}
    return {"preset": preset, **given}


def include_dirs(cfg) -> List[str]:
    """`include search path: the testbench directories, then workload.include_dirs."""
    out: List[str] = []
    for d in [os.path.dirname(f) for f in cfg["workload"]["testbench"]] + list(cfg["workload"]["include_dirs"]):
        if d not in out:
            out.append(d)
    return out


def is_liberty(path: str) -> bool:
    return path.lower().endswith((".lib", ".liberty"))


def cell_model_files(cfg, paths: Dict[str, str]) -> List[str]:
    """The cell models the testbench simulation compiles."""
    if cfg["workload"]["cell_sim_models"]:
        return list(cfg["workload"]["cell_sim_models"])
    files = cfg["library"]["cell_functions"]
    if any(is_liberty(f) for f in files):
        return [os.path.join(paths["build"], "cell_functions.v")]
    return list(files)


def workload_spec_from_config(cfg, paths: Dict[str, str]):
    from .workload import WorkloadSpec
    wl = cfg["workload"]
    d = cfg["design"]
    b = paths["build"]
    return WorkloadSpec(
        netlist=d["netlist"], top_module=d["top"], net_index=os.path.join(b, "net_index.json"),
        ff_index=os.path.join(b, "ff_index.json"),
        testbench=wl["testbench"], cell_models=cell_model_files(cfg, paths), output_dir=paths["record"],
        simulator=simulator_dict(cfg), tb_top_module=wl["tb_top"], tb_dut_inst=wl["dut_instance"],
        rtl_dut_files=wl["rtl_files"], tb_data_files=wl["data_files"], clock_port=wl["clock"],
        reset_port=wl["reset"], reset_polarity="active_low" if wl["reset_active"] == "low" else "active_high",
        defines=wl["defines"], hier_substitutions=wl["tb_substitutions"],
        auto_derive_hier_substitutions=wl["auto_derive_substitutions"],
        macro_rtl_subs=wl["macro_rtl_substitutions"], netlist_strip_modules=wl["netlist_strip_modules"],
        netlist_substitute_files=wl["netlist_substitute_files"], n_sample_cycles=wl["n_cycles"],
        sample_seed=wl["seed"], padding_cycles=wl["padding_cycles"], warmup_cycles=wl["warmup_cycles"],
        cycle_window_csv=wl["cycle_window_csv"], max_x_frac=wl["max_x_fraction"],
        reserve_factor=wl["reserve_factor"], expected_pass_string=wl["pass_string"],
        compile_timeout_s=wl["compile_timeout_s"], sim_timeout_s=wl["sim_timeout_s"])


def cell_function_file(cfg, out_dir: str, log=print) -> str:
    """The single assign-style Verilog file the build reads.

    One Verilog file is used as it is.  Liberty files are translated, and several
    files are concatenated, into <out_dir>/cell_functions.v.
    """
    files = cfg["library"]["cell_functions"]
    is_lib = [is_liberty(f) for f in files]
    if len(files) == 1 and not is_lib[0]:
        return files[0]
    from .library.liberty import liberty_to_verilog
    parts = []
    for f, lib in zip(files, is_lib):
        if lib:
            text, models = liberty_to_verilog([f])
            n_bad = sum(1 for m in models.values() if m.kind == "unsupported")
            log(f"[library] {f}: {len(models)} cells, {n_bad} not modelled")
            parts.append(text)
        else:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                parts.append(fh.read())
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "cell_functions.v")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts))
    return path


def build_spec_from_config(cfg, paths: Dict[str, str], log=print):
    from .build import BuildSpec, LibrarySpec
    lib = cfg["library"]
    obs = cfg["observe"]
    b = cfg["build"]
    fm = cfg["fault_model"]
    lspec = LibrarySpec(
        behavioral_verilog=cell_function_file(cfg, paths["build"], log),
        q_pin_names=tuple(lib["q_pins"]), d_pin_names=tuple(lib["d_pins"]),
        clock_pin_names=tuple(lib["clock_pins"]), async_pin_names=tuple(lib["async_pins"]),
        mirror_scan_data_checks=lib["mirror_scan_data_checks"],
        scan_data_pin_pairs=tuple((str(p["data"]), str(p["scan"])) for p in lib["scan_data_pin_pairs"]),
        untimed_cells=tuple(lib["untimed_cells"]), blackbox_cells=tuple(lib["blackbox_cells"]),
        non_observed_cells=tuple(obs["exclude_cells"]),
        observed_d_pins=tuple(obs["data_pins"] or lib["d_pins"]))
    d = cfg["design"]
    return BuildSpec(
        netlist=d["netlist"], sdf=d["sdf"], top=d["top"], out_dir=paths["build"], library=lspec,
        exclude_d_from_src=not fm["inject_ff_d_nets"],
        sdf_rail=cfg["propagation"]["sdf_rail"],
        non_observed_instances=tuple(obs["exclude_instances"]), self_check=b["self_check"],
        self_check_samples=b["self_check_samples"], self_check_seed=b["self_check_seed"],
        strict_sdf_crosscheck=b["strict_sdf_crosscheck"], max_undriven_nets=b["max_undriven_nets"],
        keep_intermediates=b["keep_intermediates"], allow_missing_delays=b["allow_missing_delays"])


def check_testbench(cfg) -> None:
    """Catch testbench/config mismatches before any simulation runs."""
    import re
    from .config import ConfigError
    wl = cfg["workload"]
    text = ""
    for f in wl["testbench"]:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            text += fh.read() + "\n"
    text = re.sub(r"/\*.*?\*/", " ", re.sub(r"//[^\n]*", " ", text), flags=re.S)
    m = re.search(rf"\bmodule\s+{re.escape(wl['tb_top'])}\b(.*?)\bendmodule\b", text, re.S)
    if m is None:
        raise ConfigError(f"[workload] tb_top = {wl['tb_top']!r}: no such module in "
                          f"workload.testbench")
    body = m.group(1)
    for key in ("clock", "reset"):
        if not wl[key]:
            continue
        if not re.search(rf"\b{re.escape(wl[key])}\b", body):
            raise ConfigError(f"[workload] {key} = {wl[key]!r}: no signal of that name in module "
                              f"{wl['tb_top']}")
    if not re.search(rf"\b{re.escape(wl['dut_instance'])}\s*\(", body):
        raise ConfigError(f"[workload] dut_instance = {wl['dut_instance']!r}: no instance of that "
                          f"name in module {wl['tb_top']}")
