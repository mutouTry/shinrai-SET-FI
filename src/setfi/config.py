"""The top-level configuration: one TOML file, declared here once.

Every key has a type, a default (or is required) and a one-line description.
The same table drives validation, the resolved-config dump and the commented
template written by ``setfi init``.  Unknown keys are errors, so a misspelled
key can never be silently ignored.  Relative paths are resolved against the
directory of the configuration file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class ConfigError(ValueError):
    pass


REQUIRED = object()


@dataclass(frozen=True)
class Key:
    type: str            # str | path | paths | int | float | bool | ints | strs | choice:<a|b> | table
    default: Any
    doc: str
    example: Any = None  # shown in the template for required keys


def _k(t, d, doc, example=None):
    return Key(t, d, doc, example)


# What each section is for (shown in the template).
SECTIONS = {
    "design": "The design under test.",
    "output": "Where everything is written.",
    "library": "The standard-cell library: cell logic and naming conventions.",
    "workload": "The testbench, and how clock cycles are sampled from it.",
    "simulator": "The Verilog simulator that runs the testbench.",
    "fault_model": "Which SETs are injected.",
    "observe": "Which flip-flops count as observation points.",
    "propagation": "How a pulse propagates through the gates.",
    "inject": "How the injection campaign runs.",
    "analyze": "How upset probabilities are computed.",
    "build": "Advanced options of the netlist and SDF reader.",
}

STAGES = ("build", "record", "inject", "analyze")

# The configuration each stage reads: whole sections, or (section, key) pairs.  A
# stage is re-run when one of these values (for files: their content) changes.
STAGE_KEYS = {
    "build": ["library", "observe", "build", ("design", "netlist"), ("design", "sdf"),
              ("design", "top"), ("fault_model", "inject_ff_d_nets"),
              ("propagation", "sdf_rail")],
    "record": ["workload", "simulator", ("design", "netlist"), ("design", "top")],
    "inject": ["inject", ("propagation", "delay_choice"), ("propagation", "electrical_masking"),
               ("propagation", "em_margin_ps"), ("propagation", "interconnect"),
               ("propagation", "horizon_factor"), ("design", "clock_period_ns"),
               ("fault_model", "pulse_widths_ps"), ("fault_model", "sites"),
               ("library", "timing_check_clock_pins"),
               ("observe", "allow_missing_timing_checks")],
    "analyze": ["analyze"],
}
# Keys that change how a stage runs, not its result.
EXECUTION_ONLY = {("inject", "jobs"), ("inject", "cycles_per_shard"),
                  ("workload", "compile_timeout_s"), ("workload", "sim_timeout_s")}


def stage_keys(stage: str, cfg: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for item in STAGE_KEYS[stage]:
        keys = [item] if isinstance(item, tuple) else [(item, k) for k in cfg[item]]
        out.extend(k for k in keys if k not in EXECUTION_ONLY)
    return out


def used_by(section: str, key: str) -> List[str]:
    return [st for st in STAGES
            if any(it == section or it == (section, key) for it in STAGE_KEYS[st])]

# Section -> key -> Key.
SCHEMA: Dict[str, Dict[str, Key]] = {
    "design": {
        "netlist": _k("path", REQUIRED, "Gate-level netlist (structural Verilog, named port "
                      "connections).", "design/mapped.v"),
        "sdf": _k("path", REQUIRED, "SDF of that netlist; its (DESIGN ...) must equal `top`.",
                  "design/mapped.sdf"),
        "top": _k("str", REQUIRED, "Top module of the netlist.", "my_top"),
        "clock_period_ns": _k("float", REQUIRED,
                              "Clock period T of the analysis: SET start times are uniform over "
                              "[0, T) and flip-flops capture at T. Normally the period the SDF was "
                              "timed for; the testbench clock period does not matter.", 1.0),
    },
    "output": {
        "dir": _k("path", REQUIRED, "Output directory; one sub-directory per stage.",
                  "runs/my_design"),
    },
    "library": {
        "cell_functions": _k("paths", REQUIRED,
                             "Logic of every cell: Liberty files (.lib) or Verilog cell modules "
                             "(see docs/CELLS.md).", ["lib/cells.lib"]),
        "q_pins": _k("strs", ["Q", "QN", "QB", "Q_N", "QBAR"],
                     "Names of flip-flop output pins (a sequential cell needs one)."),
        "d_pins": _k("strs", ["D", "DI", "DIN", "DATA", "I"],
                     "Names of flip-flop data pins; the first one a cell has is its data pin."),
        "clock_pins": _k("strs", ["CP", "CK", "CLK", "CLKN", "G", "E"],
                         "Names of clock pins, used only for cells whose model does not name the "
                         "clock in `always @(posedge/negedge ...)` (e.g. latches)."),
        "async_pins": _k("strs", ["RN", "SN", "R", "S", "RESET", "SET", "CDN", "SDN", "CLRN", "PRN"],
                         "Names of asynchronous set/reset pins (never taken as the data pin)."),
        "scan_data_pin_pairs": _k("tables", [{"data": "D", "scan": "SCD"}],
                                  "Data / scan-data pin pairs of scan flip-flops: when the SDF has "
                                  "setup/hold checks for only one pin of a pair, they are copied "
                                  "to the other."),
        "mirror_scan_data_checks": _k("bool", True, "Enable the copying above."),
        "untimed_cells": _k("strs", [],
                            "Regular expressions (whole cell name) of cells that have no SDF "
                            "entry. Cells without inputs or without outputs (ties, fillers) are "
                            "always treated so."),
        "blackbox_cells": _k("strs", [],
                             "Regular expressions (whole cell name) of cells that are neither in "
                             "the library nor defined in the netlist, e.g. memories. They are "
                             "neither injected nor observed."),
        "timing_check_clock_pins": _k("strs", ["CP", "CK", "CLK"],
                                      "Clock pins whose SDF SETUP/HOLD checks give a flip-flop's "
                                      "sampling window, in order of preference; if none has "
                                      "checks, the one clock pin that has them is used."),
    },
    "workload": {
        "testbench": _k("paths", REQUIRED,
                        "Testbench source file(s). It instantiates the design, is run unchanged "
                        "on the gate-level netlist, and prints `pass_string` when it passes. It "
                        "must not change design inputs between a falling and the next rising "
                        "clock edge (see docs/SIMULATORS.md).", ["tb/testbench.sv"]),
        "tb_top": _k("str", "testbench", "Top module of the testbench."),
        "dut_instance": _k("str", "dut", "Instance name of the design in `tb_top`."),
        "clock": _k("str", "clk", "Clock signal in `tb_top`."),
        "reset": _k("str", "rst_n", "Reset signal in `tb_top`; cycles are counted after reset. "
                                    "Empty: no reset, cycles are counted from the first clock "
                                    "edge."),
        "reset_active": _k("choice:low|high", "low", "Reset polarity."),
        "cell_sim_models": _k("paths", [],
                              "Verilog simulation models of the library cells, e.g. those "
                              "shipped with the library. Default: the cells of "
                              "library.cell_functions (Liberty files as translated by build "
                              "into build/cell_functions.v)."),
        "data_files": _k("paths", [], "Files the testbench opens at run time; copied into the "
                                      "simulation directory."),
        "include_dirs": _k("dirs", [], "Directories searched for `include files, besides the "
                                       "directories of the testbench files."),
        "defines": _k("strs", [], "Macro definitions, NAME or NAME=VALUE."),
        "n_cycles": _k("int", 100, "Number of clock cycles sampled from the run."),
        "seed": _k("int", 12345, "Seed of the cycle sampling."),
        "padding_cycles": _k("int", 5, "Cycles not sampled at the start and end of the run."),
        "warmup_cycles": _k("int", 0, "Further cycles not sampled after reset."),
        "cycle_window_csv": _k("path", None,
                               "CSV with columns start_cycle,end_cycle (inclusive; cycle N "
                               "begins at the N-th rising clock edge after reset): sample only "
                               "inside these windows, e.g. the kernel of a program.", '"windows.csv"'),
        "max_x_fraction": _k("float", 0.02,
                             "A sampled cycle whose recorded state has a larger fraction of "
                             "unknown (X/Z) digits is replaced by another cycle."),
        "reserve_factor": _k("int", 3, "Replacement cycles drawn per sampled cycle."),
        "pass_string": _k("str", "RESULT: PASS", "Text the testbench prints when it passes."),
        "tb_substitutions": _k("tables", [],
                               "Regex rewrites of the testbench text for references to signals "
                               "that synthesis renamed or flattened: [{regex = '...', "
                               "replacement = '...'}] (docs/SIMULATORS.md)."),
        "auto_derive_substitutions": _k("bool", False,
                                        "Derive such rewrites for references into sub-modules "
                                        "(e.g. `dut.u_core.x`) from the RTL in `rtl_files`."),
        "rtl_files": _k("paths", [], "RTL of the design, only for auto_derive_substitutions."),
        "macro_rtl_substitutions": _k("tables", [],
                                      "Behavioural RTL for macros without a model (e.g. "
                                      "memories in library.blackbox_cells): [{macro_module = "
                                      "'...', rtl_file = '...'}] (docs/SIMULATORS.md)."),
        "netlist_strip_modules": _k("strs", [], "Modules of the netlist to simulate from other "
                                                "sources: removed from the simulated copy of the "
                                                "netlist."),
        "netlist_substitute_files": _k("paths", [], "Sources that replace stripped modules."),
        "compile_timeout_s": _k("int", 600, "Compile timeout (seconds)."),
        "sim_timeout_s": _k("int", 600, "Simulation timeout per run (seconds)."),
    },
    "simulator": {
        "preset": _k("choice:icarus|vcs|custom", "icarus",
                     "Simulator: a preset, or `custom` with your own command templates "
                     "(docs/SIMULATORS.md). Keys below marked [x] apply to preset x only."),
        "extra_args": _k("strs", None, "[all] Extra compile arguments, e.g. [\"-full64\"] for VCS.", '[]'),
        "run_args": _k("strs", None, "[all] Extra run-time arguments.", '[]'),
        "env_unset": _k("strs", None, "[all] Environment variables removed before the simulator "
                                      "runs, e.g. [\"LD_LIBRARY_PATH\"].", '[]'),
        "timescale": _k("str", None, "[icarus, custom] `timescale for sources that declare "
                                     "none (icarus default \"1ns/1ps\"; custom default: "
                                     "none).", '"1ns/1ps"'),
        "iverilog": _k("str", None, "[icarus] iverilog executable (default \"iverilog\").", '"iverilog"'),
        "vvp": _k("str", None, "[icarus] vvp executable (default \"vvp\").", '"vvp"'),
        "generation": _k("str", None, "[icarus] Language flag (default \"-g2012\").", '"-g2012"'),
        "vcs_binary": _k("str", None, "[vcs] vcs executable (default \"vcs\").", '"vcs"'),
        "container": _k("path", None, "[vcs] Container image to run VCS in (default: none).", '"vcs.sif"'),
        "container_runtime": _k("str", None, "[vcs] Container runtime (default \"singularity\").", '"singularity"'),
        "container_exec_args": _k("strs", None, "[vcs] Arguments between the container runtime "
                                                "and the binds (default [\"exec\", "
                                                "\"--cleanenv\"]).", '["exec", "--cleanenv"]'),
        "binds": _k("strs", None, "[vcs, custom] Container bind specifications.", '[]'),
        "bind_output_dir": _k("bool", None, "[vcs] Bind the output directory (default true).", 'true'),
        "compile": _k("strs", None, "[custom] Compile command template.", '["xrun", "-f", "{filelist}"]'),
        "run": _k("strs", None, "[custom] Run command template.", '["{sim_bin}", "{plusargs}"]'),
        "wrapper": _k("strs", None, "[custom] Prefix of both commands, e.g. a container exec.", '[]'),
        "bind_flag": _k("str", None, "[custom] Flag before each bind (default \"-B\").", '"-B"'),
        "define_format": _k("str", None, "[custom] Form of one define (default \"+define+{define}\").", '"+define+{define}"'),
        "include_format": _k("str", None, "[custom] Form of one include directory "
                                          "(default \"+incdir+{dir}\").", '"+incdir+{dir}"'),
        "recorder": _k("choice:strobe|program", None,
                       "[custom] How the state is sampled: $fstrobe (any simulator) or an SV "
                       "program block.", '"strobe"'),
    },
    "fault_model": {
        "pulse_widths_ps": _k("ints", [20, 40, 60, 80, 100, 120, 140, 160, 180],
                              "SET pulse widths injected at every site (ps, > 0)."),
        "inject_ff_d_nets": _k("bool", True, "Also inject on nets that drive flip-flop data pins."),
        "sites": _k("strs", [], "Inject only on these nets, named as in the netlist with `/` "
                                "between hierarchy levels (default: every site)."),
    },
    "propagation": {
        "delay_choice": _k("choice:max|min", "max",
                           "When several inputs of a gate change at the same instant and each "
                           "explains the output change, use the largest or smallest arc delay."),
        "electrical_masking": _k("bool", True,
                                 "Inertial filtering: an output transition is cancelled when its "
                                 "cause reverts before the gate delay has elapsed."),
        "em_margin_ps": _k("int", 0, "Added to the gate delay when deciding whether a pulse is "
                                     "filtered (ps); positive filters more."),
        "sdf_rail": _k("choice:late|early", "late",
                       "Which value of SDF (min:typ:max) triples to use: late = max, early = min."),
        "interconnect": _k("choice:ignore|transport", "ignore",
                           "SDF INTERCONNECT delays: ignore, or add them as pure delays."),
        "horizon_factor": _k("float", 1.25, "Simulate each SET for this many clock periods "
                                            "(>= 1)."),
    },
    "analyze": {
        "width_models": _k("tables", [{"type": "uniform"}],
                           "Pulse-width distributions; each is analysed into its own "
                           "directory. A distribution gives each injected width a "
                           "probability: {type = \"uniform\"}, {type = \"gaussian\", mean_ps "
                           "= 90, sd_ps = 40}, {type = \"exponential\", tau_ps = 40}, or {type "
                           "= \"table\", weights = {\"20\" = 1.0, \"40\" = 0.5, ...}} naming "
                           "every injected width. Weights are normalised to sum to 1."),
        "phase_domain": _k("choice:half_open|closed", "half_open",
                           "SET start times: [0, T), or [0, T] including the capture instant."),
        "write_trials": _k("bool", False, "Also write the exposure of every trial "
                                          "(trial_exposure.npz)."),
    },
    "inject": {
        "jobs": _k("int", 1, "Parallel processes (does not change the results)."),
        "cycles_per_shard": _k("int", 10, "Cycles per record file; bounds memory (does not "
                                          "change the results)."),
        "phase_grid_steps": _k("int", 0,
                               "If > 0, also write inject/phase_grid_captures.csv: for start "
                               "times k*T/steps (k = 0..steps), the flip-flops that capture."),
    },
    "observe": {
        "data_pins": _k("strs", [], "Observe only flip-flops whose data pin has one of these "
                                    "names (default: library.d_pins). Clock gates, whose "
                                    "\"data\" pin is an enable, are left out this way."),
        "exclude_cells": _k("strs", [], "Regular expressions (whole cell name) of flip-flop cells "
                                        "not to observe, e.g. clock-gating cells."),
        "exclude_instances": _k("strs", [], "Flip-flop instances not to observe: substrings of "
                                            "their names."),
        "allow_missing_timing_checks": _k("bool", False,
                                          "Leave flip-flops without SDF setup/hold checks "
                                          "unobserved instead of failing."),
    },
    "build": {
        "self_check": _k("bool", True, "Verify a sample of fan-out cones by direct traversal."),
        "self_check_samples": _k("int", 50, "Number of cones verified."),
        "self_check_seed": _k("int", 1, "Seed of that sample."),
        "strict_sdf_crosscheck": _k("bool", True, "Fail when netlist and SDF instances differ."),
        "max_undriven_nets": _k("int", 0, "Nets read by a cell but driven by nothing, tolerated."),
        "keep_intermediates": _k("bool", False, "Keep intermediate files in build/intermediate/."),
        "allow_missing_delays": _k("bool", False,
                                   "Accept gate arcs without an SDF delay inside injection cones "
                                   "(pulses cannot pass them) instead of failing."),
    },
}

EXTRA_SECTIONS: Dict[str, Dict[str, Key]] = {}

# Section order of the template: what every user sets first, advanced knobs last.
ORDER = ["design", "output", "library", "workload", "simulator", "fault_model", "observe",
         "propagation", "inject", "analyze", "build"]


def schema() -> Dict[str, Dict[str, Key]]:
    s = dict(SCHEMA)
    s.update(EXTRA_SECTIONS)
    return {k: s[k] for k in ORDER + [k for k in s if k not in ORDER]}


def _check_type(section: str, name: str, key: Key, v: Any, base_dir: str) -> Any:
    where = f"[{section}] {name}"
    t = key.type
    if t == "str":
        if not isinstance(v, str):
            raise ConfigError(f"{where} must be a string")
        return v
    if t == "path":
        if not isinstance(v, str) or not v:
            raise ConfigError(f"{where} must be a non-empty path")
        return os.path.normpath(os.path.join(base_dir, os.path.expanduser(v)))
    if t == "paths":
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            raise ConfigError(f"{where} must be a list of paths")
        return [os.path.normpath(os.path.join(base_dir, os.path.expanduser(x))) for x in v]
    if t == "int":
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(f"{where} must be an integer")
        return v
    if t == "float":
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(v)
    if t == "bool":
        if not isinstance(v, bool):
            raise ConfigError(f"{where} must be true or false")
        return v
    if t == "ints":
        if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            raise ConfigError(f"{where} must be a list of integers")
        return list(v)
    if t == "strs":
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise ConfigError(f"{where} must be a list of strings")
        return list(v)
    if t.startswith("choice:"):
        choices = t.split(":", 1)[1].split("|")
        if v not in choices:
            raise ConfigError(f"{where} must be one of {choices}, got {v!r}")
        return v
    if t == "table":
        if not isinstance(v, dict):
            raise ConfigError(f"{where} must be a table")
        return copy.deepcopy(v)
    if t == "dirs":
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            raise ConfigError(f"{where} must be a list of directories")
        return [os.path.normpath(os.path.join(base_dir, os.path.expanduser(x))) for x in v]
    if t == "tables":
        if not isinstance(v, list) or not all(isinstance(x, dict) for x in v):
            raise ConfigError(f"{where} must be a list of tables")
        return copy.deepcopy(v)
    raise AssertionError(t)


def resolve(raw: Dict[str, Any], base_dir: str, sections: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Validate a parsed config and fill defaults.  ``sections`` limits which
    sections must be complete (a stage only needs its own)."""
    sch = schema()
    unknown = sorted(set(raw) - set(sch))
    if unknown:
        raise ConfigError(f"unknown section(s) {unknown}; known: {sorted(sch)}")
    out: Dict[str, Dict[str, Any]] = {}
    for sec, keys in sch.items():
        given = raw.get(sec, {})
        if not isinstance(given, dict):
            raise ConfigError(f"[{sec}] must be a table")
        bad = sorted(set(given) - set(keys))
        if bad:
            raise ConfigError(f"[{sec}] unknown key(s) {bad}; known: {sorted(keys)}")
        res: Dict[str, Any] = {}
        for name, key in keys.items():
            if name in given:
                res[name] = _check_type(sec, name, key, given[name], base_dir)
            elif key.default is REQUIRED:
                if sections is None or sec in sections:
                    raise ConfigError(f"[{sec}] {name} is required: {key.doc}")
                res[name] = None
            else:
                res[name] = copy.deepcopy(key.default)
        out[sec] = res
    return out


def load(path: str, sections: Optional[List[str]] = None,
         overrides: Optional[Dict[str, Dict[str, Any]]] = None,
         check_files: bool = True, check_width_models: bool = True) -> Dict[str, Dict[str, Any]]:
    """Read, validate and complete a configuration file."""
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"{path}: {e}") from e
    for sec, kv in (overrides or {}).items():
        raw.setdefault(sec, {}).update(kv)
    base = os.path.dirname(os.path.abspath(path))
    cfg = resolve(raw, base, sections)
    for sub in cfg["workload"].get("macro_rtl_substitutions") or []:
        if isinstance(sub.get("rtl_file"), str):
            sub["rtl_file"] = os.path.normpath(os.path.join(base, os.path.expanduser(sub["rtl_file"])))
    if sections is None:
        validate(cfg, check_files=check_files, check_width_models=check_width_models)
    return cfg


INPUT_FILES = [("design", "netlist"), ("design", "sdf"), ("library", "cell_functions"),
               ("workload", "testbench"), ("workload", "cell_sim_models"),
               ("workload", "data_files"), ("workload", "cycle_window_csv"),
               ("workload", "rtl_files"), ("workload", "netlist_substitute_files"),
               ("simulator", "container")]

# Keys of the entries of each list-of-tables option: {key: (type, required)}.
TABLE_KEYS = {
    ("library", "scan_data_pin_pairs"): {"data": (str, True), "scan": (str, True)},
    ("workload", "tb_substitutions"): {"regex": (str, True), "replacement": (str, True)},
    ("workload", "macro_rtl_substitutions"): {"macro_module": (str, True), "rtl_file": (str, True)},
}


def _check_tables(cfg: Dict[str, Dict[str, Any]], check_files: bool) -> None:
    import re
    for (sec, key), spec in TABLE_KEYS.items():
        for n, entry in enumerate(cfg[sec][key]):
            where = f"[{sec}] {key}, entry {n + 1}"
            bad = sorted(set(entry) - set(spec))
            missing = sorted(k for k, (_, req) in spec.items() if req and k not in entry)
            if bad or missing:
                raise ConfigError(f"{where}: needs the keys {sorted(spec)}"
                                  + (f"; unknown {bad}" if bad else "")
                                  + (f"; missing {missing}" if missing else ""))
            for k, (t, _) in spec.items():
                if k in entry and not isinstance(entry[k], t):
                    raise ConfigError(f"{where}: {k} must be a string")
            if key == "tb_substitutions":
                try:
                    re.compile(entry["regex"])
                except re.error as e:
                    raise ConfigError(f"{where}: bad regex {entry['regex']!r}: {e}") from None


def missing_files(cfg: Dict[str, Dict[str, Any]]) -> List[str]:
    """Input files and directories of the configuration that do not exist."""
    out = []
    for sec, key in INPUT_FILES:
        v = cfg[sec][key]
        for p in ([v] if isinstance(v, str) else (v or [])):
            if not os.path.isfile(p):
                out.append(f"[{sec}] {key}: {p}")
    for d in cfg["workload"]["include_dirs"]:
        if not os.path.isdir(d):
            out.append(f"[workload] include_dirs: {d}")
    for n, e in enumerate(cfg["workload"]["macro_rtl_substitutions"]):
        if isinstance(e.get("rtl_file"), str) and not os.path.isfile(e["rtl_file"]):
            out.append(f"[workload] macro_rtl_substitutions, entry {n + 1}: {e['rtl_file']}")
    return out


def validate(cfg: Dict[str, Dict[str, Any]], check_files: bool = True,
             check_width_models: bool = True) -> None:
    """Checks that involve values, not only types: caught here rather than in a late stage."""
    if check_files:
        missing = missing_files(cfg)
        if missing:
            raise ConfigError("not found:\n  " + "\n  ".join(missing))
    _check_tables(cfg, check_files)
    widths = cfg["fault_model"]["pulse_widths_ps"]
    if not widths or any(w <= 0 for w in widths) or len(set(widths)) != len(widths):
        raise ConfigError("[fault_model] pulse_widths_ps must be distinct positive integers")
    models = cfg["analyze"]["width_models"]
    if not models:
        raise ConfigError("[analyze] width_models must name at least one distribution")
    from .analysis.widths import WidthModelError, width_weights
    for n, m in enumerate(models if check_width_models else []):
        try:
            width_weights(m, sorted(widths))
        except WidthModelError as e:
            raise ConfigError(f"[analyze] width_models, entry {n + 1}: {e}") from None
    names = [analysis_name(m, cfg["analyze"]["phase_domain"]) for m in models]
    if len(set(names)) != len(names):
        raise ConfigError("[analyze] width_models lists the same distribution twice")
    t_ps = cfg["design"]["clock_period_ns"] * 1000
    if t_ps < 1 or abs(t_ps - round(t_ps)) > 1e-6:
        raise ConfigError(f"[design] clock_period_ns = {cfg['design']['clock_period_ns']!r}: "
                          f"must be a whole number of ps, at least 0.001 ns")
    for sec, key, lo in (("workload", "n_cycles", 1), ("inject", "jobs", 1),
                         ("inject", "cycles_per_shard", 1), ("inject", "phase_grid_steps", 0),
                         ("workload", "padding_cycles", 0), ("workload", "warmup_cycles", 0),
                         ("workload", "reserve_factor", 0)):
        if cfg[sec][key] < lo:
            raise ConfigError(f"[{sec}] {key} must be >= {lo}")
    if cfg["propagation"]["horizon_factor"] < 1:
        raise ConfigError("[propagation] horizon_factor must be >= 1")
    for key in ("tb_top", "dut_instance", "clock"):
        if not cfg["workload"][key]:
            raise ConfigError(f"[workload] {key} must not be empty")
    from .pipeline import simulator_dict
    simulator_dict(cfg)          # raises on keys that do not fit the preset


def analysis_name(width_model: Dict[str, Any], phase_domain: str = "half_open") -> str:
    """Directory name of an analysis, e.g. gaussian_mean90_sd40 (with a ``_closed``
    suffix for the closed phase domain)."""
    kind = str(width_model.get("type", "uniform"))
    if kind == "table":
        blob = json.dumps(width_model.get("weights", {}), sort_keys=True)
        base = "table_" + hashlib.sha256(blob.encode()).hexdigest()[:8]
    else:
        parts = [kind]
        for k in sorted(width_model):
            if k == "type":
                continue
            v = width_model[k]
            parts.append(f"{k.replace('_ps', '')}{v:g}" if isinstance(v, (int, float))
                         else f"{k}{v}")
        base = "_".join(parts)
    return base + ("_closed" if phase_domain == "closed" else "")


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v))


def _wrap(text: str, width: int = 88) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width=width - 2, break_on_hyphens=False) or [""]


def template() -> str:
    """A complete, commented configuration with every default spelled out."""
    lines = ["# setfi configuration.",
             "# Relative paths are relative to this file. Keys marked REQUIRED have no default;",
             "# every other key may be deleted to use its default. \"Used by\" names the stages",
             "# that read the keys of a section (a key that differs says so): changing a key",
             "# re-runs them.", ""]
    from collections import Counter
    for sec, keys in schema().items():
        users = {k: used_by(sec, k) for k in keys}
        # the section header names the usual stages; a key that differs says so
        common = list(Counter(tuple(u) for u in users.values()).most_common(1)[0][0])
        lines.append("# " + "-" * 86)
        lines.append(f"# {SECTIONS.get(sec, '')} Used by: {', '.join(common) or 'all stages'}.")
        lines.append(f"[{sec}]")
        for name, key in keys.items():
            doc = key.doc
            if key.type.startswith("choice:"):
                doc += " One of: " + " | ".join(key.type.split(":", 1)[1].split("|")) + "."
            if users[name] != common and users[name]:
                doc += f" Used by: {', '.join(users[name])}."
            lines.extend("# " + l for l in _wrap(doc))
            if key.default is REQUIRED:
                lines.append(f"{name} = {_toml_value(key.example)}   # REQUIRED")
            elif key.default is None:
                ex = key.example if key.example is not None else '""'
                lines.append(f"# {name} = {ex}")
            elif key.type == "tables" and key.default:
                lines.append(f"{name} = [")
                lines.extend(f"  {_toml_value(x)}," for x in key.default)
                lines.append("]")
            else:
                lines.append(f"{name} = {_toml_value(key.default)}")
        lines.append("")
    return "\n".join(lines)


def dump_resolved(cfg: Dict[str, Dict[str, Any]]) -> str:
    """The configuration as used: defaults filled in, including the simulator preset's."""
    from .pipeline import simulator_defaults
    out = []
    for sec, kv in cfg.items():
        kv = dict(kv)
        if sec == "simulator":
            for k, v in simulator_defaults(cfg).items():
                if kv.get(k) is None:
                    kv[k] = v
        out.append(f"[{sec}]\n" + "\n".join(f"{k} = {_toml_value(v)}" for k, v in kv.items()
                                              if v is not None) + "\n")
    return "\n".join(out)
