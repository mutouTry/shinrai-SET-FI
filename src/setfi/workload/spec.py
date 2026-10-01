"""Workload and simulator specifications.

A :class:`SimulatorSpec` says how to compile and run a simulation as command
templates.  Templates are token lists; placeholders:

    {work_dir}    simulator working directory (absolute)
    {output_dir}  the recording's output directory (absolute)
    {sim_bin}     compiled simulation executable / image (absolute)
    {filelist}    source file list, one path per line (absolute)
    {top}         top-level module name (the recorder wrapper)
    {defines}     list: each of WorkloadSpec.defines rendered with define_format
    {extra_args}  list: SimulatorSpec.extra_args (compile time)
    {run_args}    list: SimulatorSpec.run_args (run time)
    {plusargs}    list: recorder plusargs (run time)
    {bind_args}   list: bind_flag, bind, bind_flag, bind, ... (wrapper only)

A token that is exactly ``{name}`` of a list placeholder expands into zero or
more tokens; any other token is ``str.format``-ed with the scalar placeholders
(``{{`` / ``}}`` for literal braces).  ``wrapper`` (e.g. a container ``exec``
prefix) is prepended to both the compile and the run command.

Nothing site-specific is built in: container images, binds and binaries come
from the spec (or the preset's arguments).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .errors import WorkloadError
from .sampling import DEFAULT_MAX_X_FRAC, DEFAULT_RESERVE_FACTOR
from .tbgen import RECORDER_STYLES

LIST_PLACEHOLDERS = ("defines", "extra_args", "run_args", "plusargs", "bind_args")
SCALAR_PLACEHOLDERS = ("work_dir", "output_dir", "sim_bin", "filelist", "top")


def _render_token(tok: str, scalars: Mapping[str, str], lists: Mapping[str, Sequence[str]]) -> List[str]:
    stripped = tok.strip()
    if stripped.startswith("{") and stripped.endswith("}") and stripped[1:-1] in lists:
        return [str(x) for x in lists[stripped[1:-1]]]
    for name in lists:
        if "{" + name + "}" in tok:
            raise WorkloadError("config_invalid",
                                f"list placeholder {{{name}}} must be a whole token, got {tok!r}")
    try:
        return [tok.format_map(dict(scalars))]
    except KeyError as e:
        raise WorkloadError("config_invalid",
                            f"unknown placeholder {e} in command token {tok!r}") from None
    except (ValueError, IndexError) as e:
        raise WorkloadError("config_invalid", f"bad command token {tok!r}: {e}") from None


def render_command(template: Sequence[str], scalars: Mapping[str, str],
                   lists: Mapping[str, Sequence[str]]) -> List[str]:
    """Render a token-list template (see module docstring)."""
    out: List[str] = []
    for tok in template:
        out.extend(_render_token(str(tok), scalars, lists))
    return out


@dataclass
class SimulatorSpec:
    """How to compile and run the recorder simulation."""
    name: str
    compile: List[str]                       # compile command template
    run: List[str]                           # run command template
    wrapper: List[str] = field(default_factory=list)   # prefix for both, e.g. container exec
    binds: List[str] = field(default_factory=list)     # expanded into {bind_args}
    bind_flag: str = "-B"
    define_format: str = "+define+{define}"
    extra_args: List[str] = field(default_factory=list)
    run_args: List[str] = field(default_factory=list)
    recorder: str = "program"                # "program" | "strobe" (see tbgen)
    # Written as a first source file (`timescale <value>) when the simulator has
    # no command-line default timescale; None = no preamble.
    timescale_preamble: Optional[str] = None
    env_unset: List[str] = field(default_factory=list)   # removed from the environment
    required_files: List[str] = field(default_factory=list)   # e.g. the container image

    def __post_init__(self) -> None:
        if self.recorder not in RECORDER_STYLES:
            raise WorkloadError("config_invalid",
                                f"simulator.recorder must be one of {RECORDER_STYLES}, got {self.recorder!r}")
        for attr in ("compile", "run"):
            if not getattr(self, attr):
                raise WorkloadError("config_invalid", f"simulator.{attr} template is empty")

    def commands(self, *, scalars: Mapping[str, str], defines: Sequence[str],
                 plusargs: Sequence[str]) -> Dict[str, List[str]]:
        """Render the full compile and run commands."""
        bind_scalars = dict(scalars)
        bind_args: List[str] = []
        for b in self.binds:
            bind_args.extend([self.bind_flag, *(_render_token(b, bind_scalars, {}))])
        lists = {
            "defines": [self.define_format.format(define=d) for d in defines],
            "extra_args": [_render_token(a, scalars, {})[0] for a in self.extra_args],
            "run_args": [_render_token(a, scalars, {})[0] for a in self.run_args],
            "plusargs": list(plusargs),
            "bind_args": bind_args,
        }
        prefix = render_command(self.wrapper, scalars, lists)
        return {
            "compile": prefix + render_command(self.compile, scalars, lists),
            "run": prefix + render_command(self.run, scalars, lists),
        }

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
def vcs_preset(
    *,
    container: Optional[str] = None,
    container_runtime: str = "singularity",
    container_exec_args: Sequence[str] = ("exec", "--cleanenv"),
    binds: Sequence[str] = (),
    bind_output_dir: bool = True,
    vcs_binary: str = "vcs",
    extra_args: Sequence[str] = (),
    run_args: Sequence[str] = (),
    env_unset: Sequence[str] = (),
) -> SimulatorSpec:
    """Synopsys VCS, zero-delay functional simulation.

    compile: ``vcs <extra_args> -sverilog -debug_access -timescale=1ns/1ps
    <+define+X...> +notimingchecks +nospecify -o <sim_bin> -f <filelist>``;
    run: ``<sim_bin> <run_args> <plusargs>``.  With ``container`` set, both run as
    ``<runtime> exec --cleanenv -B <bind>... -B <output_dir>:<output_dir> <image> ...``.
    No ``-top``: VCS elaborates every uninstantiated module (the recorder
    wrapper and program).
    """
    wrapper: List[str] = []
    bind_list: List[str] = []
    required: List[str] = []
    if container:
        wrapper = [container_runtime, *container_exec_args, "{bind_args}", container]
        bind_list = list(binds) + (["{output_dir}:{output_dir}"] if bind_output_dir else [])
        required = [container]
    return SimulatorSpec(
        name="vcs",
        wrapper=wrapper,
        binds=bind_list,
        bind_flag="-B",
        compile=[vcs_binary, "{extra_args}", "-sverilog", "-debug_access", "-timescale=1ns/1ps",
                 "{defines}", "+notimingchecks", "+nospecify",
                 "-o", "{sim_bin}", "-f", "{filelist}"],
        run=["{sim_bin}", "{run_args}", "{plusargs}"],
        define_format="+define+{define}",
        extra_args=list(extra_args),
        run_args=list(run_args),
        recorder="program",
        env_unset=list(env_unset),
        required_files=required,
    )


def icarus_preset(
    *,
    iverilog: str = "iverilog",
    vvp: str = "vvp",
    generation: str = "-g2012",
    extra_args: Sequence[str] = (),
    run_args: Sequence[str] = (),
    timescale: Optional[str] = "1ns/1ps",
    env_unset: Sequence[str] = (),
) -> SimulatorSpec:
    """Icarus Verilog (>= 12), zero-delay functional simulation.

    Icarus ignores specify blocks and timing checks unless ``-gspecify`` is
    given, which is the zero-delay behaviour VCS gets from
    ``+nospecify +notimingchecks``.  ``-s <top>`` is required because every
    unreferenced cell-library module would otherwise become a root.  The
    recorder uses the "strobe" style (Icarus runs ``program`` blocks in the
    Active region and has no ``$system``).
    """
    return SimulatorSpec(
        name="icarus",
        compile=[iverilog, generation, "{extra_args}", "{defines}", "-s", "{top}",
                 "-o", "{sim_bin}", "-c", "{filelist}"],
        run=[vvp, "-n", "{sim_bin}", "{run_args}", "{plusargs}"],
        define_format="-D{define}",
        extra_args=list(extra_args),
        run_args=list(run_args),
        recorder="strobe",
        timescale_preamble=timescale,
        env_unset=list(env_unset),
    )


PRESETS = {"vcs": vcs_preset, "icarus": icarus_preset}


def simulator_from_dict(d: Mapping[str, Any]) -> SimulatorSpec:
    """Build a SimulatorSpec from config data.

    ``{"preset": "vcs", <preset kwargs>...}`` calls the preset; ``overrides``
    (optional) then replaces SimulatorSpec fields.  Without ``preset`` the dict
    holds SimulatorSpec fields directly.
    """
    d = dict(d)
    preset = d.pop("preset", None)
    overrides = dict(d.pop("overrides", {}) or {})
    if preset is None:
        try:
            return SimulatorSpec(**d, **overrides)
        except TypeError as e:
            raise WorkloadError("config_invalid", f"simulator: {e}") from None
    if preset not in PRESETS:
        raise WorkloadError("config_invalid",
                            f"unknown simulator preset {preset!r}; known: {sorted(PRESETS)}")
    try:
        spec = PRESETS[preset](**d)
    except TypeError as e:
        raise WorkloadError("config_invalid", f"simulator preset {preset!r}: {e}") from None
    if overrides:
        try:
            spec = dataclasses.replace(spec, **overrides)
        except TypeError as e:
            raise WorkloadError("config_invalid", f"simulator.overrides: {e}") from None
    return spec


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
def _path(v) -> Path:
    return Path(v).expanduser()


def _paths(v) -> List[Path]:
    if v is None:
        return []
    if isinstance(v, (str, Path)):
        return [_path(v)]
    return [_path(x) for x in v]


@dataclass
class WorkloadSpec:
    """Everything needed to record the per-cycle state of one workload.

    The testbench runs unchanged against the gate-level netlist (zero-delay);
    ``tb_dut_inst`` is the DUT's instance name inside ``tb_top_module``.
    """
    netlist: Path                   # gate-level netlist (.v)
    top_module: str                 # DUT top module in the netlist
    net_index: Path                 # net numbering: {"idx_to_net": {"0": name, ...}}
    testbench: List[Path]           # testbench source file(s), compiled after the netlist
    cell_models: List[Path]         # simulation models of the library cells
    output_dir: Path
    simulator: SimulatorSpec
    tb_top_module: str = "testbench"
    tb_dut_inst: str = "dut"
    ff_index: Optional[Path] = None      # flip-flop D/Q nets: a tap that cannot be read is fatal
    rtl_dut_files: List[Path] = field(default_factory=list)   # only for auto-derived substitutions
    tb_data_files: List[Path] = field(default_factory=list)   # copied into the simulator's cwd
    clock_port: str = "clk"              # clock net name inside the testbench
    reset_port: str = "rst_n"            # reset net name inside the testbench
    reset_polarity: str = "active_low"   # "active_low" | "active_high"
    defines: List[str] = field(default_factory=list)
    # Testbench adaptation (see adapt_tb): regex rewrites for XMRs / module
    # names the netlist changed, optional auto-derivation, black-box macros.
    hier_substitutions: List[Dict[str, str]] = field(default_factory=list)
    auto_derive_hier_substitutions: bool = False
    macro_rtl_subs: List[Dict[str, str]] = field(default_factory=list)
    netlist_strip_modules: List[str] = field(default_factory=list)
    netlist_substitute_files: List[Path] = field(default_factory=list)
    # Sampling
    n_sample_cycles: int = 30
    sample_seed: int = 12345
    padding_cycles: int = 5
    warmup_cycles: int = 0
    cycle_window_csv: Optional[Path] = None
    max_x_frac: float = DEFAULT_MAX_X_FRAC
    reserve_factor: int = DEFAULT_RESERVE_FACTOR
    # Run control
    expected_pass_string: str = "RESULT: PASS"
    compile_timeout_s: int = 600
    sim_timeout_s: int = 600

    def __post_init__(self) -> None:
        self.netlist = _path(self.netlist)
        self.net_index = _path(self.net_index)
        self.testbench = _paths(self.testbench)
        self.cell_models = _paths(self.cell_models)
        self.output_dir = _path(self.output_dir)
        self.ff_index = None if self.ff_index is None else _path(self.ff_index)
        self.rtl_dut_files = _paths(self.rtl_dut_files)
        self.tb_data_files = _paths(self.tb_data_files)
        self.netlist_substitute_files = _paths(self.netlist_substitute_files)
        self.cycle_window_csv = None if not self.cycle_window_csv else _path(self.cycle_window_csv)
        self.defines = [str(d) for d in (self.defines or [])]
        if isinstance(self.simulator, _MappingABC):
            self.simulator = simulator_from_dict(self.simulator)
        if not self.testbench:
            raise WorkloadError("config_invalid", "workload.testbench is empty")
        if not self.cell_models:
            raise WorkloadError("config_invalid", "workload.cell_sim_models is empty")
        if self.reset_polarity not in ("active_low", "active_high"):
            raise WorkloadError("config_invalid",
                                f"reset_polarity must be 'active_low' or 'active_high', "
                                f"got {self.reset_polarity!r}")
        for k in ("n_sample_cycles", "padding_cycles", "warmup_cycles", "reserve_factor"):
            if int(getattr(self, k)) < 0:
                raise WorkloadError("config_invalid", f"{k} must be >= 0")
        if int(self.n_sample_cycles) < 1:
            raise WorkloadError("config_invalid", "workload.n_cycles must be >= 1")
        if not (0.0 <= float(self.max_x_frac) <= 1.0):
            raise WorkloadError("config_invalid", "max_x_frac must be within [0, 1]")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "WorkloadSpec":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in d if k not in names and not str(k).startswith("_"))
        if unknown:
            raise WorkloadError("config_invalid",
                                f"unknown workload key(s): {unknown}; allowed: {sorted(names)}")
        try:
            return cls(**{k: v for k, v in d.items() if k in names})
        except TypeError as e:
            raise WorkloadError("config_invalid", f"workload: {e}") from None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, SimulatorSpec):
                v = v.to_dict()
            elif isinstance(v, Path):
                v = str(v)
            elif isinstance(v, list):
                v = [str(x) if isinstance(x, Path) else x for x in v]
            out[f.name] = v
        return out
