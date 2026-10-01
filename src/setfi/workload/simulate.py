"""Run a compile + simulate pass and parse the recorder's markers."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .errors import WorkloadError
from .spec import SimulatorSpec


@dataclass
class SimRun:
    sim_bin: Path
    compile_cmd: List[str]
    run_cmd: List[str]
    compile_rc: int
    compile_log: str
    sim_rc: int = -1          # -1: not run (compile failed)
    sim_log: str = ""


def clean_env(sim: SimulatorSpec, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment for simulator subprocesses, without ``env_unset``."""
    env = dict(os.environ if base is None else base)
    for k in sim.env_unset:
        env.pop(k, None)
    return env


def _resolve_executable(cmd: List[str], env: Mapping[str, str]) -> List[str]:
    exe = cmd[0]
    if os.sep in exe:
        if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
            raise WorkloadError("tool_unavailable", f"executable not found: {exe}")
        return cmd
    found = shutil.which(exe, path=env.get("PATH"))
    if found is None:
        raise WorkloadError("tool_unavailable", f"{exe!r} is not on PATH")
    return [found] + cmd[1:]


def check_simulator(sim: SimulatorSpec) -> None:
    """Fail early (tool_unavailable) on a missing required file."""
    for f in sim.required_files:
        if not Path(f).is_file():
            raise WorkloadError("tool_unavailable", f"simulator {sim.name}: required file not found: {f}")


def run_pass(
    sim: SimulatorSpec, *, work_dir: Path, output_dir: Path, filelist: Path, top: str,
    defines: Sequence[str], plusargs: Sequence[str], compile_timeout_s: int,
    sim_timeout_s: int, log_prefix: str, log: Callable[[str], None],
) -> SimRun:
    """Compile ``filelist`` into ``<work_dir>/simv_<log_prefix>`` and run it
    with cwd = ``work_dir``.  Logs go to ``<log_prefix>_compile.log`` /
    ``<log_prefix>_sim.log`` in ``work_dir``."""
    sim_bin = work_dir / f"simv_{log_prefix}"
    scalars = {"work_dir": str(work_dir), "output_dir": str(output_dir),
               "sim_bin": str(sim_bin), "filelist": str(filelist), "top": top}
    cmds = sim.commands(scalars=scalars, defines=defines, plusargs=plusargs)
    env = clean_env(sim)
    compile_cmd = _resolve_executable(cmds["compile"], env)

    log(f"  [{log_prefix}] compile cmd: {' '.join(compile_cmd)}")
    try:
        cp = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=work_dir,
                            timeout=compile_timeout_s, env=env)
    except subprocess.TimeoutExpired as e:
        raise WorkloadError("sim_timeout",
                            f"{sim.name} compile timed out after {compile_timeout_s}s") from e
    compile_log = (cp.stdout or "") + (cp.stderr or "")
    (work_dir / f"{log_prefix}_compile.log").write_text(compile_log)
    run = SimRun(sim_bin=sim_bin, compile_cmd=compile_cmd, run_cmd=list(cmds["run"]),
                 compile_rc=cp.returncode, compile_log=compile_log)
    if cp.returncode != 0 or not sim_bin.is_file():
        if cp.returncode == 0:
            run.compile_rc = -1       # "succeeded" without producing the executable
        return run

    run_cmd = run.run_cmd = _resolve_executable(cmds["run"], env)
    log(f"  [{log_prefix}] run cmd: {' '.join(run_cmd)}")
    try:
        sp = subprocess.run(run_cmd, capture_output=True, text=True, cwd=work_dir,
                            timeout=sim_timeout_s, env=env)
    except subprocess.TimeoutExpired as e:
        raise WorkloadError("sim_timeout",
                            f"{sim.name} simulation timed out after {sim_timeout_s}s") from e
    run.sim_log = (sp.stdout or "") + (sp.stderr or "")
    run.sim_rc = sp.returncode
    (work_dir / f"{log_prefix}_sim.log").write_text(run.sim_log)
    return run


_FIRST_RE = re.compile(r"^\[SETFI-CYC\] first=(\d+)\s*$", re.MULTILINE)
_LAST_RE = re.compile(r"^\[SETFI-CYC\] last=(\d+)\s*$", re.MULTILINE)


def parse_cycle_markers(sim_log: str, sim_returncode: int) -> Dict[str, int]:
    """The cycle counter's values {"first": N, "last": M} at the first cycle after
    reset and at the end of the simulation, from a simulation log.

    Strict: a missing or repeated marker is an error, never inferred."""
    firsts = _FIRST_RE.findall(sim_log)
    lasts = _LAST_RE.findall(sim_log)
    if not firsts:
        raise WorkloadError(
            "sim_no_first_marker",
            f"simulation emitted no '[SETFI-CYC] first=N' marker; likely the reset never "
            f"deasserted or the wrapper was not elaborated. rc={sim_returncode}; "
            f"tail: {sim_log[-400:]!r}",
        )
    if not lasts:
        raise WorkloadError(
            "sim_aborted_no_last_marker",
            f"simulation has '[SETFI-CYC] first=...' but no '[SETFI-CYC] last=...' marker: the "
            f"`final` block did not run (crash, license timeout, $fatal, kill, or a hang "
            f"into the timeout). rc={sim_returncode}; tail: {sim_log[-400:]!r}",
        )
    if len(firsts) > 1:
        raise WorkloadError("marker_parse_ambiguous",
                            f"{len(firsts)} '[SETFI-CYC] first=...' markers; expected exactly 1")
    if len(lasts) > 1:
        raise WorkloadError("marker_parse_ambiguous",
                            f"{len(lasts)} '[SETFI-CYC] last=...' markers; expected exactly 1")
    return {"first": int(firsts[0]), "last": int(lasts[0])}
