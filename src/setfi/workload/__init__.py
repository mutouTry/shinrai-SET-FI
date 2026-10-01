"""Workload recording: run the user's testbench against the gate-level netlist
and record the settled state of every net at sampled cycles.

    from setfi.workload import WorkloadSpec, vcs_preset, record_workload
    spec = WorkloadSpec(netlist=..., top_module=..., net_index=..., testbench=[...],
                        cell_models=[...], output_dir=..., simulator=vcs_preset(...))
    result = record_workload(spec)
    # <output_dir>/states/cycle_<N>.hex and <output_dir>/cycles.json

See :mod:`setfi.workload.record` for the flow and output layout,
:mod:`setfi.workload.sampling` for the sampling, unknown-value check and hex format,
:mod:`setfi.workload.tbgen` for the recorder semantics and
:mod:`setfi.workload.spec` for command templates.
"""
from .adapt_tb import AdaptResult, adapt_testbench
from .errors import ERROR_KINDS, WorkloadError
from .record import record_workload
from .sampling import (DEFAULT_MAX_X_FRAC, DEFAULT_RESERVE_FACTOR, format_state_hex,
                       load_cycle_window, parse_state_hex, sample_cycles,
                       sample_cycles_in_window, screen_cycles, screen_cycles_by_frac,
                       x_frac_of_text)
from .spec import (PRESETS, SimulatorSpec, WorkloadSpec, icarus_preset, render_command,
                   simulator_from_dict, vcs_preset)

__all__ = [
    "AdaptResult", "adapt_testbench",
    "ERROR_KINDS", "WorkloadError",
    "record_workload",
    "DEFAULT_MAX_X_FRAC", "DEFAULT_RESERVE_FACTOR", "format_state_hex", "load_cycle_window",
    "parse_state_hex", "sample_cycles", "sample_cycles_in_window", "screen_cycles",
    "screen_cycles_by_frac", "x_frac_of_text",
    "PRESETS", "SimulatorSpec", "WorkloadSpec", "icarus_preset", "render_command",
    "simulator_from_dict", "vcs_preset",
]
