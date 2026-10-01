"""Error type for the workload recorder."""
from __future__ import annotations

ERROR_KINDS = frozenset({
    "config_invalid",
    "tool_unavailable",
    "compile_failed",
    "sim_failed",
    "sim_timeout",
    "sim_aborted_no_last_marker",   # the wrapper's `final` block did not fire
    "sim_no_first_marker",          # reset never deasserted in pass 1
    "xmr_unresolved",               # a tap leaf name is absent from the netlist
    "no_cycles_in_range",           # too few cycles to sample (run / window / unknown-value check)
    "marker_parse_ambiguous",       # more than one first= / last= marker
    "unresolvable_state_tap",       # a flip-flop tap that cannot be read
    "recorder_semantics",           # simulator-specific recorder could not reproduce the contract
    "late_input",                   # the testbench changes design inputs after the falling edge
    "reset_never_released",         # the cycle counter never advanced
    # testbench adaptation (adapt_tb)
    "tb_unparseable",
    "tech_lib_missing",
    "macro_unhandled",
})


class WorkloadError(Exception):
    """A workload recording failure with a machine-readable ``kind``."""

    def __init__(self, kind: str, message: str) -> None:
        if kind not in ERROR_KINDS:
            raise AssertionError(f"unregistered error kind {kind!r}")
        super().__init__(message)
        self.kind = kind
        self.message = message
