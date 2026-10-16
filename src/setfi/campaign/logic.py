"""Logic-only translation: the flip-flops a SET upsets, without timing.

For each sampled cycle c and injection site v, v is inverted in the state
recorded for c and the gates of v's fan-out cone whose inputs change are
re-evaluated in topological order; every other net keeps its recorded value.
F is the set of observed flip-flops whose data pin changes.  There are no
delays, no inertial filtering and no capture window: every changed data pin
counts as captured.

The result is a pattern table (setfi.analysis.patterns.Patterns) with one run
per (c, v) whose F is not empty.  It has no start-time grid and no pulse
widths: period_ps = steps = 0, starts_ps = [0], pulse_widths_ps is empty, and
every run has width 0 and k_lo = k_hi = 0.  The multiplicity of a pattern
(c, F) is the number of sites that upset exactly F in cycle c.  The sites,
cycles and observed flip-flops are those the inject stage uses, so the table
compares directly with the one of the patterns stage.

The (cycle, site) pairs run in C++ (_engine.logic_block, `jobs` threads) on
the cones of setfi.campaign.fastpath.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .. import _engine
from ..analysis.patterns import Patterns
from ..records import INF_PS
from .fastpath import compile_sites
from .run import select_sites
from .state import read_cycle_bits
from .substrate import load_substrate

CYCLES_PER_BLOCK = 16          # recorded states held in memory at a time


@dataclass
class LogicSpec:
    stage_dir: str                    # build output
    state_dir: str                    # cycle_<N>.hex files
    cycles: List[int]                 # sampled cycles
    clock_pins: List[str] = field(default_factory=lambda: ["CP", "CK", "CLK"])
    allow_missing_timing_checks: bool = False
    sites: Optional[List[str]] = None         # subset of injection sites (default: all)
    jobs: int = 1


def run_logic(spec: LogicSpec, log=print) -> Patterns:
    """The logic-only pattern table of the sampled cycles and injection sites."""
    t0 = time.time()
    jobs = max(1, int(spec.jobs))
    # interconnect delays do not matter here: the cones are compiled without wires
    sub = load_substrate(spec.stage_dir, clock_pins=spec.clock_pins,
                         allow_missing_timing_checks=spec.allow_missing_timing_checks, n_threads=jobs)
    timing = {"load": time.time() - t0}
    sites = select_sites(sub, spec.sites, spec.stage_dir)
    cycles = [int(c) for c in spec.cycles]
    ff_ids = sorted(sub.ffs)
    log(f"[logic] {len(cycles)} cycles x {len(sites)} sites = {len(cycles) * len(sites):,} SETs; "
        f"{len(ff_ids)} observed FFs")

    # The table builder of the patterns stage, with one SET start time (0) and
    # capture thresholds that every record meets there: it groups the (cycle,
    # site) pairs by the set of flip-flops they upset, one run per pair.
    n_ids = ff_ids[-1] + 1 if ff_ids else 0
    builder = _engine.PatternBuilder(np.zeros(1, np.int64), np.ones((2, n_ids), np.int64),
                                     np.zeros((2, n_ids), np.int64), np.asarray(cycles, dtype=np.int64),
                                     len(sites), np.zeros(1, np.int64), INF_PS)
    if cycles:
        t = time.time()
        cones = compile_sites(sub, sites, n_threads=jobs)
        timing["compile"] = time.time() - t
        t = time.time()
        site_ids = np.arange(len(sites), dtype=np.int32)
        n_nets = len(sub.idx_to_net)
        for i in range(0, len(cycles), CYCLES_PER_BLOCK):
            block = cycles[i:i + CYCLES_PER_BLOCK]
            bits = np.stack([read_cycle_bits(spec.state_dir, c, n_nets) for c in block])
            out = _engine.logic_block(cones, bits, np.asarray(block, dtype=np.int64), site_ids, jobs)
            n = out["ff"].size
            builder.add({"cycle": out["cycle"], "site": out["site"], "width_ps": np.zeros(n, np.int32),
                         "ff": out["ff"], "base": np.zeros(n, np.uint8), "t_enter": np.zeros(n, np.int64),
                         "t_exit": np.full(n, INF_PS, np.int64)})
        timing["sets"] = time.time() - t
    pt = Patterns(period_ps=0, steps=0, starts_ps=np.zeros(1, np.int64),
                  cycles=np.asarray(cycles, dtype=np.int64), pulse_widths_ps=np.zeros(0, np.int64),
                  sites=np.asarray(sites, dtype=str), ff_ids=np.asarray(ff_ids, dtype=np.int32),
                  ff_instances=np.asarray([sub.ffs[f].inst for f in ff_ids], dtype=str), **builder.result())
    log("[logic] time: " + ", ".join(f"{k} {v:.1f}s" for k, v in timing.items()))
    log(f"[logic] done in {time.time() - t0:.1f}s: {pt.count:,} patterns from "
        f"{int(pt.pattern_multiplicity.sum()):,} SETs that upset flip-flops")
    return pt
