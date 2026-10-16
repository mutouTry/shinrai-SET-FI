"""The pattern table: which flip-flops a SET upsets together, and where from.

Take the SET start times on a grid, delta_k = k*T/steps for k = 0..steps (T is
the capture edge, the clock period; ``phase_grid``).  For each trial (cycle,
site, pulse width) of a campaign and each k, the set F of flip-flops that
capture a wrong value follows from the pulse records alone: FF f is in F when
one of its records in the trial captures at delta_k (the capture rule of
``setfi.records``).

A pattern is a pair (cycle, F) with F not empty.  The table lists each pattern
once, with

    multiplicity  the number of grid points (site, width, k) of its cycle at
                  which the SET upsets exactly F
    n_sites       the number of sites among them
    runs          where it comes from: the maximal ranges k_lo..k_hi of grid
                  indices at which one trial (site, width) upsets exactly F;
                  their lengths add up to the multiplicity

Expanding the runs gives every grid point at which a SET upsets flip-flops
(``Patterns.grid_points``).

patterns.npz (format "setfi-patterns/1"; numpy arrays only, no pickles):

    format                str               "setfi-patterns/1"
    period_ps, steps      int64             T and the number of grid steps
    starts_ps             int64[steps + 1]  delta_k in ps, rounded half up
    cycles                int64[C]          the sampled cycles of the campaign
    pulse_widths_ps       int64[W]          its injected pulse widths
    sites                 str[S]            its injection sites (run_site indexes them)
    ff_ids, ff_instances  int32[N], str[N]  its observed flip-flops: id, instance
    pattern_cycle         int32[P]          the cycle of pattern p
    pattern_ff_ptr        int64[P + 1]      the FF ids of pattern p, ascending:
    pattern_ffs           int32[...]          pattern_ffs[pattern_ff_ptr[p]:pattern_ff_ptr[p + 1]]
    pattern_multiplicity  int64[P]
    pattern_n_sites       int32[P]
    pattern_run_ptr       int64[P + 1]      the runs of pattern p: run_*[ptr[p]:ptr[p + 1]]
    run_site              int32[R]          site index
    run_width_ps          int32[R]          pulse width
    run_k_lo, run_k_hi    int32[R]          first and last grid index of the run

Patterns are sorted by cycle, then by their FF ids (compared as tuples); the
runs of a pattern by site, width, k_lo.  The table depends only on the records
and the grid: not on how the records are split into files, nor on ``jobs``.

    from setfi.analysis.patterns import compute_patterns
    pt = compute_patterns("runs/my_top/inject", steps=50)
    print(pt.count, pt.cycle(0), pt.ffs(0), pt.sources(0)[:3])
"""
from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Union

import numpy as np

from .. import _engine
from ..records import COLUMNS, INF_PS, Records

FORMAT = "setfi-patterns/1"


def phase_grid(period_ps: int, steps: int) -> np.ndarray:
    """k * T / steps for k = 0..steps, rounded half up to integer ps."""
    out = []
    for k in range(steps + 1):
        v = Fraction(k * period_ps, steps)
        out.append(int(v.numerator * 2 + v.denominator) // (2 * v.denominator))
    return np.asarray(out, dtype=np.int64)


class Source(NamedTuple):
    """One run of a pattern: a SET at `site` of width `width_ps` upsets exactly
    the pattern's flip-flops when it starts at starts_ps[k] for k_lo <= k <= k_hi,
    i.e. at start_lo_ps .. start_hi_ps on the grid."""
    site: str
    width_ps: int
    k_lo: int
    k_hi: int
    start_lo_ps: int
    start_hi_ps: int


@dataclass(eq=False, repr=False)
class Patterns:
    """The pattern table; its fields are the arrays of patterns.npz (see the
    module docstring)."""
    period_ps: int
    steps: int
    starts_ps: np.ndarray
    cycles: np.ndarray
    pulse_widths_ps: np.ndarray
    sites: np.ndarray
    ff_ids: np.ndarray
    ff_instances: np.ndarray
    pattern_cycle: np.ndarray
    pattern_ff_ptr: np.ndarray
    pattern_ffs: np.ndarray
    pattern_multiplicity: np.ndarray
    pattern_n_sites: np.ndarray
    pattern_run_ptr: np.ndarray
    run_site: np.ndarray
    run_width_ps: np.ndarray
    run_k_lo: np.ndarray
    run_k_hi: np.ndarray

    def __repr__(self) -> str:
        return (f"Patterns({self.count} patterns, {self.run_site.size} runs, "
                f"{int(self.pattern_multiplicity.sum())} grid points; steps {self.steps}, "
                f"T {self.period_ps} ps)")

    @property
    def count(self) -> int:
        """The number of patterns; they are numbered 0 .. count - 1."""
        return int(self.pattern_cycle.size)

    def cycle(self, p: int) -> int:
        return int(self.pattern_cycle[p])

    def multiplicity(self, p: int) -> int:
        """The number of grid points (site, width, k) that upset exactly ffs(p)."""
        return int(self.pattern_multiplicity[p])

    def ffs(self, p: int, names: bool = False) -> List:
        """The flip-flops of pattern p: their ids (ascending), or their instance names."""
        ids = self.pattern_ffs[self.pattern_ff_ptr[p]:self.pattern_ff_ptr[p + 1]].tolist()
        if not names:
            return ids
        name = dict(zip(self.ff_ids.tolist(), self.ff_instances.tolist()))
        return [name[i] for i in ids]

    def sources(self, p: int) -> List[Source]:
        """Where pattern p comes from: its runs, by site, width and k_lo."""
        a, b = self.pattern_run_ptr[p], self.pattern_run_ptr[p + 1]
        runs = zip(self.run_site[a:b].tolist(), self.run_width_ps[a:b].tolist(),
                   self.run_k_lo[a:b].tolist(), self.run_k_hi[a:b].tolist())
        return [Source(str(self.sites[s]), w, lo, hi, int(self.starts_ps[lo]), int(self.starts_ps[hi]))
                for s, w, lo, hi in runs]

    def grid_points(self, cycles: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        """The runs expanded to one row per grid point at which a SET upsets
        flip-flops (of the given cycles; default all), sorted by (cycle, site,
        width_ps, k).  Columns: cycle, site (an index into sites), width_ps, k,
        start_ps (starts_ps[k]) and pattern (the flip-flops upset are
        ffs(pattern))."""
        run_pattern = np.repeat(np.arange(self.count), np.diff(self.pattern_run_ptr))
        runs = np.arange(run_pattern.size)
        if cycles is not None:
            runs = runs[np.isin(self.pattern_cycle[run_pattern], np.asarray(cycles))]
        length = (self.run_k_hi[runs] - self.run_k_lo[runs] + 1).astype(np.int64)
        run = np.repeat(runs, length)
        k = np.arange(run.size) - np.repeat(np.cumsum(length) - length, length) + self.run_k_lo[run]
        pattern = run_pattern[run]
        cols = {"cycle": self.pattern_cycle[pattern], "site": self.run_site[run],
                "width_ps": self.run_width_ps[run], "k": k, "start_ps": self.starts_ps[k],
                "pattern": pattern}
        order = np.lexsort((k, cols["width_ps"], cols["site"], cols["cycle"]))
        return {name: col[order] for name, col in cols.items()}

    def to_csv(self, path: str) -> None:
        """patterns.csv: one row per pattern with its number (pattern), cycle,
        multiplicity, n_ffs, n_sites and ffs (the FF ids joined with "|")."""
        with open(path, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["pattern", "cycle", "multiplicity", "n_ffs", "n_sites", "ffs"])
            for a in range(0, self.count, 65536):          # a block of patterns at a time
                out.writerows(self._csv_rows(a, min(a + 65536, self.count)))

    def _csv_rows(self, a: int, b: int) -> Iterator[list]:
        ptr = (self.pattern_ff_ptr[a:b + 1] - self.pattern_ff_ptr[a]).tolist()
        ffs = self.pattern_ffs[self.pattern_ff_ptr[a]:self.pattern_ff_ptr[b]].tolist()
        rows = zip(self.pattern_cycle[a:b].tolist(), self.pattern_multiplicity[a:b].tolist(),
                   self.pattern_n_sites[a:b].tolist())
        for i, (c, m, s) in enumerate(rows):
            yield [a + i, c, m, ptr[i + 1] - ptr[i], s, "|".join(map(str, ffs[ptr[i]:ptr[i + 1]]))]

    def save(self, path: str) -> None:
        """Write the table to `path` (patterns.npz, compressed)."""
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, format=np.asarray(FORMAT),
                            **{f.name: np.asarray(getattr(self, f.name)) for f in fields(self)})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Patterns":
        with np.load(path) as z:
            fmt = str(z["format"]) if "format" in z.files else None
            if fmt != FORMAT:
                raise ValueError(f"{path}: not a {FORMAT} file (format={fmt!r})")
            arrays: Dict[str, Any] = {f.name: z[f.name] for f in fields(cls)}
        arrays["period_ps"] = int(arrays["period_ps"])
        arrays["steps"] = int(arrays["steps"])
        return cls(**arrays)


def compute_patterns(records_or_dir: Union[str, Records], steps: int, jobs: int = 1) -> Patterns:
    """The pattern table of a campaign's pulse records (a records directory such
    as <dir>/inject, or a Records) on the grid k*T/steps, k = 0..steps.

    The record files are swept in C++, `jobs` files at a time (one per thread),
    so the memory needed is that of the table plus `jobs` record files.  The
    table does not depend on `jobs`."""
    rec = _records(records_or_dir)
    meta = _metadata(rec, steps)
    enter, leave = rec.thresholds()
    builder = _engine.PatternBuilder(meta["starts_ps"], enter, leave, meta["cycles"], len(meta["sites"]),
                                     meta["pulse_widths_ps"], INF_PS)
    _for_each_file(rec, builder.add, jobs)
    if builder.n_split_trials:
        _merge_split_trials(rec, builder, jobs)
    return Patterns(**meta, **builder.result())


def _merge_split_trials(rec: Records, builder, jobs: int) -> None:
    """Sweep again, as a whole, the trials whose records are not one block of
    consecutive records in one file (setfi writes each trial so; other tools
    need not).  The records of these trials are held in memory together."""
    parts: List[Dict[str, np.ndarray]] = []
    _for_each_file(rec, lambda cols: parts.append(_select(cols, builder.split_mask(cols))), jobs)
    merged = {k: np.concatenate([p[k] for p in parts]) for k in COLUMNS}
    order = np.lexsort((merged["width_ps"], merged["site"], merged["cycle"]))
    builder.forget_split_trials()
    builder.add(_select(merged, order))


def reference_patterns(records_or_dir: Union[str, Records], steps: int) -> Patterns:
    """The same table as compute_patterns, straight from the definitions: the
    FF set of every grid point of every trial, then its runs.  Plain Python with
    every record in memory: for tests and cross-checks."""
    rec = _records(records_or_dir)
    meta = _metadata(rec, steps)
    starts = meta["starts_ps"]
    enter, leave = rec.thresholds()
    cols = rec.load_all()
    # F[trial][k]: the FFs that capture for a SET of the trial starting at starts[k]
    F: Dict[tuple, List[set]] = {}
    for c, s, w, f, b, t_enter, t_exit in zip(*(cols[k].tolist() for k in COLUMNS)):
        s_high = enter[b, f] - t_enter - 1
        s_low = -INF_PS if t_exit >= INF_PS else leave[b, f] - t_exit + 1
        sets = F.setdefault((c, s, w), [set() for _ in starts])
        for k in np.flatnonzero((starts >= s_low) & (starts <= s_high)):
            sets[k].add(f)
    # runs: the maximal ranges of k with one non-empty F, by pattern (cycle, F)
    runs: Dict[tuple, List[tuple]] = {}
    for (c, s, w), sets in F.items():
        k_lo = 0
        for k in range(1, len(starts) + 1):
            if k == len(starts) or sets[k] != sets[k_lo]:
                if sets[k_lo]:
                    runs.setdefault((c, tuple(sorted(sets[k_lo]))), []).append((s, w, k_lo, k - 1))
                k_lo = k
    return Patterns(**meta, **_table(runs))


def _table(runs: Dict[tuple, List[tuple]]) -> Dict[str, np.ndarray]:
    """The pattern_* and run_* arrays, in canonical order, of
    {(cycle, FF ids): [(site, width_ps, k_lo, k_hi), ...]}."""
    keys = sorted(runs)
    groups = [sorted(runs[key]) for key in keys]
    flat = [r for g in groups for r in g]

    def ptr(lengths: List[int]) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)]).astype(np.int64)

    def column(i: int) -> np.ndarray:
        return np.asarray([r[i] for r in flat], dtype=np.int32)

    return {"pattern_cycle": np.asarray([c for c, _ in keys], dtype=np.int32),
            "pattern_ff_ptr": ptr([len(ffs) for _, ffs in keys]),
            "pattern_ffs": np.asarray([f for _, ffs in keys for f in ffs], dtype=np.int32),
            "pattern_multiplicity": np.asarray([sum(r[3] - r[2] + 1 for r in g) for g in groups],
                                               dtype=np.int64),
            "pattern_n_sites": np.asarray([len({r[0] for r in g}) for g in groups], dtype=np.int32),
            "pattern_run_ptr": ptr([len(g) for g in groups]),
            "run_site": column(0), "run_width_ps": column(1), "run_k_lo": column(2), "run_k_hi": column(3)}


def _records(records_or_dir: Union[str, Records]) -> Records:
    return records_or_dir if isinstance(records_or_dir, Records) else Records.open(records_or_dir)


def _metadata(rec: Records, steps: int) -> Dict[str, Any]:
    """The arrays of patterns.npz that describe the grid and the campaign."""
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    period = int(rec.index.get("capture_edge_ps", rec.period_ps))
    ffs = sorted(rec.ffs, key=lambda f: int(f["id"]))
    return {"period_ps": period, "steps": int(steps), "starts_ps": phase_grid(period, int(steps)),
            "cycles": np.asarray(rec.cycles, dtype=np.int64),
            "pulse_widths_ps": np.asarray(rec.pulse_widths_ps, dtype=np.int64),
            "sites": np.asarray(rec.sites, dtype=str),
            "ff_ids": np.asarray([int(f["id"]) for f in ffs], dtype=np.int32),
            "ff_instances": np.asarray([f["inst"] for f in ffs], dtype=str)}


def _select(cols: Dict[str, np.ndarray], rows: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[rows] for k, v in cols.items()}


def _for_each_file(rec: Records, work: Callable[[Dict[str, np.ndarray]], None], jobs: int) -> None:
    """work(columns) for each record file of rec, on `jobs` threads, each
    holding one file (numpy decompresses and PatternBuilder.add sweeps without
    holding the GIL)."""
    def read_and_work(path: str) -> None:
        with np.load(path) as z:
            cols = {k: z[k] for k in COLUMNS}
        work(cols)

    paths = [os.path.join(rec.root, s["file"]) for s in rec.index["shards"]]
    if jobs <= 1:
        for path in paths:
            read_and_work(path)
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(read_and_work, path) for path in paths]
        try:
            for f in futures:
                f.result()
        finally:
            pool.shutdown(cancel_futures=True)       # after an error, skip the files not started
