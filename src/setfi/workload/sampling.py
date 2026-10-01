"""Cycle sampling, the unknown-value check, and the cycle-state hex format.

Everything here is pure Python and deterministic given its arguments.

Sampling
--------
The testbench runs cycles first+1..last after reset (first is 0 unless reset is
asserted again).  The sample pool leaves out ``padding`` cycles at each end and
``warmup`` more at the start: ``[first + 1 + padding + warmup, last - padding]``
(optionally intersected with a cycle window).  One
``random.Random(seed)`` stream draws, in this order:

1. the *primary* sample: ``rng.sample(pool, n)``, sorted;
2. the *reserve*: ``rng.sample(rest_of_pool, min(len(rest), reserve_factor * n))``,
   sorted.

Because the primary draw is the first use of the stream, adding the reserve or
the unknown-value check never changes which primary cycles a seed selects.  Both
primary and reserve cycles are recorded; the check afterwards keeps the primary
cycles whose recorded state is resolved and backfills rejected ones from the
reserve, in ascending order.

Unknown-value check
-------------------
A recorded cycle is usable only if at most ``max_x_frac`` of its hex digits are
unresolved (anything that is not 0-9/a-f).  The consuming engine reads X/Z as
0, so the threshold bounds how much of the analysed state may be fabricated;
the selected cycles' actual X fraction is reported.  Fully reset designs record
no X at all; the 2 % default rejects cycles in which un-reset storage is still
being filled.

Hex format
----------
``cycle_<N>.hex`` holds one line: the Verilog ``%h`` rendering of a vector of
``n_nets`` bits whose bit ``i`` is net ``i`` of ``net_index.json`` (net 0 is the
least significant bit), zero-padded to ``ceil(n_nets / 4)`` digits, followed
by ``\\n``.  A digit whose bits are all x is ``x``, partly x ``X``; all z ``z``,
partly z (and no x) ``Z``.
"""
from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .errors import WorkloadError

DEFAULT_MAX_X_FRAC = 0.02
DEFAULT_RESERVE_FACTOR = 3

_HEX_DIGITS = "0123456789abcdefABCDEF"


# ---------------------------------------------------------------------------
# Hex format
# ---------------------------------------------------------------------------
def n_hex_digits(n_nets: int) -> int:
    return (n_nets + 3) // 4


def format_state_hex(values: Sequence) -> str:
    """Render per-net values (index 0 = LSB) the way Verilog ``%h`` renders the
    recorded vector, without the trailing newline.

    Each value is 0/1 (int or str) or one of 'x', 'X', 'z', 'Z'.
    """
    vals = []
    for v in values:
        s = str(v).lower()
        if s not in ("0", "1", "x", "z"):
            raise ValueError(f"not a 4-state bit value: {v!r}")
        vals.append(s)
    n = len(vals)
    digits: List[str] = []
    for d in range(n_hex_digits(n)):
        group = vals[4 * d: min(4 * d + 4, n)]      # LSB-first bits of this digit
        if all(b == "x" for b in group):
            ch = "x"
        elif any(b == "x" for b in group):
            ch = "X"
        elif all(b == "z" for b in group):
            ch = "z"
        elif any(b == "z" for b in group):
            ch = "Z"
        else:
            ch = "0123456789abcdef"[sum(1 << k for k, b in enumerate(group) if b == "1")]
        digits.append(ch)
    return "".join(reversed(digits))


def parse_state_hex(text: str, n_nets: int) -> List[str]:
    """Inverse of :func:`format_state_hex` for resolved digits: per-net values
    ('0', '1', or 'x' for any bit of an unresolved digit), index 0 = LSB."""
    s = text.strip()
    if len(s) != n_hex_digits(n_nets):
        raise ValueError(f"expected {n_hex_digits(n_nets)} hex digits for {n_nets} nets, got {len(s)}")
    out: List[str] = []
    for d, ch in enumerate(reversed(s)):
        width = min(4, n_nets - 4 * d)
        if ch in "0123456789abcdefABCDEF":
            v = int(ch, 16)
            out.extend("1" if (v >> k) & 1 else "0" for k in range(width))
        else:
            out.extend("x" for _ in range(width))
    return out


def x_frac_of_text(text: str) -> float:
    """Share of the recorded digits that are not a resolved 0-9/a-f digit."""
    s = text.strip().replace("\n", "")
    if not s:
        return 1.0
    return sum(1 for c in s if c not in _HEX_DIGITS) / len(s)


def x_frac(path) -> float:
    return x_frac_of_text(Path(path).read_text())


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_cycles(
    first: int, last: int, n_samples: int, padding: int, seed: int,
    warmup: int = 0, reserve_factor: int = DEFAULT_RESERVE_FACTOR,
) -> Tuple[List[int], List[int]]:
    """Return (primary, reserve) drawn from ``[first+padding+warmup, last-padding]``."""
    lo = first + 1 + padding + warmup
    hi = last - padding
    if hi - lo + 1 < n_samples:
        raise WorkloadError(
            "no_cycles_in_range",
            f"only {max(0, hi - lo + 1)} cycles can be sampled (cycles {lo}..{hi} of the "
            f"{last - first} after reset, leaving out {padding} padding cycles at each end"
            + (f" and {warmup} warmup cycles" if warmup else "") + ") "
            f"but workload.n_cycles = {n_samples}. Reduce "
            f"workload.n_cycles, workload.padding_cycles or workload.warmup_cycles, or run the "
            f"testbench longer.",
        )
    rng = random.Random(seed)
    sampled = sorted(rng.sample(range(lo, hi + 1), n_samples))
    taken = set(sampled)
    rest = [c for c in range(lo, hi + 1) if c not in taken]
    n_res = min(len(rest), reserve_factor * n_samples)
    reserve = sorted(rng.sample(rest, n_res)) if n_res else []
    return sampled, reserve


def load_cycle_window(path) -> Tuple[Set[int], Dict]:
    """Read a CSV of inclusive ``start_cycle,end_cycle`` runs.

    Every row must parse and satisfy start <= end; a malformed row is an error,
    not a skipped line (a silently dropped row would shrink the pool).
    """
    p = Path(path)
    if not p.is_file():
        raise WorkloadError("config_invalid", f"cycle_window_csv not found: {p}")
    cycles: Set[int] = set()
    n_rows = 0
    with p.open(newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or not {"start_cycle", "end_cycle"} <= set(rdr.fieldnames):
            raise WorkloadError(
                "config_invalid",
                f"cycle_window_csv {p} needs a header with start_cycle,end_cycle; "
                f"got {rdr.fieldnames}")
        for i, row in enumerate(rdr, start=2):
            try:
                s, e = int(row["start_cycle"]), int(row["end_cycle"])
            except (TypeError, ValueError):
                raise WorkloadError("config_invalid",
                                    f"cycle_window_csv {p} line {i}: not two integers: {row}")
            if s > e:
                raise WorkloadError("config_invalid",
                                    f"cycle_window_csv {p} line {i}: start {s} > end {e}")
            cycles.update(range(s, e + 1))
            n_rows += 1
    if not cycles:
        raise WorkloadError("config_invalid", f"cycle_window_csv {p} holds no cycle")
    return cycles, {
        "csv": str(p.resolve()),
        "md5": hashlib.md5(p.read_bytes()).hexdigest(),
        "n_rows": n_rows,
        "n_cycles": len(cycles),
        "min_cycle": min(cycles),
        "max_cycle": max(cycles),
    }


def sample_cycles_in_window(
    first: int, last: int, n_samples: int, padding: int, seed: int,
    window: Iterable[int], warmup: int = 0,
    reserve_factor: int = DEFAULT_RESERVE_FACTOR,
) -> Tuple[List[int], List[int], int]:
    """:func:`sample_cycles` with the pool restricted to working-range cycles
    inside ``window``.  Returns (primary, reserve, pool size)."""
    window = window if isinstance(window, (set, frozenset)) else set(window)
    lo = first + 1 + padding + warmup
    hi = last - padding
    pool = [c for c in range(lo, hi + 1) if c in window]
    if len(pool) < n_samples:
        raise WorkloadError(
            "no_cycles_in_range",
            f"workload.cycle_window_csv contains {len(pool)} of the cycles that can be "
            f"sampled ({lo}..{hi}) but workload.n_cycles = {n_samples}.",
        )
    rng = random.Random(seed)
    sampled = sorted(rng.sample(pool, n_samples))
    taken = set(sampled)
    rest = [c for c in pool if c not in taken]
    n_res = min(len(rest), reserve_factor * n_samples)
    reserve = sorted(rng.sample(rest, n_res)) if n_res else []
    return sampled, reserve, len(pool)


# ---------------------------------------------------------------------------
# unknown-value check
# ---------------------------------------------------------------------------
def screen_cycles_by_frac(
    frac: Dict[int, float], primary: Sequence[int], reserve: Sequence[int],
    n_samples: int, max_x_frac: float = DEFAULT_MAX_X_FRAC,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[int], Dict]:
    """Keep the primary cycles whose X fraction is <= ``max_x_frac``; backfill
    from ``reserve`` in order.  ``frac`` maps recorded cycle -> X fraction (a
    cycle missing from it counts as fully unresolved)."""
    log = log or (lambda _m: None)
    keep = [c for c in primary if frac.get(c, 1.0) <= max_x_frac]
    rejected = [c for c in primary if c not in keep]
    backfill: List[int] = []
    if rejected:
        for c in reserve:
            if len(keep) + len(backfill) >= n_samples:
                break
            if frac.get(c, 1.0) <= max_x_frac:
                backfill.append(c)
    selected = sorted(keep + backfill)

    if rejected:
        worst = max(frac.get(c, 1.0) for c in rejected)
        log(f"  unknown values: {len(rejected)}/{len(primary)} sampled cycles replaced "
            f"(worst {100 * worst:.1f}% unresolved, threshold "
            f"{100 * max_x_frac:.1f}%); backfilled {len(backfill)} from reserve")
        log(f"    rejected: {rejected[:10]}{' ...' if len(rejected) > 10 else ''}")
    if len(selected) < n_samples:
        clean_total = sum(1 for v in frac.values() if v <= max_x_frac)
        raise WorkloadError(
            "no_cycles_in_range",
            f"only {len(selected)} of the requested {n_samples} cycles "
            f"have a resolved recorded state ({clean_total} clean out of "
            f"{len(frac)} candidates recorded). The design is in an unresolved "
            f"state (X) over most of its run -- typically un-reset storage that "
            f"never finishes filling within the testbench. Lengthen the workload, "
            f"reduce workload.n_cycles, raise workload.max_x_fraction, or reset the "
            f"storage.",
        )
    sel_fr = [frac[c] for c in selected if c in frac]
    return selected, {
        # unknown (X/Z) hex digits of the recorded states; inject reads them as 0
        "max_x_fraction": max_x_frac,
        "x_fraction_max": round(max(sel_fr), 8) if sel_fr else 0.0,
        "x_fraction_mean": round(sum(sel_fr) / len(sel_fr), 8) if sel_fr else 0.0,
        "n_candidates_recorded": len(frac),
        "replaced_cycles": rejected,
        "replacement_cycles": backfill,
        "x_fraction_by_cycle": {str(k): round(v, 6) for k, v in sorted(frac.items())},
    }


def screen_cycles(
    cs_dir, primary: Sequence[int], reserve: Sequence[int], n_samples: int,
    max_x_frac: float = DEFAULT_MAX_X_FRAC,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[int], Dict]:
    """Run the unknown-value check on the ``cycle_<N>.hex`` files recorded in ``cs_dir``."""
    cs_dir = Path(cs_dir)
    frac: Dict[int, float] = {}
    for c in list(primary) + list(reserve):
        p = cs_dir / f"cycle_{c}.hex"
        if p.is_file():
            frac[c] = x_frac(p)
    return screen_cycles_by_frac(frac, primary, reserve, n_samples, max_x_frac, log)
