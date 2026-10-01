"""The pulse-record format: what a campaign writes and the analysis reads.

The inject stage writes ``<dir>/inject/``:

    index.json          format tag, clock period, pulse widths, sampled cycles,
                        injection sites, observed FFs (with their capture
                        window), the record files, counts
    records_c<i>_s<j>.npz   the records of block i of consecutive sampled
                        cycles (inject.cycles_per_shard of them) and slice j of
                        the sites (one slice per parallel job)
    phase_grid_captures.csv   only with inject.phase_grid_steps: one row per
                        (cycle, site, start_ps, width_ps) that upsets flip-flops,
                        ``ffs`` = their ids joined with ``|``

One record = one pulse seen at the D pin of one observed FF in one trial.
A trial is (cycle, site, pulse width); the SET start time is not part of the
trial, because one simulation covers every start time within the cycle (the
circuit state is held fixed for the cycle, so a later start only shifts the
pulse).  Record columns:

    cycle    int32   sampled cycle number
    site     int32   index into index.json "sites"
    width_ps int32   injected pulse width
    ff       int32   FF id (index.json "ffs"[*]["id"])
    base     uint8   fault-free value of the D pin in this cycle
    t_enter  int64   pulse start at the D pin, ps after the SET start
    t_exit   int64   pulse end (exclusive); INF_PS if still present at the
                     end of the simulation window

Each FF in index.json has ``capture_enter_ps`` and ``capture_leave_ps``, both
indexed by ``base``: enter = T + hold and leave = T - setup, for the data edge
that starts and ends the pulse.  The FF captures a wrong value for a SET
starting at time s (ps from the start of the cycle) iff

    leave[base] - t_exit + 1  <=  s  <=  enter[base] - t_enter - 1

(the pulse overlaps the FF's setup/hold window around the capture edge at T).
Trials with no record reached no observed FF.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

FORMAT = "setfi-records/1"
INF_PS = int(2**62)

COLUMNS = {
    "cycle": np.int32, "site": np.int32, "width_ps": np.int32, "ff": np.int32,
    "base": np.uint8, "t_enter": np.int64, "t_exit": np.int64,
}


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_shard(path: str, cols: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = None
    out = {}
    for k, dt in COLUMNS.items():
        a = np.asarray(cols[k])
        if a.size and a.dtype != dt:
            b = a.astype(dt)
            if not np.array_equal(b.astype(np.int64), a.astype(np.int64)):
                raise ValueError(f"column {k} does not fit {np.dtype(dt).name}")
            a = b
        out[k] = a.astype(dt, copy=False)
        if n is None:
            n = out[k].size
        elif out[k].size != n:
            raise ValueError(f"column {k} has {out[k].size} rows, expected {n}")
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return {"file": os.path.basename(path), "n_records": int(n or 0), "md5": md5_file(path)}


@dataclass
class Records:
    """Read access to a records directory."""
    root: str
    index: Dict[str, Any]

    @classmethod
    def open(cls, root: str) -> "Records":
        path = os.path.join(root, "index.json")
        with open(path, "r", encoding="utf-8") as f:
            index = json.load(f)
        if index.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT} index (format={index.get('format')!r})")
        return cls(root=root, index=index)

    @property
    def period_ps(self) -> int:
        return int(self.index["clock_period_ps"])

    @property
    def pulse_widths_ps(self) -> List[int]:
        return [int(w) for w in self.index["pulse_widths_ps"]]

    @property
    def cycles(self) -> List[int]:
        return [int(c) for c in self.index["cycles"]]

    @property
    def sites(self) -> List[str]:
        return list(self.index["sites"])

    @property
    def ffs(self) -> List[Dict[str, Any]]:
        return list(self.index["ffs"])

    def thresholds(self) -> tuple:
        """(enter, leave): int64 arrays of shape [2, max_id + 1] indexed [base, ff id]."""
        ffs = self.ffs
        n = 1 + max((int(f["id"]) for f in ffs), default=-1)
        enter = np.zeros((2, n), dtype=np.int64)
        leave = np.zeros((2, n), dtype=np.int64)
        for f in ffs:
            i = int(f["id"])
            enter[:, i] = f["capture_enter_ps"]
            leave[:, i] = f["capture_leave_ps"]
        return enter, leave

    def shards(self, verify: bool = False) -> Iterator[Dict[str, np.ndarray]]:
        for ent in self.index["shards"]:
            path = os.path.join(self.root, ent["file"])
            if verify and md5_file(path) != ent["md5"]:
                raise ValueError(f"{path}: md5 does not match index.json")
            with np.load(path) as d:
                yield {k: d[k] for k in COLUMNS}

    def load_all(self, cycles: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
        parts = list(self.shards())
        cols = {k: (np.concatenate([p[k] for p in parts]) if parts else np.zeros(0, dt))
                for k, dt in COLUMNS.items()}
        if cycles is not None:
            keep = np.isin(cols["cycle"], np.asarray(cycles, dtype=np.int32))
            cols = {k: v[keep] for k, v in cols.items()}
        return cols
