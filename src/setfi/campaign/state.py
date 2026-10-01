"""Per-cycle circuit state recorded by the workload simulation.

One file per sampled cycle, ``cycle_<N>.hex``: a single hex number whose bit i
is the settled value of global net i (numbering of net_index.json) just before
the clock edge that ends cycle N.  X and Z are read as 0.
"""
from __future__ import annotations

import json
import os
from typing import List

import numpy as np


class StateError(RuntimeError):
    pass


def read_cycle_bits(state_dir: str, cycle: int, n_nets: int) -> np.ndarray:
    path = os.path.join(state_dir, f"cycle_{int(cycle)}.hex")
    if not os.path.exists(path):
        raise StateError(f"recorded state for cycle {cycle} is missing: {path}")
    with open(path, "r", encoding="ascii") as f:
        hex_str = f.read().strip()
    hex_clean = hex_str.translate(str.maketrans("xXzZ", "0000"))
    val = int(hex_clean, 16) if hex_clean else 0
    if val.bit_length() > n_nets:
        raise StateError(f"{path}: {val.bit_length()} bits recorded but the netlist has {n_nets} nets "
                         f"(state recorded against a different netlist?)")
    raw = np.frombuffer(val.to_bytes((n_nets + 7) // 8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n_nets].astype(np.uint8)


def read_sampled_cycles(meta_path: str) -> List[int]:
    """The sampled cycles listed in the record stage's cycles.json."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("format") != "setfi-cycles/1":
        raise StateError(f"{meta_path}: not a setfi-cycles/1 file")
    cycles = [int(c) for c in meta.get("cycles", [])]
    if not cycles:
        raise StateError(f"{meta_path} lists no sampled cycles")
    return cycles
