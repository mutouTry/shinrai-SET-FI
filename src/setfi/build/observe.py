"""Which flip-flops are observation points.

The sequential-cell inference accepts anything with a clock, a data input and a state
output, which includes integrated clock-gating cells (enable pin taken as "D") and
synthesis-inserted clock-gate registers.  Their capture is not a data observation, so
they are removed from ``ffid_to_info`` before the campaign sees the FF index.

The filter is gap-preserving: kept entries keep their original ffid; nothing is
renumbered, and ``dnet_to_ffids`` / ``qnet_to_ffids`` are left as built.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence


def _predicates(non_observed_cells: Sequence[str], observed_d_pins: Sequence[str],
                non_observed_instances: Sequence[str]) -> List[str]:
    preds: List[str] = []
    for p in non_observed_cells:
        if p.startswith("(?i)"):
            preds.append(f"celltype !~ {p[4:]} (case-insensitive)")
        else:
            preds.append(f"celltype !~ {p}")
    if len(observed_d_pins) == 1:
        preds.append(f"d_pin == {observed_d_pins[0]!r}")
    else:
        preds.append(f"d_pin in {list(observed_d_pins)!r}")
    for s in non_observed_instances:
        preds.append(f"{s!r} not in ff_inst")
    return preds


def filter_observed(ff_index: Dict[str, Any], non_observed_cells: Sequence[str],
                    observed_d_pins: Sequence[str],
                    non_observed_instances: Sequence[str]) -> Dict[str, Any]:
    """Return a copy of ``ff_index`` whose ``ffid_to_info`` holds only observed FFs, with
    the filter's account under ``_meta.observe_filter``.

    Predicates, applied in order (the first that fires names the removal bucket):
      1. cell type fully matches one of ``non_observed_cells``           -> "excluded_cell"
      2. the data pin is not one of ``observed_d_pins``                  -> "non_data_pin"
      3. the instance name contains one of ``non_observed_instances``    -> "excluded_instance"
    """
    cell_res = [re.compile(p) for p in non_observed_cells]
    d_pins = set(observed_d_pins)
    info = ff_index.get("ffid_to_info", {})
    removed: Dict[str, List[Any]] = {"excluded_cell": [], "non_data_pin": [], "excluded_instance": []}
    kept: Dict[str, Any] = {}
    for ffid, e in info.items():
        ct = e.get("celltype", "")
        dp = e.get("d_pin", "")
        inst = e.get("ff_inst", "")
        if any(r.fullmatch(ct) for r in cell_res):
            removed["excluded_cell"].append((ffid, inst, ct))
        elif dp not in d_pins:
            removed["non_data_pin"].append((ffid, inst, dp))
        elif any(s in inst for s in non_observed_instances):
            removed["excluded_instance"].append((ffid, inst))
        else:
            kept[ffid] = e
    out = dict(ff_index)
    out["ffid_to_info"] = kept
    meta = dict(out.get("_meta", {}))
    meta["observe_filter"] = {
        "predicates": _predicates(non_observed_cells, observed_d_pins, non_observed_instances),
        "n_input": len(info),
        "n_kept": len(kept),
        "n_removed": {k: len(v) for k, v in removed.items()},
    }
    out.pop("_meta", None)
    out["_meta"] = meta
    return out


def removed_examples(before: Dict[str, Any], after: Dict[str, Any], limit: int = 20
                     ) -> List[str]:
    """Instance names of (up to ``limit``) FFs the filter removed, for the manifest."""
    kept = after.get("ffid_to_info", {})
    out = [e.get("ff_inst", "") for k, e in before.get("ffid_to_info", {}).items()
           if k not in kept]
    return out[:limit]
