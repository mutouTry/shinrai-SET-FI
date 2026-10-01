"""Turn exposure totals into probabilities and write them out.

All probabilities are per SET: one SET at a site drawn uniformly from the
injection sites, in a cycle drawn uniformly from the sampled cycles, at a start
time uniform over the cycle, with a width drawn from the width model.

    summary.json        overall upset probability, per-width values,
                        multi-bit-upset distribution, provenance
    per_ff.csv          probability that the SET upsets each observed FF
    per_site.csv        probability that a SET upsets >= 1 FF, given it occurs at that site
    per_width.csv       upset probability conditional on each width
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Optional

import numpy as np

from ..records import Records
from .exposure import ExposureTotals, accumulate, domain_length
from .widths import width_weights


def analyze(records_dir: str, out_dir: str, width_model: Dict[str, Any],
            phase_domain: str = "half_open", write_trials: bool = False,
            provenance: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    rec = Records.open(records_dir)
    weights = width_weights(width_model, rec.pulse_widths_ps)
    tot = accumulate(rec, weights, phase_domain=phase_domain, keep_trials=write_trials)
    summary = summarize(rec, tot, width_model, phase_domain)
    if provenance:
        summary["provenance"] = dict(provenance)
    os.makedirs(out_dir, exist_ok=True)
    write_outputs(rec, tot, summary, out_dir, phase_domain, write_trials)
    log(f"[analyze] P(upset >= 1 FF per SET) = {summary['upset_probability']:.6g} "
        f"({width_model.get('type', 'uniform')} widths, {len(rec.cycles)} cycles x "
        f"{len(rec.sites)} sites)")
    return summary


def summarize(rec: Records, tot: ExposureTotals, width_model: Dict[str, Any],
              phase_domain: str) -> Dict[str, Any]:
    n_c, n_s = len(rec.cycles), len(rec.sites)
    dom = domain_length(rec.period_ps, phase_domain)
    denom = float(n_c * n_s * dom)
    p_w = tot.exposure_by_width / denom
    p_upset = float(np.dot(tot.weights, p_w))
    mult = {str(m): v / denom for m, v in tot.multiplicity.items()}
    mean_m = (sum(int(m) * v for m, v in tot.multiplicity.items()) /
              sum(tot.multiplicity.values())) if tot.multiplicity else 0.0
    return {
        "upset_probability": p_upset,
        "definition": "P(a SET upsets at least one observed FF): site uniform over injection "
                      "sites, cycle uniform over sampled cycles, start time uniform over the "
                      "phase domain, width from the width model",
        "multi_bit_upset_probability": {k: v for k, v in mult.items()},
        "mean_upset_multiplicity_given_upset": mean_m,
        "per_width": [{"width_ps": w, "weight": float(tot.weights[i]), "upset_probability": float(p_w[i]),
                       "trials_with_upset": int(tot.n_trials_upset_by_width[i])}
                      for i, w in enumerate(tot.widths)],
        "exposure_ps_weighted": float(np.dot(tot.weights, tot.exposure_by_width)),
        "exposure_ps_by_width": [float(x) for x in tot.exposure_by_width],
        "exposure_definition": "exposure_ps_by_width[i]: sum over all (cycle, site) trials of "
                               "width i of the length (ps) of SET start times that upset >= 1 "
                               "FF; exposure_ps_weighted: the same summed over widths with "
                               "the width-model weights",
        "width_model": dict(width_model),
        "phase_domain": phase_domain,
        "start_times_per_cycle": dom,
        "start_times_definition": "SET start times are the integer ps in [0, T) (half_open) "
                                  "or [0, T] (closed)",
        "clock_period_ps": rec.period_ps,
        "n_cycles": n_c,
        "n_sites": n_s,
        "n_observed_ffs": len(rec.ffs),
        "n_trials": int(rec.index["n_trials"]),
    }


def write_outputs(rec: Records, tot: ExposureTotals, summary: Dict[str, Any], out_dir: str,
                  phase_domain: str, write_trials: bool) -> None:
    n_c, n_s = len(rec.cycles), len(rec.sites)
    dom = domain_length(rec.period_ps, phase_domain)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out_dir, "per_ff.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ff_id", "instance", "d_net", "upset_probability"])
        for ff in sorted(rec.ffs, key=lambda x: -tot.ff_exposure[int(x["id"])]):
            wr.writerow([ff["id"], ff["inst"], ff["d_net"],
                         f"{tot.ff_exposure[int(ff['id'])] / (n_c * n_s * dom):.9g}"])
    with open(os.path.join(out_dir, "per_site.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["site", "upset_probability_given_site"])
        order = np.argsort(-tot.site_exposure, kind="stable")
        for i in order:
            wr.writerow([rec.sites[i], f"{tot.site_exposure[i] / (n_c * dom):.9g}"])
    with open(os.path.join(out_dir, "per_width.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["width_ps", "weight", "upset_probability"])
        for row in summary["per_width"]:
            wr.writerow([row["width_ps"], f"{row['weight']:.9g}", f"{row['upset_probability']:.9g}"])
    if write_trials and tot.trial_table is not None:
        t = np.concatenate(tot.trial_table) if tot.trial_table else np.zeros((0, 2), np.int64)
        n_w = len(tot.widths)
        np.savez_compressed(
            os.path.join(out_dir, "trial_exposure.npz"),
            cycle=np.asarray(rec.cycles, dtype=np.int64)[t[:, 0] // (n_s * n_w)],
            site=(t[:, 0] // n_w) % n_s, width_ps=np.asarray(tot.widths)[t[:, 0] % n_w],
            exposure_ps=t[:, 1], site_names=np.asarray(rec.sites))
