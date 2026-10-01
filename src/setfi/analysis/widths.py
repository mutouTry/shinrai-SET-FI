"""SET pulse-width models: a probability for each injected width.

The campaign injects a fixed list of widths; a width model only weights the
trials of each width afterwards, so changing the model never requires a new
campaign.  Weights are normalised to sum to 1 over the injected widths.

    uniform                       p(w) = const
    gaussian     mean_ps, sd_ps   p(w) ~ exp(-(w - mean)^2 / (2 sd^2))
    exponential  tau_ps           p(w) ~ exp(-w / tau)
    table        weights = {width_ps: weight}  (every injected width, nothing else)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping


class WidthModelError(ValueError):
    pass


def _num(model: Mapping[str, Any], key: str, label: str = "") -> float:
    v = model[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise WidthModelError(f"{label or key} must be a number, got {v!r}")
    return float(v)


def width_weights(model: Mapping[str, Any], widths_ps: List[int]) -> Dict[int, float]:
    if not widths_ps:
        raise WidthModelError("no pulse widths")
    allowed = {"uniform": set(), "gaussian": {"mean_ps", "sd_ps"},
               "exponential": {"tau_ps"}, "table": {"weights"}}
    if "type" not in model:
        raise WidthModelError(f"a width model needs a type: one of {sorted(allowed)}")
    kind = str(model["type"]).lower()
    if kind not in allowed:
        raise WidthModelError(f"unknown width model type {kind!r}; use one of {sorted(allowed)}")
    extra = set(model) - {"type"} - allowed[kind]
    if extra:
        raise WidthModelError(f"width model {kind!r} does not take {sorted(extra)}")
    missing = allowed[kind] - set(model)
    if missing:
        raise WidthModelError(f"width model {kind!r} needs {sorted(missing)}")

    if kind == "uniform":
        raw = {w: 1.0 for w in widths_ps}
    elif kind == "gaussian":
        mean, sd = _num(model, "mean_ps"), _num(model, "sd_ps")
        if sd <= 0:
            raise WidthModelError("gaussian sd_ps must be > 0")
        raw = {w: math.exp(-((w - mean) ** 2) / (2.0 * sd * sd)) for w in widths_ps}
        if max(raw.values()) < 1e-200:
            # far from every injected width: normalise in log space (the widest or
            # narrowest injected width then dominates)
            lg = {w: -((w - mean) ** 2) / (2.0 * sd * sd) for w in widths_ps}
            top = max(lg.values())
            raw = {w: math.exp(v - top) for w, v in lg.items()}
    elif kind == "exponential":
        tau = _num(model, "tau_ps")
        if tau <= 0:
            raise WidthModelError("exponential tau_ps must be > 0")
        w0 = min(widths_ps)          # a shift of w only rescales, and scale is normalised away
        raw = {w: math.exp(-(w - w0) / tau) for w in widths_ps}
    else:
        if not isinstance(model["weights"], Mapping):
            raise WidthModelError('weights must be a table {"<width_ps>" = weight, ...}')
        table = {}
        for k in model["weights"]:
            try:
                w = int(float(k))
            except ValueError:
                raise WidthModelError(f"weights: key {k!r} is not a pulse width in ps") from None
            table[w] = _num(model["weights"], k, f"weights.{k}")
        missing = sorted(set(widths_ps) - set(table))
        unknown = sorted(set(table) - set(widths_ps))
        if missing or unknown:
            raise WidthModelError(f"width table must name exactly the injected widths; "
                                  f"missing {missing}, not injected {unknown}")
        if any(v < 0 for v in table.values()):
            raise WidthModelError("width table has a negative weight")
        raw = {w: table[w] for w in widths_ps}
    total = sum(raw.values())
    # (a distribution far from every injected width is normalised in log space above)
    if total <= 0:
        raise WidthModelError("the width table gives every injected width weight 0")
    return {w: raw[w] / total for w in widths_ps}


def width_model_notes(model: Mapping[str, Any], widths_ps: List[int]) -> List[str]:
    """Remarks on a valid model whose mass lies mostly outside the injected widths."""
    lo, hi = min(widths_ps), max(widths_ps)
    if model.get("type") == "gaussian":
        mean, sd = float(model["mean_ps"]), float(model["sd_ps"])
        if mean + 2 * sd < lo or mean - 2 * sd > hi:
            return [f"most of the gaussian (mean {mean:g} ps, sd {sd:g} ps) lies outside the "
                    f"injected widths {lo}..{hi} ps; nearly all its weight falls on "
                    f"{lo if mean < lo else hi} ps"]
    if model.get("type") == "exponential":
        tau = float(model["tau_ps"])
        w = sorted(widths_ps)
        if len(w) > 1 and w[1] - w[0] > 5 * tau:
            return [f"tau_ps = {tau:g} is small against the spacing of the injected widths; "
                    f"nearly all the weight falls on {lo} ps"]
    return []
