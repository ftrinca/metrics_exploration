from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from metric_eval.core.metric_config import CATEGORIES, METRIC_DIRECTION
from metric_eval.experiments.injector.config import DISTORTION_NAMES, INJECTOR_CATEGORIES, INJECTOR_METRICS

# Relative spread below this counts as not distinguishing the eight distortions.
FLAT_RELATIVE_SPREAD = 0.10


def _row(value_table, metric):
    """One metric's values across the eight distortions, in DISTORTION_NAMES order."""
    return [value_table.get(metric, {}).get(d) for d in DISTORTION_NAMES]


def spread(value_table: dict[str, dict[str, float | None]]) -> dict[str, dict]:
    """Per metric: absolute and relative range across the eight distortions.

    "relative" is the range over the mean absolute level, because the metrics
    live on completely different scales. A metric with fewer than two non-None
    values is reported flat with both ranges None.
    """
    out = {}
    for metric in INJECTOR_METRICS:
        vals = [v for v in _row(value_table, metric) if v is not None]
        if len(vals) < 2:
            out[metric] = {"range": None, "relative": None, "flat": True}
            continue
        rng = float(max(vals) - min(vals))
        scale = float(np.mean(np.abs(vals)))
        rel = rng / scale if scale > 0 else 0.0
        out[metric] = {
            "range": rng,
            "relative": rel,
            "flat": rel < FLAT_RELATIVE_SPREAD,
        }
    return out


def zscores(value_table: dict[str, dict[str, float | None]]) -> dict[str, dict[str, float]]:
    """Signed z-score per (metric, distortion), positive meaning worse.

    The sign is adjusted for each metric's own direction. A metric with too few
    values, or with no variation across the eight, scores 0.0 everywhere.
    """
    z = {}
    for metric in INJECTOR_METRICS:
        row = value_table.get(metric, {})
        vals = [row[d] for d in DISTORTION_NAMES if row.get(d) is not None]
        sign = -1.0 if METRIC_DIRECTION[metric] == "higher" else 1.0
        if len(vals) < 2:
            z[metric] = {d: 0.0 for d in DISTORTION_NAMES}
            continue
        mean_v, std_v = float(np.mean(vals)), float(np.std(vals))
        z[metric] = {}
        for d in DISTORTION_NAMES:
            v = row.get(d)
            z[metric][d] = 0.0 if (v is None or std_v == 0.0) else sign * (v - mean_v) / std_v
    return z


def agreement(value_table: dict[str, dict[str, float | None]]) -> dict[str, dict[str, float]]:
    """Spearman correlation between every pair of metrics over the eight distortions.

    Values are sign-adjusted first so that "worse" points the same way for all
    metrics. A pair with fewer than three shared values, or with no variation on
    either side, gives nan.
    """
    oriented = {}
    for metric in INJECTOR_METRICS:
        sign = -1.0 if METRIC_DIRECTION[metric] == "higher" else 1.0
        row = _row(value_table, metric)
        oriented[metric] = [None if v is None else sign * v for v in row]

    out = {}
    for a in INJECTOR_METRICS:
        out[a] = {}
        for b in INJECTOR_METRICS:
            pairs = [(x, y) for x, y in zip(oriented[a], oriented[b])
                     if x is not None and y is not None]
            if len(pairs) < 3:
                out[a][b] = float("nan")
                continue
            xs, ys = zip(*pairs)
            if len(set(xs)) < 2 or len(set(ys)) < 2:
                out[a][b] = float("nan")
                continue
            rho = spearmanr(xs, ys).statistic
            out[a][b] = float(rho)
    return out


def summary_table(value_table, spreads) -> str:
    """Render the raw values plus the spread columns, marking flat rows with an asterisk."""
    lines = [
        "DAMAGE REACTIVITY",
        "=" * 104,
    ]
    header = f"{'metric':<10}" + "".join(f"{d[:9]:>11}" for d in DISTORTION_NAMES) + f"{'rel.spread':>12}"
    lines.append(header)
    lines.append("-" * 104)
    for cat in INJECTOR_CATEGORIES:
        lines.append(cat)
        for metric in CATEGORIES[cat]:
            if metric not in INJECTOR_METRICS:
                continue
            row = value_table.get(metric, {})
            cells = ""
            for d in DISTORTION_NAMES:
                v = row.get(d)
                cells += f"{'--':>11}" if v is None else f"{v:>11.4f}"
            sp = spreads[metric]["relative"]
            tag = "" if sp is None else (f"{sp:>11.3f}" + ("*" if spreads[metric]["flat"] else " "))
            lines.append(f"  {metric:<8}" + cells + tag)
    lines.append("-" * 104)
    lines.append(f"* relative spread below {FLAT_RELATIVE_SPREAD:.0%}")
    return "\n".join(lines)

def deviations(value_table: dict[str, dict[str, float | None]],
               pinned: tuple[str, ...] = ()) -> dict[str, dict[str, float] | None]:
    """Distance from the perfect value per (metric, distortion), as a share of
    the metric's own largest distance in this table.

    The same reading as the response grid of Figure "response_grid": lower-is-
    better metrics are distances already; for the higher-is-better ones the
    distance is (perfect - value), with perfect = 1 for Pearson and R2 and the
    largest observed value standing in for MI. A metric in `pinned` (MAE and ND
    without the RMSE-calibrated pass) is returned as None, since its column is
    held constant by the calibration and carries no signal.
    """
    out: dict[str, dict[str, float] | None] = {}
    for metric in INJECTOR_METRICS:
        if metric in pinned:
            out[metric] = None
            continue
        row = [value_table.get(metric, {}).get(d) for d in DISTORTION_NAMES]
        vals = [v for v in row if v is not None]
        if not vals:
            out[metric] = None
            continue
        if METRIC_DIRECTION[metric] == "higher":
            perfect = 1.0 if metric in ("pearson", "r2") else max(vals)
            row = [None if v is None else abs(perfect - v) for v in row]
        else:
            row = [None if v is None else abs(v) for v in row]
        top = max(v for v in row if v is not None)
        out[metric] = {d: (0.0 if v is None or top <= 0 else v / top)
                       for d, v in zip(DISTORTION_NAMES, row)}
    return out
