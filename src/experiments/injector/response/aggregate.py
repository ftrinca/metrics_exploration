import argparse
import json
import os

import numpy as np

from core.metric_config import CATEGORIES, METRIC_DIRECTION
from metric_eval.experiments.injector.config import INJECTOR_METRICS
from metric_eval.experiments.injector.config import (
    DAMAGE_LEVELS, DISTORTION_NAMES, INJECTOR_CATEGORIES, RESPONSE_PATTERN,
    RESPONSE_PLOT_DIR, RESPONSE_RATE, RESPONSE_REPORT_DIR, response_dir,
)
from metric_eval.experiments.injector.config import METRIC_LABEL
from experiments.injector.response import plot_all_sweeps

FLAT_TOLERANCE = 1e-9


def _load_all():
    """Read every distortion's sweep scores as {distortion: {metric: [value per level]}}."""
    out = {}
    for name in DISTORTION_NAMES:
        path = os.path.join(response_dir(name), "scores.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} — run python -m injector.response.score first.")
        with open(path) as f:
            scores = json.load(f)
        out[name] = {
            metric: [scores[metric].get(f"L{i}") for i in range(1, len(DAMAGE_LEVELS) + 1)]
            for metric in INJECTOR_METRICS
        }
    return out


def _classify(metric, values):
    """Return ('flat' | 'monotonic' | 'non-monotonic' | 'no data', total movement)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return "no data", 0.0
    move = float(max(vals) - min(vals))
    if move <= FLAT_TOLERANCE:
        return "flat", 0.0
    if METRIC_DIRECTION[metric] == "higher":
        vals = [-v for v in vals]
    diffs = np.diff(vals)
    if np.all(diffs >= -1e-12):
        return "monotonic", move
    return "non-monotonic", move


def report(data) -> str:
    """Render the per-metric, per-distortion flat/monotonic/non-monotonic table."""
    lines = [
        f"DAMAGE RESPONSE — {RESPONSE_PATTERN}, {RESPONSE_RATE:.0%} missing",
        "=" * 96,
        f"damage levels (MAE / sigma): {DAMAGE_LEVELS}",
        "cells: mono = rises with damage, NON-mono = does not, FLAT = never moves",
        "-" * 96,
        f"{'metric':<10}" + "".join(f"{d[:9]:>11}" for d in DISTORTION_NAMES),
        "-" * 96,
    ]
    counts = {}
    for cat in INJECTOR_CATEGORIES:
        lines.append(cat)
        for metric in CATEGORIES[cat]:
            if metric not in INJECTOR_METRICS:
                continue
            cells = ""
            mono = 0
            for d in DISTORTION_NAMES:
                kind, _ = _classify(metric, data[d][metric])
                short = {"monotonic": "mono", "non-monotonic": "NON-mono",
                         "flat": "FLAT", "no data": "--"}[kind]
                cells += f"{short:>11}"
                mono += (kind == "monotonic")
            counts[metric] = mono
            lines.append(f"  {metric:<8}" + cells)
    lines.append("-" * 96)
    lines.append("Monotonic in all eight: " + ", ".join(
        METRIC_LABEL[m] for m, c in counts.items() if c == len(DISTORTION_NAMES)) or "  (none)")
    weak = sorted((c, m) for m, c in counts.items())[:5]
    lines.append("Fewest monotonic sweeps: " + ", ".join(
        f"{METRIC_LABEL[m]} ({c}/8)" for c, m in weak))

    # The movement magnitudes back the chapter's size comparisons (ACF under
    # lag against spikes), which the mono/NON-mono cells alone cannot.
    lines += ["", "MOVEMENT OVER THE SWEEP  (max - min across the seven "
              "levels, in each metric's own scale)",
              f"{'metric':<10}" + "".join(f"{d[:9]:>11}" for d in DISTORTION_NAMES)]
    for cat in INJECTOR_CATEGORIES:
        for metric in CATEGORIES[cat]:
            if metric not in INJECTOR_METRICS:
                continue
            cells = ""
            for d in DISTORTION_NAMES:
                _, move = _classify(metric, data[d][metric])
                cells += f"{move:>11.4f}"
            lines.append(f"  {metric:<8}" + cells)
    acf_moves = {d: _classify("acf", data[d]["acf"])[1] for d in DISTORTION_NAMES}
    lines.append(
        f"ACF moves {acf_moves['lag']:.4f} under lag against "
        f"{acf_moves['spikes']:.4f} under spikes: "
        f"1/{acf_moves['spikes'] / acf_moves['lag']:.0f} of it")
    return "\n".join(lines)


def aggregate_phase():
    """Write one figure per category and the monotonicity report."""
    data = _load_all()

    plot_all_sweeps(DAMAGE_LEVELS, data, RESPONSE_PLOT_DIR)

    os.makedirs(RESPONSE_REPORT_DIR, exist_ok=True)
    path = os.path.join(RESPONSE_REPORT_DIR, "damage_response.txt")
    with open(path, "w") as f:
        f.write(report(data) + "\n")
    print(f"   report -> {path}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Damage response — aggregate phase.").parse_args()
    aggregate_phase()
