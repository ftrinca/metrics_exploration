import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from core.metric_config import CATEGORIES, METRIC_DIRECTION
from injector.analysis import INJECTOR_METRICS
from injector.config import (
    DAMAGE_LEVELS, DISTORTION_NAMES, INJECTOR_CATEGORIES, SWEEP_PATTERN,
    SWEEP_PLOT_DIR, SWEEP_RATE, SWEEP_REPORT_DIR, sweep_dir,
)
from injector.plotting import METRIC_LABEL, plot_sweep

FLAT_TOLERANCE = 1e-9


def _load_all():
    """Read every distortion's sweep scores as {distortion: {metric: [value per level]}}."""
    out = {}
    for name in DISTORTION_NAMES:
        path = os.path.join(sweep_dir(name), "scores.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} — run injector/score_sweep.py first.")
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
        f"DAMAGE SWEEP — {SWEEP_PATTERN}, {SWEEP_RATE:.0%} missing",
        "=" * 96,
        f"damage levels (MAE / sigma): {DAMAGE_LEVELS}",
        "",
        "Every distortion is solved to the same damage at each level, so the",
        "eight columns are directly comparable. 'flat' means the metric did",
        "not move at all across the whole sweep for that distortion, which is",
        "an exact blind spot rather than a weak reaction.",
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
    return "\n".join(lines)


def aggregate_sweep_phase():
    """Write one figure per category and the monotonicity report."""
    data = _load_all()

    for cat in INJECTOR_CATEGORIES:
        slug = cat.lower().replace(" / ", "_").replace(" ", "_")
        plot_sweep(
            DAMAGE_LEVELS, data, cat,
            os.path.join(SWEEP_PLOT_DIR, f"{slug}.png"),
        )

    os.makedirs(SWEEP_REPORT_DIR, exist_ok=True)
    path = os.path.join(SWEEP_REPORT_DIR, "damage_sweep.txt")
    with open(path, "w") as f:
        f.write(report(data) + "\n")
    print(f"   report -> {path}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Injector v2 — damage sweep aggregate.").parse_args()
    aggregate_sweep_phase()
