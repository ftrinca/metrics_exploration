"""Background figures — the introduction's disagreement pair and the
missingness-pattern panels.

Both draw from the algorithm-ranking caches rather than from ImputeGAP
directly, so they reproduce from what is already on disk: the truth comes
from a deterministic.json and the pattern panels show the masks the
experiments actually ran on.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from core.metrics import dtw, mae

from experiments.algorank.cache import deterministic_path
from experiments.algorank.config import PATTERNS

from paths import PLOTS_DIR

PLOT_DIR = os.path.join(PLOTS_DIR, "background")

# The disagreement pair: one gap, one series, two reconstructions built so
# that each metric prefers a different one. The smoothing window is wide
# enough to erase the oscillation inside the gap and the lag small enough to
# keep the shape, which is the whole point of the figure.
DISAGREEMENT_DATASET = "temperature"
DISAGREEMENT_SERIES = 0
GAP_LENGTH = 80
SMOOTH_WINDOW = 41
LAG_STEPS = 5

# The pattern panels: the first four series of the same dataset under the
# cached 20% masks of all three patterns.
PATTERN_RATE = 0.2
PATTERN_SERIES = 4


def _truth(dataset: str, pattern: str = "mcar", rate: float = 0.1) -> np.ndarray:
    with open(deterministic_path(dataset, pattern, rate)) as f:
        return np.array(json.load(f)["y_true"])


def _mask(dataset: str, pattern: str, rate: float) -> np.ndarray:
    with open(deterministic_path(dataset, pattern, rate)) as f:
        return np.array(json.load(f)["mask"]).astype(bool)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   {path}")


def disagreement_pair() -> None:
    """The two reconstructions of one gap that two metrics order oppositely."""
    y = _truth(DISAGREEMENT_DATASET)[DISAGREEMENT_SERIES]
    start = len(y) // 3
    gap = slice(start, start + GAP_LENGTH)

    smoothed = y.copy()
    smoothed[gap] = uniform_filter1d(y, size=SMOOTH_WINDOW)[gap]
    lagged = y.copy()
    lagged[gap] = y[start - LAG_STEPS:start + GAP_LENGTH - LAG_STEPS]

    window = slice(max(0, start - 60), min(len(y), start + GAP_LENGTH + 60))
    t = np.arange(window.start, window.stop)
    for name, recon in (("smoothed", smoothed), ("lagged", lagged)):
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.axvspan(start, start + GAP_LENGTH - 1, color="grey", alpha=0.15, lw=0)
        ax.plot(t, y[window], color="0.45", lw=1.1, label="ground truth")
        ax.plot(t, recon[window], color="#C44E52", lw=1.2, label=name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _save(fig, os.path.join(PLOT_DIR, f"disagreement_{name}.png"))

    # The figure's claim, printed so a changed window or lag cannot silently
    # break it: the smoothed reconstruction wins on MAE, the lagged one on DTW.
    print(f"   MAE  smoothed {mae(y[gap], smoothed[gap]):.3f}  "
          f"lagged {mae(y[gap], lagged[gap]):.3f}")
    print(f"   DTW  smoothed {dtw(y, smoothed):.3f}  lagged {dtw(y, lagged):.3f}")


def pattern_panels() -> None:
    """The first four series under the cached masks of the three patterns."""
    for pattern in PATTERNS:
        try:
            y = _truth(DISAGREEMENT_DATASET, pattern, PATTERN_RATE)
            m = _mask(DISAGREEMENT_DATASET, pattern, PATTERN_RATE)
        except FileNotFoundError as exc:
            print(f"   SKIP {pattern}: {exc}")
            continue
        fig, axes = plt.subplots(PATTERN_SERIES, 1, figsize=(4.2, 4.2),
                                 sharex=True)
        window = slice(0, min(400, y.shape[1]))
        t = np.arange(window.start, window.stop)
        for s, ax in enumerate(axes):
            ax.plot(t, y[s][window], color="0.45", lw=0.8)
            missing = np.full(window.stop - window.start, np.nan)
            missing[m[s][window]] = y[s][window][m[s][window]]
            ax.plot(t, missing, color="#C44E52", lw=1.4)
            ax.set_yticks([])
        axes[-1].set_xlabel("timestep", fontsize=8)
        _save(fig, os.path.join(PLOT_DIR, f"pattern_{pattern}.png"))


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Background figures — disagreement pair and patterns.").parse_args()
    disagreement_pair()
    pattern_panels()
