"""The distortion panels of the thesis: what each distortion puts into a gap.

One small plot per distortion, drawn from the built cache of one condition,
so all eight show the same series, the same gap and the calibrated severity
that gives every distortion the same pointwise damage.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metric_eval.experiments.injector.config import (DISTORTION_NAMES, PLOT_DIR, RESPONSE_PATTERN,
                             RESPONSE_RATE, rate_dir)

PANEL_DIR = os.path.join(PLOT_DIR, "panels")

# The condition and series the thesis shows; the sweep uses the same one.
PANEL_PATTERN = RESPONSE_PATTERN
PANEL_RATE = RESPONSE_RATE
PANEL_SERIES = 0

# Timesteps of observed context shown on either side of the gap.
CONTEXT = 80


def _load_condition() -> dict:
    path = os.path.join(rate_dir(PANEL_PATTERN, PANEL_RATE), "data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run injector/build.py for "
            f"pattern={PANEL_PATTERN!r} rate={PANEL_RATE} first.")
    with open(path) as f:
        return json.load(f)


def plot_panels() -> None:
    """One panel per distortion: the truth in grey, the distorted
    reconstruction in red over the shaded gap."""
    built = _load_condition()
    # the cache stores (n_series, n_timesteps), one row per series
    y_true = np.array(built["y_true"])[PANEL_SERIES]
    mask = np.array(built["mask"])[PANEL_SERIES].astype(bool)

    gap = np.flatnonzero(mask)
    start, end = int(gap[0]), int(gap[-1])
    window = slice(max(0, start - CONTEXT), min(len(y_true), end + CONTEXT))
    t = np.arange(window.start, window.stop)

    for name in DISTORTION_NAMES:
        distorted = np.array(built[name])[PANEL_SERIES]
        # red only across the gap, anchored to the truth at its edges
        visible = np.full(len(y_true), np.nan)
        visible[mask] = distorted[mask]
        visible[max(0, start - 1)] = y_true[max(0, start - 1)]
        visible[min(len(y_true) - 1, end + 1)] = y_true[min(len(y_true) - 1, end + 1)]

        fig, ax = plt.subplots(figsize=(3.0, 2.2))
        ax.axvspan(start - 0.5, end + 0.5, color="grey", alpha=0.15, lw=0)
        ax.plot(t, y_true[window], color="0.55", lw=0.9)
        ax.plot(t, visible[window], color="#A32A31", lw=0.9)
        ax.tick_params(labelsize=7, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, alpha=0.25)
        os.makedirs(PANEL_DIR, exist_ok=True)
        path = os.path.join(PANEL_DIR, f"dist_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   plot -> {path}")


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="The eight distortion panels, from the built cache.").parse_args()
    plot_panels()
