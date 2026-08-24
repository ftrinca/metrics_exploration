import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

import core.dataset_io as dataset_io
from core.missingness_patterns import make_mask

from injector import distortions as D
from injector.calibrate import solve_series
from injector.build import load_ground_truth
from injector.config import (
    DAMAGE_LEVELS, DISTORTION_NAMES, SEED, SWEEP_PATTERN, SWEEP_RATE, sweep_dir,
)


def build_one(name, y_true, mask, force=False):
    """Build one distortion at every level of config.DAMAGE_LEVELS.

    The severity is solved separately at each level, so all eight distortions
    share one damage axis. Severities are stored alongside the data.
    """
    path = os.path.join(sweep_dir(name), "data.json")
    if not force and os.path.exists(path):
        print(f"   SKIP (already built): {path}")
        return

    out = {
        "y_true": dataset_io.matrix_to_lists(y_true),
        "mask": dataset_io.bool_matrix_to_mask(mask),
        "levels": DAMAGE_LEVELS,
        "severity": {},
        "achieved": {},
    }

    for level_idx, target in enumerate(DAMAGE_LEVELS, start=1):
        severities, achieved = {}, []
        for series_idx in range(y_true.shape[1]):
            idx = np.where(mask[:, series_idx])[0]
            if idx.size == 0:
                continue
            y_col = y_true[:, series_idx]
            if D.gap_sigma(y_col, idx) <= 0.0:
                continue
            sev, got, _ = solve_series(
                name, y_col, idx, SEED + 1000 + series_idx, target)
            severities[series_idx] = sev
            achieved.append(got)

        distorted = D.apply_one(y_true, mask, name, severities, SEED)
        key = f"L{level_idx}"
        out[key] = dataset_io.matrix_to_lists(distorted)
        out["severity"][key] = {str(k): v for k, v in severities.items()}
        out["achieved"][key] = float(np.mean(achieved)) if achieved else None
        gap = "" if not achieved else f"  (achieved {np.mean(achieved):.3f})"
        print(f"     level {level_idx}: target {target:.2f}{gap}")

    dataset_io.save_dataset(path, out)
    print(f"   built -> {path}")


def build_sweep_phase(names, force=False):
    """Build the damage sweep for every distortion, on one fixed missingness scenario."""
    print("Loading ground truth")
    y_true = load_ground_truth()
    print(f"  shape={y_true.shape}")
    print(f"Building mask ({SWEEP_PATTERN}, rate={SWEEP_RATE:.0%})")
    mask = make_mask(y_true, SWEEP_PATTERN, SWEEP_RATE)
    print(f"  missing={mask.sum()}\n")

    for name in names:
        print(f"=== {name} " + "=" * 46)
        build_one(name, y_true, mask, force=force)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Injector v2 — damage sweep build.")
    ap.add_argument("--distortions", nargs="+", default=DISTORTION_NAMES, choices=DISTORTION_NAMES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_sweep_phase(a.distortions, force=a.force)
