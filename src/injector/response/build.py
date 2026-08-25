import argparse
import os

import numpy as np

import core.dataset_io as dataset_io
from core.missingness_patterns import make_mask

from injector import distortions as D
from injector.reactivity.calibrate import solve_series
from injector.reactivity.build import load_ground_truth
from injector.config import (
    DAMAGE_LEVELS, DISTORTION_NAMES, SEED, RESPONSE_PATTERN, RESPONSE_RATE, response_dir,
)


def build_one(name, y_true, mask, force=False):
    """Build one distortion at every level of config.DAMAGE_LEVELS."""
    path = os.path.join(response_dir(name), "data.json")
    if not force and os.path.exists(path):
        print(f"   SKIP (already built): {path}")
        return
    out = {
        "y_true": dataset_io.matrix_to_lists(y_true),
        "mask": dataset_io.bool_matrix_to_mask(mask),
        "levels": DAMAGE_LEVELS,
        "severity": {},
        "achieved": {},
        "reached": {},
    }

    for level_idx, target in enumerate(DAMAGE_LEVELS, start=1):
        severities, achieved, missed = {}, [], 0
        for series_idx in range(y_true.shape[1]):
            idx = np.where(mask[:, series_idx])[0]
            if idx.size == 0:
                continue
            y_col = y_true[:, series_idx]
            if D.gap_sigma(y_col, idx) <= 0.0:
                continue
            sev, got, ok = solve_series(
                name, y_col, idx, SEED + 1000 + series_idx, target)
            severities[series_idx] = sev
            achieved.append(got)
            missed += not ok

        distorted = D.apply_one(y_true, mask, name, severities, SEED)
        key = f"L{level_idx}"
        out[key] = dataset_io.matrix_to_lists(distorted)
        out["severity"][key] = {str(k): v for k, v in severities.items()}
        out["achieved"][key] = float(np.mean(achieved)) if achieved else None
        out["reached"][key] = len(achieved) - missed
        gap = "" if not achieved else f"  (achieved {np.mean(achieved):.3f})"
        # a level no distortion can reach is not equal damage, so say so loudly
        warn = f"   !! {missed}/{len(achieved)} series short of target" if missed else ""
        print(f"     level {level_idx}: target {target:.2f}{gap}{warn}")

    dataset_io.save_dataset(path, out)
    print(f"   built -> {path}")


def build_phase(names, force=False):
    """Build the damage-response curve for every distortion, on one fixed scenario."""
    print("Loading ground truth")
    y_true = load_ground_truth()
    print(f"  shape={y_true.shape}")
    print(f"Building mask ({RESPONSE_PATTERN}, rate={RESPONSE_RATE:.0%})")
    mask = make_mask(y_true, RESPONSE_PATTERN, RESPONSE_RATE)
    print(f"  missing={mask.sum()}\n")

    for name in names:
        print(f"=== {name} " + "=" * 46)
        build_one(name, y_true, mask, force=force)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Damage response — build phase.")
    ap.add_argument("--distortions", nargs="+", default=DISTORTION_NAMES, choices=DISTORTION_NAMES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_phase(a.distortions, force=a.force)
