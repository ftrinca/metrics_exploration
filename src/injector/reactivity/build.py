import argparse
import json
import os

import numpy as np

import core.dataset_io as dataset_io
from core.data import normalization, real_world_ground_truth
from core.missingness_patterns import make_mask

from injector import distortions as D
from injector.config import (
    DATASET, DISTORTION_NAMES, N_SERIES, NORMALIZATION, PATTERNS, RATES, SEED,
    rate_dir,
)


def load_ground_truth() -> np.ndarray:
    """Load and normalise the configured dataset."""
    y_true = real_world_ground_truth.generate(DATASET, N_SERIES)
    return normalization.apply_normalization(y_true, NORMALIZATION)


def _load_calibration(pattern: str, rate: float) -> dict:
    """Read the severities calibrate.py solved for one (pattern, rate)."""
    path = os.path.join(rate_dir(pattern, rate), "calibration.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run injector/calibrate.py for "
            f"pattern={pattern!r} rate={rate} first."
        )
    with open(path) as f:
        return json.load(f)["distortions"]


def build_one(pattern: str, rate: float, y_true: np.ndarray, force: bool = False) -> None:
    """Apply all eight distortions at their solved severities and cache the result."""
    data_path = os.path.join(rate_dir(pattern, rate), "data.json")
    if not force and os.path.exists(data_path):
        print(f"   SKIP (already built): {data_path}")
        return

    mask = make_mask(y_true, pattern, rate)
    calib = _load_calibration(pattern, rate)

    json_out = {
        "y_true": dataset_io.matrix_to_lists(y_true),
        "mask": dataset_io.bool_matrix_to_mask(mask),
    }
    for name in DISTORTION_NAMES:
        severities = {int(k): v for k, v in calib[name]["severity"].items()}
        distorted = D.apply_one(y_true, mask, name, severities, SEED)
        json_out[name] = dataset_io.matrix_to_lists(distorted)

    dataset_io.save_dataset(data_path, json_out)
    print(f"   built -> {data_path}")


def build_phase(patterns, rates, force=False):
    """Build every (pattern, rate) of the damage-reactivity experiment."""
    print(f"Loading ground truth: {DATASET} ({N_SERIES} series)")
    y_true = load_ground_truth()
    print(f"  shape={y_true.shape}  nan={np.isnan(y_true).sum()}\n")

    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        for rate in rates:
            print(f"  -- rate {rate:.0%} --")
            build_one(pattern, rate, y_true, force=force)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Damage reactivity — build phase.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    ap.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_phase(a.patterns, a.rates, force=a.force)
