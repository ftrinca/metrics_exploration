import argparse
import json
import os

import numpy as np

from metric_eval.core.scoring import compute_all_scores

from metric_eval.experiments.algorank import cache
from metric_eval.experiments.algorank.config import (ALGO_METRICS, DATASETS, N_SEEDS, PATTERNS,
                                         RATES, STOCHASTIC_ALGO_NAMES, rate_dir)

from metric_eval.experiments.cis.config import ALGO_NAMES, CIS_CACHE_DIR, REFERENCE_NAME, cache_path


def mean_reconstruction(y_true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Every masked position set to the mean of its own series' observed values.

    This is the reference every CIS component is measured against: it needs no
    model, it exists in every scenario, and it is the reconstruction to fall
    back on when nothing else is available.
    """
    out = y_true.copy().astype(np.float64)
    for s in range(y_true.shape[0]):
        observed = y_true[s][~mask[s]]
        out[s][mask[s]] = float(np.mean(observed)) if observed.size else 0.0
    return out


def _std_ratio(reconstruction: np.ndarray, y_true: np.ndarray,
               mask: np.ndarray) -> float:
    """Spread of the reconstructed values over the spread of the truth, both at
    the masked positions.

    Experiment 2 measures degeneracy this way, so the gate and the ranking
    chapter describe a reconstruction with one number.
    """
    return float(np.std(reconstruction[mask]) / (np.std(y_true[mask]) + 1e-12))


def _scenario_scores(dataset: str, pattern: str, rate: float) -> dict:
    with open(os.path.join(rate_dir(dataset, pattern, rate), "scores.json")) as f:
        return json.load(f)


def build_scenario(dataset: str, pattern: str, rate: float) -> dict:
    """Reference scores and standard-deviation ratios of one scenario."""
    per_seed_ratio: dict[str, list[float]] = {}
    per_seed_unique: dict[str, list[float]] = {}
    reference = None

    for seed in range(N_SEEDS):
        built = cache.load_scenario(dataset, pattern, rate, seed)
        y_true, mask = built["y_true"], built["mask"].astype(bool)

        if reference is None:
            ref = mean_reconstruction(y_true, mask)
            scored = compute_all_scores(
                y_true, {REFERENCE_NAME: ref, "TRUTH": y_true},
                mask=mask, metric_names=list(ALGO_METRICS),
            )
            reference = {
                "scores": {m: scored[m][REFERENCE_NAME] for m in ALGO_METRICS},
                "mi_self": scored["mi"]["TRUTH"],
                "r2_ceiling": scored["r2"]["TRUTH"],
                "n_timesteps": int(y_true.shape[-1]),
            }

        for name in ALGO_NAMES:
            if name not in built:
                continue
            if name not in STOCHASTIC_ALGO_NAMES and per_seed_ratio.get(name):
                continue
            per_seed_ratio.setdefault(name, []).append(
                _std_ratio(built[name], y_true, mask))
            per_seed_unique.setdefault(name, []).append(
                float(np.unique(built[name][mask]).size))

    return {
        "reference": reference,
        "std_ratio": {a: float(np.mean(v)) for a, v in per_seed_ratio.items()},
        "n_unique": {a: float(np.mean(v)) for a, v in per_seed_unique.items()},
        "scores": _scenario_scores(dataset, pattern, rate),
    }


def build_dataset(dataset: str, patterns: list[str], rates: list[float],
                  force: bool = False) -> dict:
    """Every scenario of one dataset, merged into that dataset's cache file.

    Scenarios already in the file are kept, so an interrupted build resumes and a
    subset can be rebuilt on its own.
    """
    path = cache_path(dataset)
    out = {}
    if os.path.exists(path) and not force:
        with open(path) as f:
            out = json.load(f)
    for pattern in patterns:
        for rate in rates:
            key = f"{pattern}|{round(rate * 100):02d}"
            if key in out:
                continue
            out[key] = build_scenario(dataset, pattern, rate)
            print(f"  {dataset} {key}")
    os.makedirs(CIS_CACHE_DIR, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"written: {path}  ({len(out)} scenarios)")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 3 (CIS) — derive the reference and the gate ratios.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in args.datasets:
        build_dataset(name, args.patterns, args.rates, force=args.force)
