import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import metric_eval.core.dataset_io as dataset_io
from core.data import real_world_ground_truth
from core.data import normalization
from metric_eval.core.missingness_patterns import make_mask

from experiments.algorank import cache
from experiments.algorank import algorithms
from experiments.algorank.config import (
    DATASETS, MAX_TIMESTEPS, N_SEEDS, N_SERIES, NORMALIZATION, PATTERNS,
    RATES,
)


def _run_algorithms_isolated(
    y_true_t: np.ndarray, mask_t: np.ndarray, seed: int, algo_names: set[str],
) -> dict[str, np.ndarray]:
    """Run each algorithm in its own subprocess and return {algo_name: array}."""
    ordered_names = [name for name, _, _ in algorithms.ALGORITHMS if name in algo_names]
    results: dict[str, np.ndarray] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        with open(input_path, "w") as f:
            json.dump({"y_true": y_true_t.tolist(), "mask": mask_t.tolist()}, f)

        for name in ordered_names:
            output_path = os.path.join(tmpdir, f"{name}.json")
            t0 = time.perf_counter()
            proc = subprocess.run(
                [
                    sys.executable, "-u", "-m", "metric_eval.experiments.algorank._run_algorithm",
                    "--input", input_path, "--algo", name, "--seed", str(seed),
                    "--output", output_path,
                ],
                cwd=SRC,
            )
            elapsed = time.perf_counter() - t0
            if proc.returncode != 0:
                print(f"{name}: subprocess exited abnormally (code {proc.returncode}) "
                      f"after {elapsed:.1f}s - excluded from this run")
                continue
            with open(output_path) as f:
                out = json.load(f)
            if "result" in out:
                results[name] = np.array(out["result"])

    return results


def load_ground_truth(dataset: str) -> np.ndarray:
    """Load, truncate to MAX_TIMESTEPS, and normalize one dataset, as (n_timesteps, n_series)."""
    y_true = real_world_ground_truth.generate(dataset, N_SERIES)
    if y_true.shape[0] > MAX_TIMESTEPS:
        y_true = y_true[:MAX_TIMESTEPS]
    return normalization.apply_normalization(y_true, NORMALIZATION)


def build_deterministic(
    dataset: str, pattern: str, rate: float, y_true: np.ndarray, force: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build, or load, the mask and DETERMINISTIC_ALGORITHMS reconstructions of one scenario.

    Caching the mask guarantees every seed of a scenario uses the same one, since make_mask takes no seed.
    Returns (mask, reconstructions), the mask as (n_timesteps, n_series) and the reconstructions in native (n_series, n_timesteps) orientation.
    """
    det_path = cache.deterministic_path(dataset, pattern, rate)
    if not force and os.path.exists(det_path):
        with open(det_path) as f:
            cached = json.load(f)
        mask = np.array(cached["mask"]).T
        assert mask.shape == y_true.shape, (
            f"cached mask shape {mask.shape} != y_true shape {y_true.shape} "
            f"in {det_path} - stale cache from a different dataset config?"
        )
        reconstructions = {name: np.array(mat) for name, mat in cached.items()
                           if name not in ("mask", "y_true")}
        print(f"   SKIP (already built): {det_path}")
        return mask, reconstructions

    mask = make_mask(y_true, pattern, rate)
    y_true_t = y_true.T
    mask_t = mask.T
    reconstructions = _run_algorithms_isolated(
        y_true_t, mask_t, seed=0, algo_names=algorithms.DETERMINISTIC_ALGORITHMS,
    )

    json_out = {
        "y_true": dataset_io.matrix_to_lists(y_true),
        "mask": dataset_io.bool_matrix_to_mask(mask),
    }
    for name, mat in reconstructions.items():
        json_out[name] = dataset_io.matrix_to_lists(mat.T)
    dataset_io.save_dataset(det_path, json_out)
    print(f"   built (mask + deterministic algorithms, shared across all seeds) -> {det_path}")
    return mask, reconstructions


def build_one(
    dataset: str, pattern: str, rate: float, seed: int, y_true: np.ndarray,
    mask: np.ndarray, force: bool = False,
) -> None:
    """Draw and cache this seed's stochastic reconstructions."""
    data_path = cache.seed_path(dataset, pattern, rate, seed)
    if not force and os.path.exists(data_path):
        print(f"   SKIP (already built): {data_path}")
        return

    stochastic = _run_algorithms_isolated(y_true.T, mask.T, seed=seed, algo_names=algorithms.STOCHASTIC_ALGORITHMS)

    json_out = {name: dataset_io.matrix_to_lists(mat.T) for name, mat in stochastic.items()}
    dataset_io.save_dataset(data_path, json_out)
    print(f"   built -> {data_path}")


def build_rate(dataset: str, pattern: str, rate: float, force: bool = False) -> None:
    """Build one full (dataset, pattern, rate) unit: ground truth, shared mask, then each seed."""
    y_true = load_ground_truth(dataset)
    mask, _ = build_deterministic(dataset, pattern, rate, y_true, force=force)
    for seed in range(N_SEEDS):
        build_one(dataset, pattern, rate, seed, y_true, mask, force=force)


def build_phase(
    datasets: list[str], patterns: list[str], rates: list[float], force: bool = False,
) -> None:
    """Build every (dataset, pattern, rate) unit, caching to disk as it goes."""
    for dataset in datasets:
        for pattern in patterns:
            for rate in rates:
                print(f"=== {dataset} / {pattern} / {rate:.0%} " + "=" * 30)
                build_rate(dataset, pattern, rate, force=force)
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm ranking — build phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_phase(args.datasets, args.patterns, args.rates, force=args.force)
