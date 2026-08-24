"""Algorithm Ranking build phase: run every algorithm on every (dataset,
pattern, rate) and cache the raw reconstructions to disk.

The deterministic algorithms and the mask are built once per (dataset, pattern,
rate) and shared by every seed, so that the mask cannot silently differ between
seeds and the algorithms whose output cannot change are not recomputed for each
one. No metrics are computed here, which is what lets this phase, the expensive
one, be resumed independently of scoring.

core's generators and make_mask work in (n_timesteps, n_series), so everything
is transposed once to ImputeGAP-native (n_series, n_timesteps) before it
reaches algorithms.build, which passes its input straight to ImputeGAP.

Usage:
  python algo_ranking/build.py
  python algo_ranking/build.py --datasets climate --patterns mcar --rates 0.2 0.5   # subset, for quick checks
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

import core.dataset_io as dataset_io
from core.data import normalization, real_world_ground_truth
from core.missingness_patterns import make_mask

from algo_ranking import algorithms
from algo_ranking.config import (
    DATASETS, MAX_TIMESTEPS, N_SEEDS, N_SERIES, NORMALIZATION, PATTERNS,
    RATES, rate_dir, seed_dir,
)


def _run_algorithms_isolated(
    y_true_t: np.ndarray, mask_t: np.ndarray, seed: int, algo_names: set[str],
) -> dict[str, np.ndarray]:
    """Run each algorithm in algo_names in its own fresh subprocess, one at a
    time, and return {algo_name: array(n_series, n_timesteps)}.

    An algorithm whose subprocess crashed or failed is absent from the result,
    the same contract algorithms.build() offers. See algo_ranking/_run_algorithm.py
    for why the isolation is needed.

    algo_names is an unordered set, so it is iterated in ALGORITHMS' order to
    keep printing and dict ordering stable across runs.
    """
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
                    sys.executable, "-u", "-m", "algo_ranking._run_algorithm",
                    "--input", input_path, "--algo", name, "--seed", str(seed),
                    "--output", output_path,
                ],
                cwd=SRC,
            )
            elapsed = time.perf_counter() - t0
            if proc.returncode != 0:
                # A subprocess killed by a segfault or abort never reaches
                # algorithms.build()'s own ERROR line, so the failure has to be
                # reported from this side instead.
                print(f"{name}: subprocess exited abnormally (code {proc.returncode}) "
                      f"after {elapsed:.1f}s - excluded from this run")
                continue
            with open(output_path) as f:
                out = json.load(f)
            if "result" in out:
                results[name] = np.array(out["result"])
            # A missing "result" key means the algorithm raised inside its own
            # subprocess, which already printed the error on the inherited stdout.

    return results


def load_ground_truth(dataset: str) -> np.ndarray:
    """Load, truncate to MAX_TIMESTEPS, and normalize one dataset. Returns
    (n_timesteps, n_series)."""
    y_true = real_world_ground_truth.generate(dataset, N_SERIES)
    if y_true.shape[0] > MAX_TIMESTEPS:
        y_true = y_true[:MAX_TIMESTEPS]
    return normalization.apply_normalization(y_true, NORMALIZATION)


def _deterministic_cache_path(dataset: str, pattern: str, rate: float) -> str:
    return os.path.join(rate_dir(dataset, pattern, rate), "deterministic.json")


def build_deterministic(
    dataset: str, pattern: str, rate: float, y_true: np.ndarray, force: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build, or load from deterministic.json, the mask and the
    DETERMINISTIC_ALGORITHMS reconstructions shared by every seed of one
    (dataset, pattern, rate).

    Caching the mask is what guarantees every seed of a scenario uses the same
    one even across separate process invocations, which a fresh make_mask() per
    seed would not, since make_mask takes no seed.

    Returns (mask, reconstructions). mask is (n_timesteps, n_series), matching
    y_true and make_mask, whether it was just drawn or came from the cache;
    reconstructions is {algo_name: array(n_series, n_timesteps)}, ready to merge
    with algorithms.build()'s stochastic output in build_one.
    """
    det_path = _deterministic_cache_path(dataset, pattern, rate)
    if not force and os.path.exists(det_path):
        with open(det_path) as f:
            cached = json.load(f)
        # dataset_io stores natively as (n_series, n_timesteps) whatever
        # orientation it was handed, so the mask needs one transpose to get back
        # to the (n_timesteps, n_series) callers expect. The reconstructions do
        # not, because native orientation is what callers want for those.
        mask = np.array(cached["mask"]).T
        assert mask.shape == y_true.shape, (
            f"cached mask shape {mask.shape} != y_true shape {y_true.shape} "
            f"in {det_path} - stale cache from a different dataset config?"
        )
        reconstructions = {name: np.array(mat) for name, mat in cached.items() if name != "mask"}
        print(f"   SKIP (already built): {det_path}")
        return mask, reconstructions

    mask = make_mask(y_true, pattern, rate)
    y_true_t = y_true.T
    mask_t = mask.T
    reconstructions = _run_algorithms_isolated(
        y_true_t, mask_t, seed=0, algo_names=algorithms.DETERMINISTIC_ALGORITHMS,
    )

    json_out = {"mask": dataset_io.bool_matrix_to_mask(mask)}
    for name, mat in reconstructions.items():
        json_out[name] = dataset_io.matrix_to_lists(mat.T)
    dataset_io.save_dataset(det_path, json_out)
    print(f"   built (mask + deterministic algorithms, shared across all seeds) -> {det_path}")
    return mask, reconstructions


def build_one(
    dataset: str, pattern: str, rate: float, seed: int, y_true: np.ndarray, mask: np.ndarray,
    deterministic: dict[str, np.ndarray], force: bool = False,
) -> None:
    """Draw this seed's STOCHASTIC_ALGORITHMS reconstructions, merge them with
    the shared deterministic ones, and write {y_true, mask, **reconstructions}
    to that seed's data.json. Skips an existing file unless force=True."""
    data_path = os.path.join(seed_dir(dataset, pattern, rate, seed), "data.json")
    if not force and os.path.exists(data_path):
        print(f"   SKIP (already built): {data_path}")
        return

    y_true_t = y_true.T
    mask_t = mask.T

    stochastic = _run_algorithms_isolated(
        y_true_t, mask_t, seed=seed, algo_names=algorithms.STOCHASTIC_ALGORITHMS,
    )
    reconstructions = {**deterministic, **stochastic}

    json_out = {
        "y_true": dataset_io.matrix_to_lists(y_true),
        "mask": dataset_io.bool_matrix_to_mask(mask),
    }
    for name, mat in reconstructions.items():
        json_out[name] = dataset_io.matrix_to_lists(mat.T)
    dataset_io.save_dataset(data_path, json_out)
    print(f"   built -> {data_path}")


def build_rate(dataset: str, pattern: str, rate: float, force: bool = False) -> None:
    """Build one full (dataset, pattern, rate) unit: ground truth, then the
    shared mask and deterministic reconstructions, then each seed."""
    y_true = load_ground_truth(dataset)
    mask, deterministic = build_deterministic(dataset, pattern, rate, y_true, force=force)
    for seed in range(N_SEEDS):
        build_one(dataset, pattern, rate, seed, y_true, mask, deterministic, force=force)


def build_phase(
    datasets: list[str], patterns: list[str], rates: list[float], force: bool = False,
) -> None:
    """Build every (dataset, pattern, rate) unit, caching to disk as it goes.

    Safe to call repeatedly with different subsets across separate process
    invocations, because an already-cached combination is skipped.
    """
    for dataset in datasets:
        for pattern in patterns:
            for rate in rates:
                print(f"=== {dataset} / {pattern} / {rate:.0%} " + "=" * 30)
                build_rate(dataset, pattern, rate, force=force)
                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm Ranking (Part 2) — build phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    build_phase(args.datasets, args.patterns, args.rates, force=args.force)
