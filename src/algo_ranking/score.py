import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from core.metric_config import METRIC_LIST
from core.scoring import compute_all_scores

from algo_ranking.algorithms import ALGORITHMS
from algo_ranking.config import (
    ALGO_METRICS, DATASETS, N_SEEDS, PATTERNS, RATES, rate_dir, seed_dir,
)


def _load_built(dataset: str, pattern: str, rate: float, seed: int) -> dict:
    """Read one seed's cached reconstructions."""
    data_path = os.path.join(seed_dir(dataset, pattern, rate, seed), "data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing {data_path} - run algo_ranking/build.py for "
            f"dataset={dataset!r} pattern={pattern!r} rate={rate} seed={seed} first."
        )
    with open(data_path) as f:
        return json.load(f)


def _average_scores(
    per_seed_scores: list[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | None]]:
    """Average across seeds per (metric, algorithm).

    None means the algorithm failed on that draw and is left out of the mean; an
    algorithm that failed on every draw averages to None. The algorithm list is
    the union across seeds, in ALGORITHMS order, so a stochastic algorithm that
    failed on one draw is not dropped entirely.
    """
    present: set[str] = set()
    for s in per_seed_scores:
        for per_algo in s.values():
            present.update(per_algo.keys())
    algos = [name for name, _, _ in ALGORITHMS if name in present]

    metrics = [k for k in METRIC_LIST if k in per_seed_scores[0]]

    averaged: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        averaged[metric] = {}
        for algo in algos:
            vals = [
                s[metric][algo] for s in per_seed_scores
                if s[metric].get(algo) is not None
            ]
            averaged[metric][algo] = float(np.mean(vals)) if vals else None
    return averaged


def score_one(dataset: str, pattern: str, rate: float, force: bool = False) -> dict:
    """Score one scenario from its cached builds, returning {metric: {algo: score}}.

    Every metric is stored, not only the ones config.ALGO_CATEGORIES selects, so
    a later change to the selection is an aggregate pass rather than a re-score.
    """
    scores_path = os.path.join(rate_dir(dataset, pattern, rate), "scores.json")
    if not force and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    per_seed_scores = []
    for seed in range(N_SEEDS):
        built = _load_built(dataset, pattern, rate, seed)
        y_true_t = np.array(built["y_true"])
        mask_t = np.array(built["mask"])
        reconstructions = {
            name: np.array(built[name])
            for name in built if name not in ("y_true", "mask")
        }
        scores = compute_all_scores(y_true_t, reconstructions, mask=mask_t)
        per_seed_scores.append(scores)

    averaged = _average_scores(per_seed_scores)

    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(averaged, f, indent=2)
    print(f"   scored -> {scores_path}  (averaged over {N_SEEDS} seeds)")
    return averaged


def missing_metrics(scores: dict) -> list[str]:
    """Which of the currently selected metrics are absent from a cached score file."""
    return [m for m in ALGO_METRICS if m not in scores]


def ensure_scored(dataset: str, pattern: str, rate: float) -> dict:
    """Return the scores for one scenario, computing whatever is missing.

    A cache written before the metric set changed has only the absent metrics
    computed and merged in, so changing one metric does not cost a full
    re-score.
    """
    scores_path = os.path.join(rate_dir(dataset, pattern, rate), "scores.json")
    if not os.path.exists(scores_path):
        return score_one(dataset, pattern, rate)

    with open(scores_path) as f:
        scores = json.load(f)

    absent = missing_metrics(scores)
    if not absent:
        return scores

    print(f"   topping up {rate_dir(dataset, pattern, rate)}: {', '.join(absent)}")
    per_seed: list[dict] = []
    for seed in range(N_SEEDS):
        built = _load_built(dataset, pattern, rate, seed)
        y_true_t = np.array(built["y_true"])
        mask_t = np.array(built["mask"])
        reconstructions = {
            name: np.array(built[name])
            for name in built if name not in ("y_true", "mask")
        }
        per_seed.append(
            compute_all_scores(y_true_t, reconstructions, mask=mask_t, metric_names=absent)
        )

    scores.update(_average_scores(per_seed))
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def score_phase(
    datasets: list[str], patterns: list[str], rates: list[float], force: bool = False,
) -> None:
    """Score every (dataset, pattern, rate) from cached builds.

    Without force this goes through ensure_scored, so a cache predating a change
    to config.ALGO_CATEGORIES is topped up instead of being skipped as complete.
    """
    for dataset in datasets:
        print(f"=== dataset: {dataset} " + "=" * 50)
        for pattern in patterns:
            print(f"  -- pattern: {pattern} --")
            for rate in rates:
                print(f"    -- rate {rate:.0%} --")
                if force:
                    score_one(dataset, pattern, rate, force=True)
                else:
                    ensure_scored(dataset, pattern, rate)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm Ranking (Part 2) — score phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    score_phase(args.datasets, args.patterns, args.rates, force=args.force)
