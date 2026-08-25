import argparse
import json
import os

import numpy as np

from core.metric_config import METRIC_LIST
from core.buckets import subjects_present
from core.scoring import compute_all_scores

from algo_ranking import cache
from algo_ranking.algorithms import ALGORITHMS
from algo_ranking.config import (
    ALGO_METRICS, DATASETS, N_SEEDS, PATTERNS, RATES, rate_dir,
)


def _average_scores(
    per_seed_scores: list[dict[str, dict[str, float | None]]],
) -> dict[str, dict[str, float | None]]:
    """Average across seeds per (metric, algorithm)."""
    metrics = [k for k in METRIC_LIST if k in per_seed_scores[0]]
    by_seed = dict(enumerate(per_seed_scores))
    algos = subjects_present(by_seed, list(by_seed), metrics,
                             [name for name, _, _ in ALGORITHMS])

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
    """Score one scenario from its cached builds, returning {metric: {algo: score}}."""
    scores_path = os.path.join(rate_dir(dataset, pattern, rate), "scores.json")
    if not force and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    per_seed_scores = []
    for seed in range(N_SEEDS):
        built = cache.load_scenario(dataset, pattern, rate, seed)
        scores = compute_all_scores(built["y_true"], cache.reconstructions(built),
                                    mask=built["mask"])
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
    """Return the scores for one scenario, computing whatever is missing."""
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
        built = cache.load_scenario(dataset, pattern, rate, seed)
        per_seed.append(
            compute_all_scores(built["y_true"], cache.reconstructions(built),
                               mask=built["mask"], metric_names=absent)
        )

    scores.update(_average_scores(per_seed))
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    return scores


def score_phase(
    datasets: list[str], patterns: list[str], rates: list[float], force: bool = False,
) -> None:
    """Score every (dataset, pattern, rate) from cached builds."""
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
    parser = argparse.ArgumentParser(description="Algorithm ranking — score phase.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    score_phase(args.datasets, args.patterns, args.rates, force=args.force)
