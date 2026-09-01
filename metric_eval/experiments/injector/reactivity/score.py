import argparse
import json
import os

import numpy as np

from metric_eval.core.scoring import compute_all_scores

from metric_eval.experiments.injector.config import PATTERNS, RATES, pass_filename, rate_dir


def _load_built(pattern: str, rate: float, damage_metric: str) -> dict:
    """Read the cache build.py wrote for one (pattern, rate, target)."""
    path = os.path.join(rate_dir(pattern, rate),
                        pass_filename("data.json", damage_metric))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run injector/build.py for "
            f"pattern={pattern!r} rate={rate} "
            f"(--damage-metric {damage_metric}) first."
        )
    with open(path) as f:
        return json.load(f)


def score_one(pattern: str, rate: float, force: bool = False,
              damage_metric: str = "mae") -> dict:
    """Score every distortion of one (pattern, rate), caching to scores.json."""
    scores_path = os.path.join(rate_dir(pattern, rate),
                               pass_filename("scores.json", damage_metric))
    if not force and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    built = _load_built(pattern, rate, damage_metric)
    y_true = np.array(built["y_true"])
    mask = np.array(built["mask"])
    reconstructions = {
        name: np.array(built[name])
        for name in built if name not in ("y_true", "mask")
    }
    scores = compute_all_scores(y_true, reconstructions, mask=mask)

    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"   scored -> {scores_path}")
    return scores


def score_phase(patterns, rates, force=False, damage_metric="mae"):
    """Score every (pattern, rate) of the damage-reactivity experiment."""
    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        for rate in rates:
            print(f"  -- rate {rate:.0%} --")
            score_one(pattern, rate, force=force, damage_metric=damage_metric)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Damage reactivity — score phase.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    ap.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    ap.add_argument("--damage-metric", default="mae", choices=("mae", "rmse"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    score_phase(a.patterns, a.rates, force=a.force, damage_metric=a.damage_metric)
