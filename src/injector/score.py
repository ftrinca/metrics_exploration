import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from core.scoring import compute_all_scores

from injector.config import PATTERNS, RATES, rate_dir


def _load_built(pattern: str, rate: float) -> dict:
    """Read the cache build.py wrote for one (pattern, rate)."""
    path = os.path.join(rate_dir(pattern, rate), "data.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path} — run injector/build.py for "
            f"pattern={pattern!r} rate={rate} first."
        )
    with open(path) as f:
        return json.load(f)


def score_one(pattern: str, rate: float, force: bool = False) -> dict:
    """Score every distortion of one (pattern, rate), caching to scores.json.

    Anything read back from data.json is already in (n_series, n_timesteps)
    orientation and must not be transposed again.
    """
    scores_path = os.path.join(rate_dir(pattern, rate), "scores.json")
    if not force and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    built = _load_built(pattern, rate)
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


def score_phase(patterns, rates, force=False):
    """Score every (pattern, rate) of the equal-damage experiment."""
    for pattern in patterns:
        print(f"=== pattern: {pattern} " + "=" * 46)
        for rate in rates:
            print(f"  -- rate {rate:.0%} --")
            score_one(pattern, rate, force=force)
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Injector v2 — score phase.")
    ap.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    ap.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    score_phase(a.patterns, a.rates, force=a.force)
