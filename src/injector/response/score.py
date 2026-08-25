import argparse
import json
import os

import numpy as np

from core.scoring import compute_all_scores
from injector.config import DAMAGE_LEVELS, DISTORTION_NAMES, response_dir


def score_one(name, force=False):
    """Score every damage level of one distortion, caching to scores.json."""
    data_path = os.path.join(response_dir(name), "data.json")
    scores_path = os.path.join(response_dir(name), "scores.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing {data_path} — run python -m injector.response.build first.")
    if not force and os.path.exists(scores_path):
        with open(scores_path) as f:
            return json.load(f)

    with open(data_path) as f:
        built = json.load(f)

    y_true = np.array(built["y_true"])
    mask = np.array(built["mask"])
    levels = {f"L{i}": np.array(built[f"L{i}"])
              for i in range(1, len(DAMAGE_LEVELS) + 1) if f"L{i}" in built}

    scores = compute_all_scores(y_true, levels, mask=mask)
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"   scored -> {scores_path}")
    return scores


def score_phase(names, force=False):
    """Score every damage level of every distortion."""
    for name in names:
        print(f"=== {name} " + "=" * 46)
        score_one(name, force=force)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Damage response — score phase.")
    ap.add_argument("--distortions", nargs="+", default=DISTORTION_NAMES, choices=DISTORTION_NAMES)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    score_phase(a.distortions, force=a.force)
