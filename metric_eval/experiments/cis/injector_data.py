import json
import os

import numpy as np

import metric_eval.experiments.injector.config as injector_config
from metric_eval.experiments.algorank.config import ALGO_METRICS
from metric_eval.core.scoring import compute_all_scores

from metric_eval.experiments.cis.config import REFERENCE_NAME
from metric_eval.experiments.cis.build import mean_reconstruction


def _condition(folder: str) -> dict | None:
    """Reference scores, distorted scores and spread ratios of one Injector condition."""
    data_path = os.path.join(folder, "data.json")
    scores_path = os.path.join(folder, "scores.json")
    if not (os.path.exists(data_path) and os.path.exists(scores_path)):
        return None

    with open(data_path) as f:
        data = json.load(f)
    with open(scores_path) as f:
        scores = json.load(f)

    y_true = np.array(data["y_true"])
    mask = np.array(data["mask"]).astype(bool)
    reference = mean_reconstruction(y_true, mask)
    scored = compute_all_scores(y_true, {REFERENCE_NAME: reference, "TRUTH": y_true},
                                mask=mask, metric_names=list(ALGO_METRICS))
    subjects = [k for k in data
                if k not in ("y_true", "mask")
                and np.ndim(data[k]) == y_true.ndim]
    return {
        "reference": {
            "scores": {m: scored[m][REFERENCE_NAME] for m in ALGO_METRICS},
            "mi_self": scored["mi"]["TRUTH"],
            "r2_ceiling": scored["r2"]["TRUTH"],
        },
        "scores": scores,
        "std_ratio": {
            s: float(np.std(np.array(data[s])[mask]) / (np.std(y_true[mask]) + 1e-12))
            for s in subjects
        },
        "subjects": subjects,
    }


def load_equal_damage() -> dict[str, dict]:
    """Chapter 4's equal-damage run, one entry per missingness condition.

    Every distortion in a condition is solved to the same pointwise damage, so
    the pointwise component is pinned and whatever spread is left in CIS is
    spread in the kind of damage.
    """
    out = {}
    for pattern in injector_config.PATTERNS:
        for rate in injector_config.RATES:
            payload = _condition(injector_config.rate_dir(pattern, rate))
            if payload is not None:
                out[f"{pattern}/{round(rate * 100):02d}pct"] = payload
    return out


def load_damage_sweep() -> dict[str, dict]:
    """Chapter 4's seven-level sweep, one entry per distortion."""
    out = {}
    for distortion in injector_config.DISTORTION_NAMES:
        payload = _condition(injector_config.response_dir(distortion))
        if payload is not None:
            with open(os.path.join(injector_config.response_dir(distortion),
                                   "data.json")) as f:
                payload["damage_levels"] = json.load(f).get("levels")
            out[distortion] = payload
    return out
