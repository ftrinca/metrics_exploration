import json
import os

import numpy as np

import injector.config as injector_config
from cis.gate import _iqr


def load_injector_reactivity() -> dict[str, dict]:
    """Read Experiment 1's damage-reactivity cache, or {} when it has not been run.

    Returns {"pattern/rate": {scores, iqr_ratio, n_timesteps}}. The IQR ratio is
    recomputed here, since the gate is CIS's own instrument.
    """
    out = {}
    for pattern in injector_config.PATTERNS:
        for rate in injector_config.RATES:
            folder = injector_config.rate_dir(pattern, rate)
            data_path = os.path.join(folder, "data.json")
            scores_path = os.path.join(folder, "scores.json")
            if not (os.path.exists(data_path) and os.path.exists(scores_path)):
                continue
            with open(data_path) as f:
                data = json.load(f)
            with open(scores_path) as f:
                scores = json.load(f)
            y_true = np.array(data["y_true"])
            mask = np.array(data["mask"]).astype(bool)
            true_iqr = _iqr(y_true[mask])
            names = [d for d in injector_config.DISTORTION_NAMES if d in data]
            out[f"{pattern}/{round(rate * 100):02d}pct"] = {
                "scores": scores,
                "n_timesteps": y_true.shape[-1],
                "iqr_ratio": {d: _iqr(np.array(data[d])[mask]) / (true_iqr + 1e-12)
                              for d in names},
            }
    return out


def load_injector_response() -> dict[str, dict]:
    """Read Experiment 1's damage-response curve, or {} when it has not been run."""
    out = {}
    for distortion in injector_config.DISTORTION_NAMES:
        folder = injector_config.response_dir(distortion)
        data_path = os.path.join(folder, "data.json")
        scores_path = os.path.join(folder, "scores.json")
        if not (os.path.exists(data_path) and os.path.exists(scores_path)):
            continue
        with open(data_path) as f:
            data = json.load(f)
        with open(scores_path) as f:
            scores = json.load(f)
        y_true = np.array(data["y_true"])
        mask = np.array(data["mask"]).astype(bool)
        true_iqr = _iqr(y_true[mask])
        levels = sorted(k for k in data if len(k) == 2 and k.startswith("L"))
        out[distortion] = {
            "scores": scores,
            "n_timesteps": y_true.shape[-1],
            "damage_levels": data.get("levels"),
            "iqr_ratio": {L: _iqr(np.array(data[L])[mask]) / (true_iqr + 1e-12)
                          for L in levels},
        }
    return out
