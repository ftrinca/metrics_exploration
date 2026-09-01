import json
import os

import numpy as np

from metric_eval.experiments.algorank.config import rate_dir, seed_dir


def deterministic_path(dataset: str, pattern: str, rate: float) -> str:
    """Path of the ground truth, mask and deterministic reconstructions of one scenario."""
    return os.path.join(rate_dir(dataset, pattern, rate), "deterministic.json")


def seed_path(dataset: str, pattern: str, rate: float, seed: int) -> str:
    """Path of one seed's stochastic reconstructions."""
    return os.path.join(seed_dir(dataset, pattern, rate, seed), "data.json")


def load_scenario(dataset: str, pattern: str, rate: float, seed: int) -> dict:
    """Return {y_true, mask, **reconstructions} for one seed, all in native
    (n_series, n_timesteps) orientation.

    The ground truth, the mask and the deterministic reconstructions live once
    per scenario in deterministic.json; only the stochastic ones are stored per
    seed. This merges the two halves back into the flat dict the callers expect.
    """
    det_path = deterministic_path(dataset, pattern, rate)
    data_path = seed_path(dataset, pattern, rate, seed)
    for path in (det_path, data_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} - run the algorank build stage for "
                f"dataset={dataset!r} pattern={pattern!r} rate={rate} seed={seed} first."
            )

    with open(det_path) as f:
        det = json.load(f)
    with open(data_path) as f:
        sto = json.load(f)

    out = {"y_true": np.array(det["y_true"]), "mask": np.array(det["mask"])}
    for name, mat in det.items():
        if name not in ("y_true", "mask"):
            out[name] = np.array(mat)
    for name, mat in sto.items():
        out[name] = np.array(mat)
    return out


def reconstructions(scenario: dict) -> dict[str, np.ndarray]:
    """The reconstructions of a loaded scenario, without the truth and the mask."""
    return {k: v for k, v in scenario.items() if k not in ("y_true", "mask")}
