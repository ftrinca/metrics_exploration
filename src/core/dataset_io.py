"""Transposing between ImputeGAP's (n_timesteps, n_series) arrays and the
[series][timestep] JSON the caches use."""

import json
import os

import numpy as np


def matrix_to_lists(mat: np.ndarray, decimals: int = 4) -> list:
    """Round to `decimals` and transpose. The rounding bounds cache size and is
    the reason exact identities have to be checked at a tolerance."""
    n_series = mat.shape[1]
    return [
        [round(float(value), decimals) for value in mat[:, series_idx]]
        for series_idx in range(n_series)
    ]


def matrix_to_mask(mat_contaminated: np.ndarray) -> list:
    """True marks a missing value."""
    is_missing = np.isnan(mat_contaminated)
    n_timesteps, n_series = is_missing.shape
    return [
        [bool(is_missing[t, series_idx]) for t in range(n_timesteps)]
        for series_idx in range(n_series)
    ]


def bool_matrix_to_mask(mask: np.ndarray) -> list:
    """Transpose a mask that is already boolean."""
    n_timesteps, n_series = mask.shape
    return [
        [bool(mask[t, series_idx]) for t in range(n_timesteps)]
        for series_idx in range(n_series)
    ]


def save_dataset(output_path: str, json_out: dict) -> None:
    """Write `json_out`, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_out, f, indent=2)
