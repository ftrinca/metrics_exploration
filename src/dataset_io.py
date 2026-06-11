"""Shared helpers for converting (n_timesteps, n_series) numpy matrices to the
[series][timestep] JSON format used throughout this project, and for writing
that JSON to a dataset's output path.

data/ and reconstruction/ produce arrays in ImputeGAP's (n_timesteps,
n_series) orientation, but generate_reports.load_data / compute_all_scores
expect [series][timestep] lists - these helpers do that transpose in one
place instead of duplicating it across build_datasets.py.
"""

import json
import os

import numpy as np


def matrix_to_lists(mat: np.ndarray, decimals: int = 4) -> list:
    """Convert a (n_timesteps, n_series) array into [series][timestep] lists,
    rounding values to `decimals` places to keep file sizes reasonable."""
    n_series = mat.shape[1]
    return [
        [round(float(value), decimals) for value in mat[:, series_idx]]
        for series_idx in range(n_series)
    ]


def matrix_to_mask(mat_contaminated: np.ndarray) -> list:
    """Convert a NaN-contaminated (n_timesteps, n_series) array into a
    boolean [series][timestep] mask, where True = "this value is missing"."""
    is_missing = np.isnan(mat_contaminated)  # shape: (n_timesteps, n_series)
    n_timesteps, n_series = is_missing.shape
    return [
        [bool(is_missing[t, series_idx]) for t in range(n_timesteps)]
        for series_idx in range(n_series)
    ]


def bool_matrix_to_mask(mask: np.ndarray) -> list:
    """Convert an existing (n_timesteps, n_series) boolean mask into
    [series][timestep] lists (no NaN-detection step needed)."""
    n_timesteps, n_series = mask.shape
    return [
        [bool(mask[t, series_idx]) for t in range(n_timesteps)]
        for series_idx in range(n_series)
    ]


def save_dataset(output_path: str, json_out: dict) -> None:
    """Write `json_out` to `output_path`, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_out, f, indent=2)
