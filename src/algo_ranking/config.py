"""Experiment design for the Algorithm Ranking part: datasets, missingness
patterns, rates, seeds, metric grouping, and the on-disk output layout.

The six datasets span three quadrants of a cross-series-correlation by
periodicity design space, two datasets per quadrant. Series length is not held
constant across them, so it stays a secondary variable when comparing rankings
between datasets. The Algorithm Ranking chapter covers the design and the
per-dataset statistics behind it.
"""

import os

DATASETS = [
    "temperature", "chlorine",           # high corr, strong periodicity
    "climate", "electricity",            # low corr, weak periodicity
    "drift", "forecast-economy",         # high corr, weak/no periodicity
]

# Read by nothing in the pipeline; kept for writing up results.
DATASET_QUADRANTS: dict[str, str] = {
    "temperature":      "high-corr / strong-periodicity",
    "chlorine":         "high-corr / strong-periodicity",
    "climate":          "low-corr / weak-periodicity",
    "electricity":      "low-corr / weak-periodicity",
    "drift":            "high-corr / weak-periodicity",
    "forecast-economy": "high-corr / weak-periodicity",
}

N_SERIES = 10
# A cap rather than a target: a dataset shorter than this keeps its native length.
MAX_TIMESTEPS = 2000
# Idempotent on datasets that are already z-scored, so it is applied to all of them.
NORMALIZATION = "z_score"

PATTERNS = ["mcar", "scattered", "blackout"]

# One representative rate per RANGE_BUCKETS entry, which makes each bucket's
# mean a mean over a single rate.
RATES = [0.2, 0.5, 0.8]

RANGE_BUCKETS = {
    "low":    [0.2],
    "medium": [0.5],
    "high":   [0.8],
}

# Independent draws per (dataset, pattern, rate); score.py averages across them.
N_SEEDS = 3

# One fixed series for every reconstruction plot, so the plots are comparable
# with each other. visualize.py clamps this to 0 for a dataset with fewer series.
PLOT_SERIES_INDEX = 1

# Window length for the reconstruction plots, because the full series is too
# dense to read. Where the window starts is chosen per scenario by visualize.py's
# _choose_window instead of being fixed here, since no single offset can be
# guaranteed to overlap the gap that scattered and blackout place at a random
# position.
PLOT_WINDOW_TIMESTEPS = 200

# The 8 metrics kept from the metric-evaluation part, two per category.
ALGO_CATEGORIES: dict[str, list[str]] = {
    "Pointwise Distance":      ["mae", "rmse"],
    "Statistical Agreement":   ["r2", "mi"],
    "Distributional Divergence": ["wd", "jsd"],
    "Temporal Structure":      ["dtw", "smae"],
}

ALGO_METRICS: list[str] = [m for cat in ALGO_CATEGORIES.values() for m in cat]

# How each metric is spelled in the thesis, which upper-casing the internal key
# gets wrong for sMAPE, nRMSE, sMAE and the rest of the mixed-case names.
METRIC_LABEL: dict[str, str] = {
    "mae": "MAE", "rmse": "RMSE", "mse": "MSE", "mre": "MRE",
    "smape": "sMAPE", "nrmse": "nRMSE", "nd": "ND",
    "wd": "WD", "jsd": "JSD", "kld": "KLD",
    "acf": "ACF", "dtw": "DTW", "smae": "sMAE",
    "pearson": "Pearson", "mi": "MI", "r2": "R2",
    "tost": "TOST", "ba": "BA", "cdt": "CDT",
}


def label(metric: str) -> str:
    """Thesis spelling for a metric key, falling back to the upper-cased key."""
    return METRIC_LABEL.get(metric, metric.upper())

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)

REPORT_DIR = os.path.join(_SRC, "reports", "algo_ranking")
PLOT_DIR = os.path.join(_SRC, "plots", "algo_ranking")
DATA_DIR = os.path.join(_SRC, "time_series", "algo_ranking")


# Outputs are grouped by dataset first and by kind second, so one dataset's
# results can be read without filtering a flat directory by filename prefix.

def heatmap_dir(dataset: str) -> str:
    """Ranking heatmaps for one dataset."""
    return os.path.join(PLOT_DIR, dataset, "heatmap")


def reconstruction_dir(dataset: str) -> str:
    """Reconstruction plots for one dataset. Visual inspection only, since
    these do not feed the ranking."""
    return os.path.join(PLOT_DIR, dataset, "reconstruction")


def report_dir(dataset: str) -> str:
    """Text ranking reports for one dataset."""
    return os.path.join(REPORT_DIR, dataset)


def rate_dir(dataset: str, pattern: str, rate: float) -> str:
    """Cache directory holding one scenario's scores.json, and the parent of
    each of its seed directories."""
    return os.path.join(DATA_DIR, dataset, pattern, f"{round(rate * 100):02d}pct")


def seed_dir(dataset: str, pattern: str, rate: float, seed: int) -> str:
    """Cache directory holding one seed's data.json."""
    return os.path.join(rate_dir(dataset, pattern, rate), f"seed{seed}")
