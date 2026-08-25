import os

# Six datasets, two per quadrant of a cross-series-correlation by periodicity design.
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
MAX_TIMESTEPS = 2000        # a cap: a shorter dataset keeps its native length
NORMALIZATION = "z_score"   # idempotent on the already z-scored datasets

# Scenario grid, matching the injector so the two experiments bucket alike.
PATTERNS = ["mcar", "scattered", "blackout"]
RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

RANGE_BUCKETS = {
    "low":    [0.1, 0.2, 0.3],
    "medium": [0.4, 0.5, 0.6],
    "high":   [0.7, 0.8],
}

N_SEEDS = 3                 # independent draws per scenario; score.py averages them

# Reconstruction plots
PLOT_SERIES_INDEX = 1       # clamped to 0 for a dataset with fewer series
PLOT_WINDOW_TIMESTEPS = 200 # the start offset is chosen per scenario by visualize.py

# The 8 metrics kept from the metric-evaluation part, two per category.
ALGO_CATEGORIES: dict[str, list[str]] = {
    "Pointwise Distance":      ["mae", "rmse"],
    "Distributional Divergence": ["wd", "jsd"],
    "Temporal Structure":      ["dtw", "smae"],
    "Statistical Agreement":   ["r2", "mi"],
}

ALGO_METRICS: list[str] = [m for cat in ALGO_CATEGORIES.values() for m in cat]

# Thesis spelling, which upper-casing the internal key gets wrong for the
# mixed-case names.
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


def heatmap_dir(dataset: str) -> str:
    """Ranking heatmaps for one dataset."""
    return os.path.join(PLOT_DIR, dataset, "heatmap")


def reconstruction_dir(dataset: str) -> str:
    """Reconstruction plots for one dataset. Visual inspection only."""
    return os.path.join(PLOT_DIR, dataset, "reconstruction")


def report_dir(dataset: str) -> str:
    """Text ranking reports for one dataset."""
    return os.path.join(REPORT_DIR, dataset)


def rate_dir(dataset: str, pattern: str, rate: float) -> str:
    """Cache directory holding one scenario's scores.json and its seed directories."""
    return os.path.join(DATA_DIR, dataset, pattern, f"{round(rate * 100):02d}pct")


def seed_dir(dataset: str, pattern: str, rate: float, seed: int) -> str:
    """Cache directory holding one seed's data.json."""
    return os.path.join(rate_dir(dataset, pattern, rate), f"seed{seed}")
