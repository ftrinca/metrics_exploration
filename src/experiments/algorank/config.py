import os

from paths import PLOTS_DIR, REPORTS_DIR, TIME_SERIES_DIR

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

# Repeated from algorithms.py, which cannot be imported without ImputeGAP and so
# without every algorithm's own dependencies. The parts of the pipeline that only
# read cached results import the names from here, and algorithms.py asserts that
# the two still agree.
ALGO_NAMES = ["CDRec", "ROSL", "DynaMMo", "STMVL", "BRITS", "MPIN"]
STOCHASTIC_ALGO_NAMES = {"BRITS", "MPIN"}

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


REPORT_DIR = os.path.join(REPORTS_DIR, "algorank")
PLOT_DIR = os.path.join(PLOTS_DIR, "algorank")
DATA_DIR = os.path.join(TIME_SERIES_DIR, "algorank")


def heatmap_dir(dataset: str) -> str:
    """Ranking heatmaps for one dataset, bucketed over a range of rates."""
    return os.path.join(PLOT_DIR, dataset, "heatmap")


def rate_heatmap_dir(dataset: str) -> str:
    """Ranking heatmaps for one dataset, one per individual rate."""
    return os.path.join(PLOT_DIR, dataset, "heatmap", "by_rate")


def reconstruction_dir(dataset: str) -> str:
    """Reconstruction plots for one dataset. Visual inspection only."""
    return os.path.join(PLOT_DIR, dataset, "reconstruction")


def report_dir(dataset: str) -> str:
    """Text ranking reports for one dataset, bucketed over a range of rates."""
    return os.path.join(REPORT_DIR, dataset)


def rate_report_dir(dataset: str) -> str:
    """Text ranking reports for one dataset, one per individual rate."""
    return os.path.join(REPORT_DIR, dataset, "by_rate")


def rate_dir(dataset: str, pattern: str, rate: float) -> str:
    """Cache directory holding one scenario's scores.json and its seed directories."""
    return os.path.join(DATA_DIR, dataset, pattern, f"{round(rate * 100):02d}pct")


def seed_dir(dataset: str, pattern: str, rate: float, seed: int) -> str:
    """Cache directory holding one seed's data.json."""
    return os.path.join(rate_dir(dataset, pattern, rate), f"seed{seed}")
