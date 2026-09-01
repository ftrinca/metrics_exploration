import os

from metric_eval.core.metric_config import CATEGORIES
from metric_eval.paths import PLOTS_DIR, REPORTS_DIR, TIME_SERIES_DIR

# Dataset
DATASET = "airq"
N_SERIES = 10
NORMALIZATION = "none"  # airq ships already z-scored

# Missingness grid
PATTERNS = ["mcar", "scattered", "blackout"]
RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

RANGE_BUCKETS = {
    "low":    [0.1, 0.2, 0.3],
    "medium": [0.4, 0.5, 0.6],
    "high":   [0.7, 0.8],
}

# Damage targets, in units of sigma
TARGET_DAMAGE = 0.5
DAMAGE_TOLERANCE = 0.01     # a solve counts as reached within this of the target
DAMAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Fixed scenario for the damage-response experiment
RESPONSE_PATTERN = "blackout"
RESPONSE_RATE = 0.4

# Per distortion: the property it disturbs, its severity knob, and the properties
# it leaves exactly intact. invariance.py turns each invariant into an assertion.
#   "multiset"  every true value still appears exactly once in the output
#   "mean"      the mean of the reconstruction equals the mean of the truth
#   "affine"    the output is a positive-slope affine transform of the truth
#   "rank"      the output is a non-decreasing function of the truth
DISTORTIONS = {
    "noise":      {"disturbs": "pointwise accuracy, at random",
                   "param": "noise sd / sigma",
                   "preserves": ()},
    "bias":       {"disturbs": "pointwise accuracy, systematically",
                   "param": "offset / sigma",
                   "preserves": ("affine",)},
    "reorder":    {"disturbs": "order in time",
                   "param": "fraction of gap positions rotated",
                   "preserves": ("multiset", "mean")},
    "discretise": {"disturbs": "shape of the value distribution",
                   "param": "grid step / sigma",
                   "preserves": ("rank",)},
    "lag":        {"disturbs": "alignment in time",
                   "param": "lag (timesteps)",
                   "preserves": ()},
    "smooth":     {"disturbs": "short-term detail and variance",
                   "param": "moving-average window (timesteps)",
                   "preserves": ()},
    "spikes":     {"disturbs": "the tails, leaving most values exact",
                   "param": "spike magnitude / sigma",
                   "preserves": ()},
    "rescale":    {"disturbs": "variance, leaving shape intact",
                   "param": "scale factor minus one",
                   "preserves": ("affine", "mean")},
}

DISTORTION_NAMES = list(DISTORTIONS.keys())

SPIKE_RATE = 0.05   # fraction of gap positions that receive a spike, held fixed

SEED = 42

# Row grouping for figures and reports, and the metrics that follow from it.
INJECTOR_CATEGORIES = [
    "Pointwise Distance",
    "Distributional Divergence",
    "Temporal Structure",
    "Statistical Agreement",
]

INJECTOR_METRICS: list[str] = [m for cat in INJECTOR_CATEGORIES for m in CATEGORIES[cat]]

# Code label -> the name used in the thesis.
METRIC_LABEL = {
    "mae": "MAE", "rmse": "RMSE", "mse": "MSE", "mre": "MRE",
    "smape": "sMAPE", "nrmse": "nRMSE", "nd": "ND",
    "wd": "WD", "jsd": "JSD", "kld": "KLD",
    "acf": "ACF", "dtw": "DTW", "smae": "sMAE",
    "pearson": "Pearson", "mi": "MI", "r2": "R²",
    "tost": "TOST", "ba": "BA", "cdt": "CDT",
}

DISTORTION_LABEL = {d: d for d in DISTORTION_NAMES}

CATEGORY_COLOR = {
    "Pointwise Distance": "#1f4e79",
    "Distributional Divergence": "#7b3294",
    "Temporal Structure": "#1b7837",
    "Statistical Agreement": "#b35806",
}


def ordered_metrics() -> list[tuple[str, str]]:
    """(category, metric) pairs in report order."""
    return [(cat, m) for cat in INJECTOR_CATEGORIES for m in CATEGORIES[cat]]

# Output paths

REPORT_DIR = os.path.join(REPORTS_DIR, "injector")
PLOT_DIR = os.path.join(PLOTS_DIR, "injector")
DATA_DIR = os.path.join(TIME_SERIES_DIR, "injector")

REACTIVITY_PLOT_DIR = os.path.join(PLOT_DIR, "reactivity")
RESPONSE_PLOT_DIR = os.path.join(PLOT_DIR, "response")
REACTIVITY_REPORT_DIR = os.path.join(REPORT_DIR, "reactivity")
RESPONSE_REPORT_DIR = os.path.join(REPORT_DIR, "response")


def rate_dir(pattern: str, rate: float) -> str:
    """Cache directory for one (pattern, rate) of the damage-reactivity experiment."""
    return os.path.join(DATA_DIR, DATASET, pattern, f"{round(rate * 100):02d}pct")


def pass_filename(base: str, damage_metric: str) -> str:
    """The cache filename of one calibration pass.

    The MAE pass keeps the unsuffixed names so every existing cache stays
    valid; the RMSE pass writes e.g. calibration_rmse.json beside it.
    """
    if damage_metric == "mae":
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}_{damage_metric}{ext}"


def response_dir(distortion: str) -> str:
    """Cache directory for one distortion's damage-response curve."""
    return os.path.join(DATA_DIR, "_response", DATASET, RESPONSE_PATTERN, distortion)
