import os

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

# Fixed scenario for the damage sweep
SWEEP_PATTERN = "blackout"
SWEEP_RATE = 0.4

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

# Row grouping for figures and reports
INJECTOR_CATEGORIES = [
    "Pointwise Error",
    "Distributional",
    "Temporal / Shape",
    "Statistical Agreement",
]

# Output paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)

REPORT_DIR = os.path.join(_SRC, "reports", "injector")
PLOT_DIR = os.path.join(_SRC, "plots", "injector")
DATA_DIR = os.path.join(_SRC, "time_series", "injector")

EQUAL_PLOT_DIR = os.path.join(PLOT_DIR, "equal_damage")
SWEEP_PLOT_DIR = os.path.join(PLOT_DIR, "damage_sweep")
EQUAL_REPORT_DIR = os.path.join(REPORT_DIR, "equal_damage")
SWEEP_REPORT_DIR = os.path.join(REPORT_DIR, "damage_sweep")


def rate_dir(pattern: str, rate: float) -> str:
    """Cache directory for one (pattern, rate) of the equal-damage experiment."""
    return os.path.join(DATA_DIR, DATASET, pattern, f"{round(rate * 100):02d}pct")


def sweep_dir(distortion: str) -> str:
    """Cache directory for one distortion's damage sweep."""
    return os.path.join(DATA_DIR, "_sweep", DATASET, SWEEP_PATTERN, distortion)
