"""Experiment design for the Injector: dataset, missingness grid, damage
targets, distortion metadata and cache paths.

Every distortion is applied at a severity solved so that all eight cause the
same damage, defined as mean(|y_hat - y|) at the masked positions divided by
sigma, the standard deviation of the true values in that series' missing
block. Holding damage constant means any variation left in a metric is about
the kind of damage rather than its size.
"""

import os

DATASET = "airq"
N_SERIES = 10
NORMALIZATION = "none"  # airq ships already z-scored (mean~0, std~1 per series)

PATTERNS = ["mcar", "scattered", "blackout"]
RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

RANGE_BUCKETS = {
    "low":    [0.1, 0.2, 0.3],
    "medium": [0.4, 0.5, 0.6],
    "high":   [0.7, 0.8],
}

# The damage every distortion is solved to, in units of sigma. It has to sit
# under the ceiling of the most bounded distortion, because smoothing cannot
# exceed E|y - mu| ~ 0.8 sigma however wide the window and reordering cannot
# exceed E|y_i - y_j| ~ 1.13 sigma even at a full permutation. A target above
# roughly 0.7 would be unreachable for smoothing, so it is kept well below
# that and every solve reports whether it actually reached (see calibrate.py).
TARGET_DAMAGE = 0.5

# Solver tolerance in the same sigma units: a solve counts as reached when
# |achieved - target| <= this.
#
# The binding constraint is reorder, whose number of moved positions is an
# integer, so its damage can only move in steps of roughly |y_i - y_j| / n and
# a single step can be around 0.01 sigma. Every other distortion lands orders
# of magnitude inside this, and tightening it would only make reorder report
# misses it cannot avoid.
DAMAGE_TOLERANCE = 0.01

# Sweeping damage rather than each distortion's own parameter gives all eight
# sweeps one shared x-axis, so a single plot can show which distortion a given
# metric is blind to.
DAMAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# Fixed scenario for the sweep. Blackout gives one contiguous block, so a
# distortion acts on real structure rather than on isolated points, and 40% of
# the timesteps leaves room for the widest smoothing window and the longest lag
# the solver may pick.
SWEEP_PATTERN = "blackout"
SWEEP_RATE = 0.4

# Per distortion: the property it disturbs, its severity knob, and the
# properties it leaves EXACTLY intact. The invariants are not commentary,
# because invariance.py turns each one into an assertion about specific metrics
# and the aggregate report prints a pass/fail table. A failure means either the
# distortion is not doing what it claims or the metric implementation is wrong.
#
# Invariant vocabulary:
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

# Fraction of gap positions that receive a spike. Held fixed so that the solver
# has one knob (magnitude) and the "rare but large" character of the distortion
# survives at every damage level.
SPIKE_RATE = 0.05

SEED = 42

# Used only to group rows in figures and reports; no claim is attached to the
# grouping here.
INJECTOR_CATEGORIES = [
    "Pointwise Error",
    "Distributional",
    "Temporal / Shape",
    "Statistical Agreement",
]

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
