import os

from algo_ranking.algorithms import ALGORITHMS

ALGO_NAMES = [name for name, _, _ in ALGORITHMS]

# CIS's four components, one per Experiment 2 category. COMPONENT_VARIANTS below
# re-measures every single-metric substitution.
CIS_METRICS = ("mae", "wd", "dtw", "mi")

# Each component divides by its scale here, read off Experiment 1's damage-reactivity run:
# the mean value the metric takes over the eight calibrated distortions,
# averaged over Experiment 1's conditions. derive_component_scales recomputes them.
COMPONENT_SCALES = {
    "mae": 0.4964,
    "wd": 0.3338,
    "dtw": 0.0146,   # applied to DTW / n_timesteps, not to raw DTW
    "mi": 0.9400,
}

# Used by the substitution sweep for metrics the adopted score does not use.
FALLBACK_SCALE = 1.0

# Gate thresholds, both applied to the IQR ratio. Picked by eye from
# cis_gate_distribution.png; derive_gate_thresholds checks them against the data.
FLAT_THRESHOLD = 0.15
UNSTABLE_THRESHOLD = 3.0

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIS_PLOT_DIR = os.path.join(_SRC, "plots", "cis")
CIS_REPORT_DIR = os.path.join(_SRC, "reports", "cis")

ALGO_COLORS = {
    "CDRec": "#4C72B0", "ROSL": "#DD8452", "DynaMMo": "#55A868",
    "STMVL": "#C44E52", "BRITS": "#8172B2", "MPIN": "#937860",
}
PATTERN_COLORS = {"mcar": "#4C72B0", "scattered": "#55A868", "blackout": "#C44E52"}


_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIS_PLOT_DIR = os.path.join(_SRC, "plots", "cis")
CIS_REPORT_DIR = os.path.join(_SRC, "reports", "cis")
