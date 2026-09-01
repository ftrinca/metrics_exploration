import os

from metric_eval.experiments.algorank.config import ALGO_NAMES  # noqa: F401  re-exported for the CIS modules
from metric_eval.paths import PLOTS_DIR, REPORTS_DIR, TIME_SERIES_DIR

# The three components are chosen so that no kind of damage Experiment 1 defines
# goes undetected: each one is blind to something the other two see. Every other
# subset is reported in the substitution table, which is what rules them out.
CIS_METRICS = ("mae", "wd", "mi")

REFERENCE_NAME = "MEAN"

# Both thresholds are read off Experiment 2's standard-deviation ratios: a constant
# reconstruction sits at exactly 0, and no algorithm that does not diverge exceeds
# 1.7, which leaves the band up to BRITS's median of 13.2 empty.
FLAT_THRESHOLD = 0.15
UNSTABLE_THRESHOLD = 3.0

# Reported next to the adopted exponent so that it is not a hidden choice.
POWER_VARIANTS = (1.0, 2.0, float("inf"))
ADOPTED_POWER = 2.0



CIS_PLOT_DIR = os.path.join(PLOTS_DIR, "cis")
CIS_REPORT_DIR = os.path.join(REPORTS_DIR, "cis")
CIS_CACHE_DIR = os.path.join(TIME_SERIES_DIR, "cis")

ALGO_COLORS = {
    "CDRec": "#4C72B0", "ROSL": "#DD8452", "DynaMMo": "#55A868",
    "STMVL": "#C44E52", "BRITS": "#8172B2", "MPIN": "#937860",
}
PATTERN_COLORS = {"mcar": "#4C72B0", "scattered": "#55A868", "blackout": "#C44E52"}


def cache_path(dataset: str) -> str:
    """Derived quantities of one dataset's 24 scenarios."""
    return os.path.join(CIS_CACHE_DIR, f"{dataset}.json")
