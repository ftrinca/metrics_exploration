"""Single source of truth for metric categorisation: which metrics exist, how
they're grouped for reporting, which direction ("lower"/"higher") is better,
and which metrics need special handling (full-series input, probabilistic
output).
"""

CATEGORIES: dict[str, list[str]] = {
    "Pointwise Error":      ["mae", "rmse", "mse", "mre", "smape", "nrmse", "nd"],
    "Distributional":       ["wd", "jsd", "kld"],
    "Probabilistic":        ["crps", "nll"],         # only for probabilistic outputs, see PROBABILISTIC_METRICS
    "Temporal / Shape":     ["acf", "dtw", "smae"],  # always need the full series, see FULL_SERIES_METRICS
    "Statistical Agreement":["pearson", "mi", "r2", "tost", "ba", "cdt"],
    "Domain-specific":      ["pfc"],
}

# flat ordered list derived from CATEGORIES — single source of truth for ordering
METRIC_LIST: list[str] = [m for metrics in CATEGORIES.values() for m in metrics]

# Metrics that must receive the full series, since masking would destroy what
# they measure: ACF needs all timesteps for the autocorrelation structure,
# DTW needs a contiguous sequence for the warping path, and sMAE (spectral)
# needs all timesteps to estimate the power spectral density. All other
# metrics are evaluated only on the missing (masked) positions. Derived from
# category membership: every "Temporal / Shape" metric is full-series.
FULL_SERIES_METRICS: set[str] = set(CATEGORIES["Temporal / Shape"])

# Metrics that only make sense when an algorithm's output is probabilistic
# (an extra "samples" dimension per missing point, rather than a single point
# estimate). Derived from category membership. generate_reports skips these
# for any algorithm whose output is not probabilistic (see is_probabilistic).
PROBABILISTIC_METRICS: set[str] = set(CATEGORIES["Probabilistic"])

# "lower" = smaller value is better, "higher" = larger value is better
METRIC_DIRECTION: dict[str, str] = {
    "mae":     "lower",
    "rmse":    "lower",
    "mse":     "lower",
    "mre":     "lower",
    "smape":   "lower",
    "nrmse":   "lower",
    "smae":    "lower",
    "nd":      "lower",
    "wd":      "lower",
    "jsd":     "lower",
    "kld":     "lower",
    "cdt":     "lower",
    "crps":    "lower",
    "nll":     "lower",
    "acf":     "lower",
    "dtw":     "lower",
    "pearson": "higher",
    "mi":      "higher",
    "r2":      "higher",
    "tost":    "lower",   # max(p1, p2); equivalence requires p < alpha, so lower = stronger evidence
    "ba":      "lower",   # ranked by |mean_diff|; closer to 0 = less bias
    "pfc":     "higher",
}
