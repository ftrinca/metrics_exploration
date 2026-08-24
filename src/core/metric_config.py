"""Which metrics exist, how they are grouped for reporting, which direction
counts as better, and which of them need special handling.
"""

CATEGORIES: dict[str, list[str]] = {
    "Pointwise Error":      ["mae", "rmse", "mse", "mre", "smape", "nrmse", "nd"],
    "Distributional":       ["wd", "jsd", "kld"],
    "Temporal / Shape":     ["acf", "dtw", "smae"],
    "Statistical Agreement":["pearson", "mi", "r2", "tost", "ba", "cdt"],
    "Domain-specific":      ["pfc"],
}

METRIC_LIST: list[str] = [m for metrics in CATEGORIES.values() for m in metrics]

# Masking these would destroy what they measure: the autocorrelation structure,
# the warping path and the power spectrum all need the whole series. Every
# other metric is evaluated only at the missing positions.
FULL_SERIES_METRICS: set[str] = set(CATEGORIES["Temporal / Shape"])

# Not undefined for a point estimate, but redundant on one: CRPS is exactly
# MAE there, and NLL with a single sigma fitted to the residuals is a
# monotone function of RMSE. Since no algorithm in this project produces
# posterior samples, scoring them would add two columns and no information,
# so they are absent from CATEGORIES and never scored. Kept here because
# core.scoring.metric_applies still guards on the set.
PROBABILISTIC_METRICS: set[str] = {"crps", "nll"}

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
    "acf":     "lower",
    "dtw":     "lower",
    "pearson": "higher",
    "mi":      "higher",
    "r2":      "higher",
    "tost":    "lower",   # a p-value: equivalence needs it below alpha
    "ba":      "lower",   # ranked on |mean_diff|
    "pfc":     "higher",
}
