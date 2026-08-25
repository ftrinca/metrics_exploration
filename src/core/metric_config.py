# Metric registry: grouping for reports, direction, and the sets needing special handling.

CATEGORIES: dict[str, list[str]] = {
    "Pointwise Distance": ["mae", "rmse", "mse", "mre", "smape", "nrmse", "nd"],
    "Distributional Divergence": ["wd", "jsd", "kld"],
    "Temporal Structure": ["acf", "dtw", "smae"],
    "Statistical Agreement": ["pearson", "mi", "r2", "tost", "ba", "cdt"],
    "Domain-specific": ["pfc"],
}

METRIC_LIST: list[str] = [m for metrics in CATEGORIES.values() for m in metrics]

# Evaluated on the whole series; every other metric sees the missing positions only.
FULL_SERIES_METRICS: set[str] = set(CATEGORIES["Temporal Structure"])

# Never scored: redundant with MAE and RMSE on a point estimate.
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
    "tost":    "lower",
    "ba":      "lower",
    "pfc":     "higher",
}