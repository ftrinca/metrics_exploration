"""Metric implementations, one function per metric, each taking
(y_true, y_pred) 1D arrays and returning a float (except `ba`, which returns
a (mean_diff, loa) tuple). Called by generate_reports.compute_all_scores.
See metric_config.py for how metrics are grouped and ranked, and
metric_verification.md for a detailed review of each formula.
"""

import warnings

import numpy as np
from scipy.stats import wasserstein_distance, pearsonr, entropy, norm
from scipy.spatial.distance import jensenshannon
from scipy.signal import lombscargle
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,   # requires sklearn >= 1.4
    mean_squared_error,
    mutual_info_score,
    r2_score,
)
from statsmodels.tsa.stattools import acf as _acf
from dtaidistance import dtw as _dtw
import properscoring as ps
import pingouin as pg


# ── rank 1 ─────────────────────────────────────────────────────────────────
# MAE — same unit as y, lower is better
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


# ── rank 2 ─────────────────────────────────────────────────────────────────
# RMSE — same unit as y, lower is better
def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


# ── rank 3 ─────────────────────────────────────────────────────────────────
# MSE — unit = y², lower is better
def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


# ── rank 4 ─────────────────────────────────────────────────────────────────
# MRE — unitless ratio ≥ 0, lower is better
# Zero-valued ground truth entries are excluded: relative error is undefined
# when the true value is 0, and including them (even with an epsilon) inflates
# the metric by several orders of magnitude.
def mre(y_true, y_pred):
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


# ── rank 5 ─────────────────────────────────────────────────────────────────
# sMAPE (Symmetric MAPE) — percentage in [0, 200%], lower is better.
# Standard MAPE uses per-element y_true[i] as denominator, which explodes for
# zero or near-zero values and produces nonsensical negative percentages for
# zero-mean series (e.g. z-score normalised data). sMAPE uses the mean of
# |y_true| and |y_pred| per element, bounding the result and handling any sign.
def smape(y_true, y_pred):
    denom = 0.5 * (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


# ── rank 6 ─────────────────────────────────────────────────────────────────
# WD (Wasserstein / Earth Mover's Distance) — same unit as y, lower is better
def wd(y_true, y_pred):
    return wasserstein_distance(y_true, y_pred)


# ── rank 7 ─────────────────────────────────────────────────────────────────
# Pearson — unitless in [-1, 1], higher is better
def pearson(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


# ── rank 8 ─────────────────────────────────────────────────────────────────
# JSD (Jensen-Shannon Divergence) — unitless in [0, ln2] (base-e), lower is better.
# scipy.jensenshannon returns the JS *distance* (sqrt of divergence); the
# result is squared to recover the divergence.
# Passing raw time-series arrays (which can be negative, e.g. z-score data)
# to jensenshannon produces inf. Empirical histograms are built first.
def jsd(y_true, y_pred):
    bins = max(10, int(np.sqrt(len(y_true))))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    p, _ = np.histogram(y_true, bins=bins, range=(lo, hi))
    q, _ = np.histogram(y_pred, bins=bins, range=(lo, hi))
    p = p.astype(float) + 1e-10
    q = q.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q) ** 2)


# ── rank 9 ─────────────────────────────────────────────────────────────────
# MI — non-negative float (nats/bits), higher is better
# mutual_info_score expects discrete labels, so both arrays are binned into
# sqrt(n) equal-width bins before calling it.
def mi(y_true, y_pred):
    bins = max(10, int(np.sqrt(len(y_true))))
    lo   = min(y_true.min(), y_pred.min())
    hi   = max(y_true.max(), y_pred.max()) + 1e-10
    edges = np.linspace(lo, hi, bins + 1)
    return float(mutual_info_score(
        np.digitize(y_true, edges),
        np.digitize(y_pred, edges),
    ))


# ── rank 10 ────────────────────────────────────────────────────────────────
# R² — unitless in (-∞, 1], higher is better
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


# ── rank 11 ────────────────────────────────────────────────────────────────
# CRPS — same unit as y, lower is better.
# For deterministic algorithms (point estimates): CRPS == MAE.
# For probabilistic algorithms (posterior samples): y_pred should be shape
#   (n_timesteps, n_samples); crps_ensemble then evaluates the full distribution.
# TODO: posterior-sample format for probabilistic algorithms is unverified -
#   see metric_verification.md "Next Steps".
def crps(y_true, y_pred):
    forecasts = y_pred[:, np.newaxis] if y_pred.ndim == 1 else y_pred
    return float(np.mean(ps.crps_ensemble(y_true, forecasts)))


# ── rank 12 ────────────────────────────────────────────────────────────────
# ACF difference — unitless in [0, 2], lower is better
# statsmodels computes each ACF; compare the resulting vectors.
def acf(y_true, y_pred):
    nlags = min(40, len(y_true) // 4)
    acf_true = _acf(y_true, nlags=nlags, fft=True)
    acf_pred = _acf(y_pred, nlags=nlags, fft=True)
    return float(np.mean(np.abs(acf_true - acf_pred)))


# ── rank 13 ────────────────────────────────────────────────────────────────
# TOST — p-value in [0, 1], LOWER is better (p < 0.05 for both one-sided
# tests → equivalence demonstrated).
# pingouin.tost returns a DataFrame; take max(p-values) as the binding test.
# epsilon defaults to 10 % of std(y_true).
def tost(y_true, y_pred, epsilon=None):
    if epsilon is None:
        epsilon = 0.1 * float(np.std(y_true))
    # Suppress scipy's precision-loss RuntimeWarning, which fires when the
    # difference series is nearly constant (e.g. a perfect constant offset).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pg.tost(y_true, y_pred, bound=epsilon, paired=True)
    return float(result["pval"].max())


# ── rank 14 ────────────────────────────────────────────────────────────────
# BA (Bland-Altman) — returns (mean_diff, loa), both same unit as y, both
# closer to 0 is better.
# pyCompare.blandAltman and pingouin.plot_blandaltman only return plot objects, not numeric values
# Compute directly with numpy.
def ba(y_true, y_pred):
    diff = y_true - y_pred
    mean_diff = float(np.mean(diff))
    loa = float(1.96 * np.std(diff, ddof=1))
    return mean_diff, loa


# ── rank 15 ────────────────────────────────────────────────────────────────
# NRMSE — unitless in [0, ∞), lower is better
# RMSE normalised by std(y_true), following the MissForest definition.
# Std normalisation remains well-defined for zero-mean (z-score normalised)
# data, unlike mean or min-max normalisation; see the MRE section for the
# corresponding z-score caveat.
def nrmse(y_true, y_pred):
    denom = float(np.std(y_true))
    if denom == 0:
        return 0.0
    return float(root_mean_squared_error(y_true, y_pred) / denom)


# ── rank 16 ────────────────────────────────────────────────────────────────
# KLD — non-negative float (nats), lower is better
# scipy.stats.entropy(p, q) = KL(p || q). Arrays are binned into histograms
# to form proper probability distributions.
def kld(y_true, y_pred):
    bins = max(10, int(np.sqrt(len(y_true))))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    p, _ = np.histogram(y_true, bins=bins, range=(lo, hi))
    q, _ = np.histogram(y_pred, bins=bins, range=(lo, hi))
    p = p.astype(float) + 1e-10
    q = q.astype(float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


# ── rank 17 ────────────────────────────────────────────────────────────────
# DTW — same unit as y (cumulative warping cost), lower is better
# dtaidistance has a fast C backend.
# pip install dtaidistance
def dtw(y_true, y_pred):
    return _dtw.distance(
        y_true.astype(np.float64),
        y_pred.astype(np.float64),
    )


# ── rank 18 ────────────────────────────────────────────────────────────────
# CDT (Cohen's Distance Test / Cohen's d) — unitless standardised mean
# difference, lower is better (0 = no difference in means).
# pooled_std is the average of the two sample variances (ddof=1), following
# the standard Cohen's d definition.
def cdt(y_true, y_pred):
    pooled_std = np.sqrt(0.5 * (np.var(y_true, ddof=1) + np.var(y_pred, ddof=1)))
    if pooled_std == 0:
        return 0.0
    return float(abs(np.mean(y_true) - np.mean(y_pred)) / pooled_std)


# ── rank 19 ────────────────────────────────────────────────────────────────
# NLL (Negative Log-Likelihood, Gaussian assumption) — float, lower is better.
# For deterministic algorithms (point estimates): sigma is a single global
# value estimated from the residuals, and NLL reduces to a monotone function
# of RMSE (see metric_verification.md).
# For probabilistic algorithms (posterior samples): y_pred should be shape
#   (n_timesteps, n_samples); mu and sigma are then estimated per-timestep
#   from the sample distribution, giving a genuine per-point predictive
#   likelihood instead of a single global sigma.
# TODO: same open question as CRPS - see metric_verification.md "Next Steps".
#
# Two degenerate cases need guarding for the deterministic (1D) path:
#   1. sigma == 0 exactly (e.g. a perfect predictor) → return 0.
#   2. sigma is near-zero due to float rounding of a near-constant offset
#      (e.g. JSON values rounded to 4 d.p.) → norm.logpdf with scale≈1e-15
#      produces astronomically large values. Any sigma below 1e-6 of
#      σ(y_true) is treated as effectively zero, returning 0.
def nll(y_true, y_pred):
    if y_pred.ndim == 2:
        mu = np.mean(y_pred, axis=1)
        sigma = np.std(y_pred, axis=1)
        sigma_y = float(np.std(y_true))
        floor = sigma_y if sigma_y > 0 else 1.0
        sigma = np.where(sigma < 1e-6 * floor, floor, sigma)
        return float(-np.mean(norm.logpdf(y_true, loc=mu, scale=sigma)))

    residuals = y_true - y_pred
    sigma = float(np.std(residuals))
    if sigma == 0:
        return 0.0
    sigma_y = float(np.std(y_true))
    if sigma_y > 0 and sigma < 1e-6 * sigma_y:
        return 0.0
    return float(-np.mean(norm.logpdf(y_true, loc=y_pred, scale=sigma)))


# ── rank 20 ────────────────────────────────────────────────────────────────
# sMAE (Spectral MAE, LSCD) — unitless in [0, 2], lower is better.
# MAE between the normalised Lomb-Scargle power spectral densities (PSDs) of
# the true and imputed series. Each PSD is normalised to sum to 1 (a
# distribution over frequency bins), so sMAE measures how much the relative
# frequency content differs, independent of overall amplitude — capturing
# distortions (e.g. smoothing out periodicity) that MAE/RMSE can miss.
# This is a FULL_SERIES_METRIC: PSD estimation needs the whole series, like
# ACF and DTW (see metric_config.py).
def smae(y_true, y_pred, n_freqs=50):
    n = len(y_true)
    t = np.arange(n, dtype=np.float64)
    freqs = np.linspace(2 * np.pi / n, np.pi, n_freqs)

    psd_true = lombscargle(t, y_true - np.mean(y_true), freqs)
    psd_pred = lombscargle(t, y_pred - np.mean(y_pred), freqs)

    psd_true = psd_true / (psd_true.sum() + 1e-10)
    psd_pred = psd_pred / (psd_pred.sum() + 1e-10)

    return float(np.mean(np.abs(psd_true - psd_pred)))


# ── rank 21 ────────────────────────────────────────────────────────────────
# ND (Normalized Deviation) — unitless ratio ≥ 0, lower is better
# gluonts implements ND but as a class inside an evaluation framework, not
# as a simple function. One-liner numpy is simpler.
def nd(y_true, y_pred):
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


# ── rank 22 ────────────────────────────────────────────────────────────────
# PFC (Proportion of Forecasts within tolerance) — % in [0, 100],
# HIGHER is better. No pip library implements this metric.
# Default tolerance = 10 % relative error.
def pfc(y_true, y_pred, tolerance=0.10):
    rel_err = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10)
    return float(np.mean(rel_err <= tolerance) * 100)