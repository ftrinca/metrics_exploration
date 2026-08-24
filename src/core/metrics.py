"""One function per metric, each taking 1-D (y_true, y_pred) and returning a
float. `ba` is the exception and returns (mean_diff, loa).

Units, since they differ and none of them is visible in the signature: mae,
rmse, wd, dtw, crps and ba are in the unit of y; mse is in y squared; smape is
a percentage in [0, 200]; pfc is a percentage in [0, 100]; kld, jsd, mi and
nll are in nats; the rest are unitless. core/metric_config.py holds which
direction counts as better.
"""

import warnings

import numpy as np
from scipy.stats import wasserstein_distance, pearsonr, entropy, norm
from scipy.spatial.distance import jensenshannon
from scipy.signal import lombscargle
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,   # sklearn >= 1.4
    mean_squared_error,
    mutual_info_score,
    r2_score,
)
from statsmodels.tsa.stattools import acf as _acf
from dtaidistance import dtw as _dtw
import properscoring as ps
import pingouin as pg


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def mre(y_true, y_pred):
    # Relative error is undefined where the truth is zero, and including those
    # positions with an epsilon denominator inflates the result by orders of
    # magnitude, so they are dropped.
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def smape(y_true, y_pred):
    # Symmetric, so the denominator is the mean of |y_true| and |y_pred| rather
    # than |y_true| alone. Plain MAPE divides by the truth, which explodes near
    # zero and goes negative on zero-mean data such as the z-scored series used
    # throughout this project.
    denom = 0.5 * (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def wd(y_true, y_pred):
    return wasserstein_distance(y_true, y_pred)


def pearson(y_true, y_pred):
    """Return 0.0 rather than NaN when either series is constant."""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pearsonr(y_true, y_pred)[0]


def jsd(y_true, y_pred):
    """Jensen-Shannon divergence, in [0, ln 2].

    scipy returns the JS *distance*, so the result is squared to recover the
    divergence. Both series are binned first: passing raw values, which can be
    negative on z-scored data, gives inf.
    """
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


def mi(y_true, y_pred):
    # mutual_info_score wants discrete labels, so both series are binned into
    # sqrt(n) equal-width bins spanning their shared range.
    bins = max(10, int(np.sqrt(len(y_true))))
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max()) + 1e-10
    edges = np.linspace(lo, hi, bins + 1)
    return float(mutual_info_score(
        np.digitize(y_true, edges),
        np.digitize(y_pred, edges),
    ))


def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def crps(y_true, y_pred):
    """Equal to MAE for a point estimate; evaluates the full distribution when
    y_pred has shape (n_timesteps, n_samples).

    The posterior-sample path is unverified against a reference implementation
    and no algorithm in this project produces that shape.
    """
    forecasts = y_pred[:, np.newaxis] if y_pred.ndim == 1 else y_pred
    return float(np.mean(ps.crps_ensemble(y_true, forecasts)))


def acf(y_true, y_pred):
    """Mean absolute difference between the two autocorrelation functions."""
    nlags = min(40, len(y_true) // 4)
    acf_true = _acf(y_true, nlags=nlags, fft=True)
    acf_pred = _acf(y_pred, nlags=nlags, fft=True)
    return float(np.mean(np.abs(acf_true - acf_pred)))


def tost(y_true, y_pred, epsilon=None):
    """Two one-sided equivalence test. Returns the binding p-value, so LOWER is
    better: below 0.05 means equivalence within `epsilon` was demonstrated.

    epsilon defaults to 10% of std(y_true).
    """
    if epsilon is None:
        epsilon = 0.1 * float(np.std(y_true))
    # scipy raises a precision-loss RuntimeWarning when the difference series
    # is nearly constant, which a pure offset makes it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = pg.tost(y_true, y_pred, bound=epsilon, paired=True)
    return float(result["pval"].max())


def ba(y_true, y_pred):
    """Bland-Altman, as (mean difference, limit of agreement). Both are in the
    unit of y and both are better closer to zero."""
    diff = y_true - y_pred
    mean_diff = float(np.mean(diff))
    loa = float(1.96 * np.std(diff, ddof=1))
    return mean_diff, loa


def nrmse(y_true, y_pred):
    # Normalised by std rather than by mean or range, which stay well defined
    # on the zero-mean series this project uses.
    denom = float(np.std(y_true))
    if denom == 0:
        return 0.0
    return float(root_mean_squared_error(y_true, y_pred) / denom)


def kld(y_true, y_pred):
    """KL(truth || prediction), over the same histograms `jsd` builds."""
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


def dtw(y_true, y_pred):
    # Sakoe-Chiba band at 10% of the series length, which makes this O(n*w)
    # instead of O(n^2) and bounds how far a point may be warped.
    window = max(1, int(0.1 * len(y_true)))
    return _dtw.distance(
        y_true.astype(np.float64),
        y_pred.astype(np.float64),
        window=window,
    )


def cdt(y_true, y_pred):
    """Cohen's d between the two means, unitless, zero when the means agree."""
    pooled_std = np.sqrt(0.5 * (np.var(y_true, ddof=1) + np.var(y_pred, ddof=1)))
    if pooled_std == 0:
        return 0.0
    return float(abs(np.mean(y_true) - np.mean(y_pred)) / pooled_std)


def nll(y_true, y_pred):
    """Gaussian negative log-likelihood.

    For a point estimate sigma is one global value from the residuals, which
    makes this a monotone function of RMSE. For y_pred of shape (n_timesteps,
    n_samples) mu and sigma are estimated per timestep instead; that path is
    unverified against a reference implementation and unused here.
    """
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
    # A near-constant residual, which the cache's rounding to four decimals can
    # produce, leaves sigma at around 1e-15 and makes logpdf astronomical.
    sigma_y = float(np.std(y_true))
    if sigma_y > 0 and sigma < 1e-6 * sigma_y:
        return 0.0
    return float(-np.mean(norm.logpdf(y_true, loc=y_pred, scale=sigma)))


def smae(y_true, y_pred, n_freqs=50):
    """Mean absolute difference between the two normalised Lomb-Scargle power
    spectra, so it compares relative frequency content independently of
    amplitude. Needs the full series, like `acf` and `dtw`."""
    n = len(y_true)
    t = np.arange(n, dtype=np.float64)
    freqs = np.linspace(2 * np.pi / n, np.pi, n_freqs)

    psd_true = lombscargle(t, y_true - np.mean(y_true), freqs)
    psd_pred = lombscargle(t, y_pred - np.mean(y_pred), freqs)

    psd_true = psd_true / (psd_true.sum() + 1e-10)
    psd_pred = psd_pred / (psd_pred.sum() + 1e-10)

    return float(np.mean(np.abs(psd_true - psd_pred)))


def nd(y_true, y_pred):
    """Summed absolute error over sum(|y_true|)."""
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def pfc(y_true, y_pred, tolerance=0.10):
    """Percentage of positions within `tolerance` relative error. HIGHER is
    better, unlike every other error metric here."""
    rel_err = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10)
    return float(np.mean(rel_err <= tolerance) * 100)
