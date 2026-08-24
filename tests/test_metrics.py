"""Algebraic identities the metric implementations must satisfy.

These are not accuracy tests. Each one asserts a property that the thesis's
arguments depend on, so that a future change to core/metrics.py cannot quietly
invalidate a claim:

  Redundancy  ND is a fixed multiple of MAE and nRMSE a fixed multiple of
              RMSE, because both divide by a quantity that depends only on the
              ground truth. MSE is RMSE squared. The claim that these four are
              two signals rather than four rests on exactly that.

  Blind spots Pearson is invariant to any positive-slope affine transform;
              WD, JSD and KLD are invariant to a permutation of the values;
              Bland-Altman and CDT are invariant to anything that preserves
              the mean. The blind-spot results of Experiment 1 are these
              identities, measured.

  Weighting   With the error concentrated at a fraction p of positions,
              RMSE / MAE is exactly 1/sqrt(p), and with the error constant it
              is exactly 1. That bracket is what the MAE-and-RMSE pair
              measures between them.

Run with `pytest` from the repository root, or directly:

    python tests/test_metrics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import core.metrics as m  # noqa: E402

RTOL = 1e-9
N = 400


@pytest.fixture
def truth():
    """A z-scored series with periodicity and autocorrelation, so the
    temporal metrics see something structured rather than white noise."""
    rng = np.random.default_rng(0)
    t = np.arange(N)
    x = np.sin(2 * np.pi * t / 37) + 0.4 * np.sin(2 * np.pi * t / 11)
    x = x + np.cumsum(rng.normal(0, 0.05, N)) + rng.normal(0, 0.3, N)
    return (x - x.mean()) / x.std()


# ── redundancy identities ────────────────────────────────────────────────

def test_nd_is_a_fixed_multiple_of_mae(truth):
    """ND divides the summed absolute error by sum|y|, which depends only on
    the truth, so ND / MAE is the same for any reconstruction of one series."""
    rng = np.random.default_rng(1)
    ratios = []
    for scale in (0.1, 0.5, 1.0, 2.0):
        pred = truth + rng.normal(0, scale, N)
        ratios.append(m.nd(truth, pred) / m.mae(truth, pred))
    assert np.allclose(ratios, ratios[0], rtol=RTOL)
    assert np.isclose(ratios[0], 1.0 / np.mean(np.abs(truth)), rtol=RTOL)


def test_nrmse_is_a_fixed_multiple_of_rmse(truth):
    rng = np.random.default_rng(2)
    ratios = []
    for scale in (0.1, 0.5, 1.0, 2.0):
        pred = truth + rng.normal(0, scale, N)
        ratios.append(m.nrmse(truth, pred) / m.rmse(truth, pred))
    assert np.allclose(ratios, ratios[0], rtol=RTOL)
    assert np.isclose(ratios[0], 1.0 / np.std(truth), rtol=RTOL)


def test_mse_is_rmse_squared(truth):
    rng = np.random.default_rng(3)
    pred = truth + rng.normal(0, 0.4, N)
    assert np.isclose(m.mse(truth, pred), m.rmse(truth, pred) ** 2, rtol=RTOL)


# ── error weighting ──────────────────────────────────────────────────────

def test_rmse_equals_mae_when_every_error_is_the_same(truth):
    """A constant offset makes every error identical, which is the one case
    where the two coincide. It is the floor of the RMSE / MAE ratio."""
    pred = truth + 0.5
    assert np.isclose(m.rmse(truth, pred), m.mae(truth, pred), rtol=RTOL)


@pytest.mark.parametrize("p", [0.02, 0.05, 0.10, 0.25])
def test_rmse_over_mae_is_one_over_root_p_for_concentrated_error(truth, p):
    """With the error confined to a fraction p of positions and zero
    elsewhere, MAE = p*g and RMSE = g*sqrt(p), so the ratio is 1/sqrt(p)
    whatever the magnitude g."""
    rng = np.random.default_rng(4)
    k = int(round(p * N))
    pred = truth.copy()
    where = rng.choice(N, size=k, replace=False)
    pred[where] += 3.0
    assert np.isclose(m.rmse(truth, pred) / m.mae(truth, pred),
                      1.0 / np.sqrt(k / N), rtol=1e-12)


# ── blind spots ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("a,b", [(1.0, 0.5), (1.6, 0.0), (2.5, -1.2), (0.4, 3.0)])
def test_pearson_cannot_see_a_positive_slope_affine_transform(truth, a, b):
    """corr(y, a*y + b) = 1 for every a > 0. This is the whole of Pearson's
    blind spot to bias and to rescaling."""
    assert np.isclose(m.pearson(truth, a * truth + b), 1.0, atol=1e-12)


def test_r2_does_see_what_pearson_cannot(truth):
    """R^2 = 1 - alpha^2 under an offset of alpha standard deviations, so
    unlike Pearson it falls as the offset grows."""
    for alpha in (0.25, 0.5, 0.75):
        pred = truth + alpha * np.std(truth)
        assert np.isclose(m.r2(truth, pred), 1.0 - alpha ** 2, rtol=1e-6)


def test_distribution_metrics_cannot_see_a_permutation(truth):
    """A permutation leaves the multiset of values unchanged, so any statistic
    computed from the empirical distribution alone cannot move."""
    pred = np.random.default_rng(5).permutation(truth)
    assert m.wd(truth, pred) == pytest.approx(0.0, abs=1e-12)
    assert m.jsd(truth, pred) == pytest.approx(0.0, abs=1e-12)
    assert m.kld(truth, pred) == pytest.approx(0.0, abs=1e-12)


def test_mean_difference_metrics_cannot_see_a_permutation(truth):
    pred = np.random.default_rng(6).permutation(truth)
    mean_diff, _ = m.ba(truth, pred)
    assert mean_diff == pytest.approx(0.0, abs=1e-12)
    assert m.cdt(truth, pred) == pytest.approx(0.0, abs=1e-12)


def test_mean_difference_metrics_cannot_see_a_rescaling(truth):
    """Expanding around the mean leaves the mean where it was."""
    mu = truth.mean()
    pred = mu + 1.6 * (truth - mu)
    mean_diff, _ = m.ba(truth, pred)
    assert mean_diff == pytest.approx(0.0, abs=1e-12)
    assert m.cdt(truth, pred) == pytest.approx(0.0, abs=1e-12)


def test_temporal_metrics_do_see_a_permutation(truth):
    """The counterpart to the above: destroying the order has to register
    somewhere, or no metric in the set detects reordering at all."""
    pred = np.random.default_rng(7).permutation(truth)
    assert m.acf(truth, pred) > 0.0
    assert m.dtw(truth, pred) > 0.0


# ── sanity ───────────────────────────────────────────────────────────────

def test_a_perfect_reconstruction_scores_perfectly(truth):
    assert m.mae(truth, truth) == pytest.approx(0.0, abs=1e-12)
    assert m.rmse(truth, truth) == pytest.approx(0.0, abs=1e-12)
    assert m.wd(truth, truth) == pytest.approx(0.0, abs=1e-12)
    assert m.pearson(truth, truth) == pytest.approx(1.0, abs=1e-12)
    assert m.r2(truth, truth) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("metric", ["mae", "rmse", "mse", "wd", "dtw"])
def test_lower_is_better_metrics_rise_with_noise(truth, metric):
    rng = np.random.default_rng(8)
    fn = getattr(m, metric)
    values = [fn(truth, truth + rng.normal(0, s, N)) for s in (0.1, 0.3, 0.6, 1.0)]
    assert all(b > a for a, b in zip(values, values[1:])), values


@pytest.mark.parametrize("metric", ["pearson", "r2", "mi"])
def test_higher_is_better_metrics_fall_with_noise(truth, metric):
    rng = np.random.default_rng(9)
    fn = getattr(m, metric)
    values = [fn(truth, truth + rng.normal(0, s, N)) for s in (0.1, 0.3, 0.6, 1.0)]
    assert all(b < a for a, b in zip(values, values[1:])), values


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
