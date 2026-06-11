"""Synthetic ground truth: a multivariate time series where each series
follows a different base "shape" (sine + trend, sum of sines, smoothed random
walk, blended square wave - see BASE_SHAPES), with randomized phase/
frequency/trend/noise per series. This mimics the kind of variety seen across
channels in a real multivariate recording (e.g. EEG), while every value
remains fully known and reproducible.

Used by build_datasets.py for any ExperimentSpec with source="synthetic".
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

# ── default settings ────────────────────────────────────────────────────────
SEED = 42
N_TIMESTEPS = 200  # series length (timesteps)
N_SERIES = 20      # number of series/channels - matches the imputegap_benchmark dataset


# ── one base "shape" per series ─────────────────────────────────────────────
# Each function returns an array of length n. Per-series randomization
# (phase, frequency, trend slope, noise) is drawn from `rng`, which is seeded
# differently for every series so results are reproducible per-series and
# independent of n_series.

def _shape_sine_trend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sine wave with a slow upward trend (the original single-series shape)."""
    t = np.linspace(0, 4 * np.pi, n)
    freq = rng.uniform(0.85, 1.15)
    phase = rng.uniform(0, 2 * np.pi)
    trend = rng.uniform(0.005, 0.025) * np.arange(n)
    return 10 * np.sin(freq * t + phase) + trend


def _shape_double_sine(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sum of two sines at different frequencies - a more irregular waveform."""
    t = np.linspace(0, 4 * np.pi, n)
    f1, f2 = rng.uniform(0.8, 1.2), rng.uniform(2.5, 3.5)
    p1, p2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    return 7 * np.sin(f1 * t + p1) + 3 * np.sin(f2 * t + p2)


def _shape_random_walk(n: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothed random walk with a small drift - a non-periodic shape."""
    walk = np.cumsum(rng.normal(0, 0.6, n))
    smoothed = uniform_filter1d(walk, size=5)
    drift = rng.uniform(-0.01, 0.01) * np.arange(n)
    return smoothed + drift


def _shape_square_blend(n: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothed square wave blended with a sine - sharper transitions."""
    t = np.linspace(0, 4 * np.pi, n)
    square = np.sign(np.sin(rng.uniform(0.4, 0.6) * t + rng.uniform(0, 2 * np.pi)))
    smooth_square = uniform_filter1d(square, size=8)
    sine = 4 * np.sin(t * rng.uniform(0.9, 1.1))
    return 6 * smooth_square + sine


# Cycled round-robin across series (series 0, 4, 8, ... use the first shape, etc.)
BASE_SHAPES = [_shape_sine_trend, _shape_double_sine, _shape_random_walk, _shape_square_blend]


def generate(n_timesteps: int = N_TIMESTEPS, n_series: int = N_SERIES, seed: int = SEED) -> np.ndarray:
    """Return ground truth of shape (n_timesteps, n_series).

    Each series gets its own RNG (seed + series index) so results are
    reproducible per-series and independent of n_series.
    """
    series_list = []
    for i in range(n_series):
        rng = np.random.default_rng(seed + i)
        shape_fn = BASE_SHAPES[i % len(BASE_SHAPES)]
        offset = 15 + rng.uniform(-2, 2)
        noise = rng.normal(0, 0.4, n_timesteps)
        series_list.append(shape_fn(n_timesteps, rng) + offset + noise)
    return np.stack(series_list, axis=1)  # (n_timesteps, n_series)
