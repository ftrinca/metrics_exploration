"""Six hand-crafted "distortions" of a ground-truth matrix, each representing
a distinct imputation-algorithm failure mode. Used as a cheap, deterministic
reconstruction for any ExperimentSpec with reconstruction="synthetic_distortions"
(see experiment_config.py).

Each is computed independently per series, from that series' own ground
truth, so magnitudes (spike size, smoothing window, etc.) scale naturally
even though series differ in amplitude and shape.

  1. constant_offset  - perfect shape, systematic +4 bias
     Expect: Pearson=1, BA large mean_diff, MAE=RMSE=4
  2. random_spikes    - correct everywhere except ~5% outlier positions
     Expect: RMSE >> MAE (squared penalty amplifies spikes), PFC drops sharply
  3. time_shift       - series rolled forward by SHIFT steps (phase shift)
     Expect: DTW cheap, MAE/RMSE suffer; WD~=0 (same distribution); Pearson drops
  4. oversmoothed     - heavy moving-average flattens peaks and troughs
     Expect: ACF changes, WD shifts (narrower distribution), Pearson reasonable
  5. shuffled         - same values as ground truth but in random order
     Expect: WD=0, KLD~=0, CDT~=0; Pearson~=0, DTW huge, ACF collapses
  6. amplitude_scaled - oscillations x1.6 around the series mean; mean preserved
     Expect: Pearson=1, WD/NLL/KLD bad (wider distribution)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

SHIFT = 20  # steps for time_shift, ~1/5 of a 4*pi sine cycle over 200 timesteps
SEED = 42


def build(y_true: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    """Return {name: array(n_timesteps, n_series)} for all six distortions,
    computed independently per series."""
    n_timesteps, n_series = y_true.shape
    distorted = {name: np.empty_like(y_true) for name in (
        "constant_offset", "random_spikes", "time_shift",
        "oversmoothed", "shuffled", "amplitude_scaled",
    )}

    for i in range(n_series):
        series = y_true[:, i]
        rng = np.random.default_rng(seed + 1000 + i)
        std = float(np.std(series))

        distorted["constant_offset"][:, i] = series + 4.0

        spikes = series.copy()
        n_spikes = max(1, round(0.05 * n_timesteps))
        spike_idx = rng.choice(n_timesteps, size=n_spikes, replace=False)
        spike_sign = rng.choice([-1.0, 1.0], size=n_spikes)
        spikes[spike_idx] += spike_sign * 3.0 * std
        distorted["random_spikes"][:, i] = spikes

        distorted["time_shift"][:, i] = np.roll(series, SHIFT)
        distorted["oversmoothed"][:, i] = uniform_filter1d(series, size=25)
        distorted["shuffled"][:, i] = rng.permutation(series)

        mean = float(np.mean(series))
        distorted["amplitude_scaled"][:, i] = mean + (series - mean) * 1.6

    return distorted
