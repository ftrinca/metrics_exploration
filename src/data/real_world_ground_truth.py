"""Real-world ground truth: loads an ImputeGAP dataset, cut down to
`n_series` channels.

Used by build_datasets.py for any ExperimentSpec whose source is an ImputeGAP
dataset id (e.g. "eeg-alcohol").

Normalization is intentionally NOT applied here - see normalization.py, which
build_datasets.py applies uniformly to both synthetic and real-world ground
truth.
"""

import numpy as np
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


def generate(source: str, n_series: int) -> np.ndarray:
    """Return ground truth of shape (n_timesteps, n_series), raw (un-normalized)."""
    ts = TimeSeries()
    ts.load_series(
        utils.search_path(source),
        nbr_series=n_series,
        normalizer=None,
        verbose=False,
    )
    return ts.data
