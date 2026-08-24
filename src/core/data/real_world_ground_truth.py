import numpy as np
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils


def generate(source: str, n_series: int) -> np.ndarray:
    """Return the first `n_series` channels of ImputeGAP dataset `source`, un-normalised.

    Shape is (n_timesteps, n_series); callers apply core.data.normalization themselves.
    """
    ts = TimeSeries()
    ts.load_series(
        utils.search_path(source),
        nbr_series=n_series,
        normalizer=None,
        verbose=False,
    )
    return ts.data
