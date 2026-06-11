"""Runs a fixed list of real ImputeGAP imputation algorithms against a
NaN-contaminated copy of `y_true`, and returns each algorithm's
reconstruction. Used for any ExperimentSpec with
reconstruction="imputation_algorithms" (see experiment_config.py).

ImputeGAP's own scoring (RMSE, MAE, MI, CORRELATION) is printed here as a
quick sanity check while running. The full evaluation (all metrics from
metrics.py) happens later, in evaluate_metrics.py, on the JSON file produced
by build_datasets.py.

TODO (open items - see metric_verification.md "Next Steps"):
  - Check whether ImputeGAP exposes any *probabilistic* algorithms
    (e.g. BayOTIDE, CSDI, GP-VAE, PRISTI) and, if so, what `recov_data`
    looks like for them (single point estimate, or multiple posterior
    samples?). CRPS and NLL both have a "2D / posterior samples" branch
    in metrics.py that is currently untested because none of the
    algorithms below produce that shape.
  - If posterior samples are available, dataset_io.matrix_to_lists() would
    need a 3rd dimension (samples) - the current JSON format only
    supports [series][timestep], i.e. one value per point.
"""

import os
import time
from contextlib import contextmanager

import numpy as np

from imputegap.recovery.imputation import Imputation


@contextmanager
def _suppress_c_stdout():
    """Temporarily silence terminal output from C/C++ libraries.

    Some imputation algorithms are written in C/C++ and print debug
    messages directly to the terminal. Python's normal `verbose=False`
    settings cannot catch those - only redirecting the actual file
    descriptor can. This context manager does that, then restores normal
    printing afterwards. It is purely cosmetic (keeps the console clean)
    and has no effect on the results.
    """
    old_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        os.dup2(old_fd, 1)
        os.close(devnull)
        os.close(old_fd)


# ── algorithms to benchmark ─────────────────────────────────────────────────
# Each entry is (display name, family, the ImputeGAP class that implements it).
#
# Families included:
#   - Statistics:        very simple baselines (fill with 0 / mean / etc.).
#                         Why: cheap sanity-check baselines - any "real"
#                         algorithm should beat these.
#   - Matrix Completion:  exploit correlations BETWEEN series to fill gaps.
#                         Why: the main "interesting" algorithms for
#                         multivariate data like EEG.
#   - Pattern Search:     STMVL looks for similar temporal/spatial patterns
#                         elsewhere in the data.
#
# TODO: this list contains only DETERMINISTIC algorithms (one fixed value per
#  missing point). No probabilistic / deep-learning algorithms are included
#  yet - adding one (if available in ImputeGAP) is needed to exercise the
#  CRPS/NLL "posterior samples" branch (see at top of file).
ALGORITHMS = [
    # (label, family, imputer_class)
    ("ZeroImpute",        "Statistics",        Imputation.Statistics.ZeroImpute),
    ("MeanImpute",        "Statistics",        Imputation.Statistics.MeanImpute),
    ("MeanImputeBySeries","Statistics",        Imputation.Statistics.MeanImputeBySeries),
    ("MinImpute",         "Statistics",        Imputation.Statistics.MinImpute),
    ("Interpolation",     "Statistics",        Imputation.Statistics.Interpolation),
    ("CDRec",             "Matrix Completion", Imputation.MatrixCompletion.CDRec),
    ("IterativeSVD",      "Matrix Completion", Imputation.MatrixCompletion.IterativeSVD),
    ("SoftImpute",        "Matrix Completion", Imputation.MatrixCompletion.SoftImpute),
    ("SVT",               "Matrix Completion", Imputation.MatrixCompletion.SVT),
    ("SPIRIT",            "Matrix Completion", Imputation.MatrixCompletion.SPIRIT),
    ("GROUSE",            "Matrix Completion", Imputation.MatrixCompletion.GROUSE),
    ("STMVL",             "Pattern Search",    Imputation.PatternSearch.STMVL),
]


def build(y_true: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Return {algo_name: array(n_timesteps, n_series)} - the reconstruction
    produced by each algorithm in ALGORITHMS, run on a copy of y_true with
    NaNs at every position where mask is True.

    Algorithms that error out, or leave NaNs behind, are skipped - they will
    simply be absent from the returned dict (one less algorithm for
    evaluate_metrics.py to compare).
    """
    ts_m = np.where(mask, np.nan, y_true)
    results: dict[str, np.ndarray] = {}

    print(f"{'Algorithm':<24} {'Family':<20} {'Time (s)':>10}  "
          f"{'RMSE':>10}  {'MAE':>10}  {'MI':>10}  {'CORRELATION':>12}")
    print("-" * 104)

    for name, family, AlgoClass in ALGORITHMS:
        try:
            imputer = AlgoClass(ts_m)
            imputer.logs = False
            imputer.verbose = False

            # Run the imputation, timing how long it takes.
            # Timing rationale: purely informational (useful for discussing
            # the speed/accuracy trade-off of each algorithm). It has no
            # effect on any metric and is not used downstream.
            t0 = time.perf_counter()
            with _suppress_c_stdout():
                imputer.impute()
            elapsed = time.perf_counter() - t0

            # Safety check: an algorithm that leaves NaNs behind has not
            # actually filled in the missing values, so this is treated as a
            # failure.
            if np.isnan(imputer.recov_data).any():
                raise ValueError("imputed data contains NaN")

            # Compute ImputeGAP's built-in metrics (RMSE, MAE, MI, CORRELATION).
            imputer.score(y_true, imputer.recov_data, verbose=False)
            rmse        = imputer.metrics.get("RMSE", float("nan"))
            mae         = imputer.metrics.get("MAE", float("nan"))
            mi          = imputer.metrics.get("MI", float("nan"))
            correlation = imputer.metrics.get("CORRELATION", float("nan"))

            print(f"{name:<24} {family:<20} {elapsed:>10.4f}  "
                  f"{rmse:>10.4f}  {mae:>10.4f}  {mi:>10.4f}  {correlation:>12.4f}")

            results[name] = imputer.recov_data

        except Exception as e:
            print(f"{name:<24} {family:<20} {'ERROR':>10}  ({e})")

    return results
