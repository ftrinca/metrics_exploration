import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from algo_ranking.config import (
    reconstruction_dir,
    DATASETS, PATTERNS, PLOT_DIR, PLOT_SERIES_INDEX,
    PLOT_WINDOW_TIMESTEPS, RATES, seed_dir,
)
from algo_ranking.plotting import plot_reconstruction


def _choose_window(mask_series: np.ndarray, window_size: int, n_timesteps: int) -> int:
    """Return a window start containing missing positions, or 0 when the series has none.

    Centres on the longest run of missing positions, since scattered and
    blackout each place one contiguous gap at a random start that a fixed offset
    has no guarantee of overlapping.
    """
    missing_idx = np.flatnonzero(mask_series)
    if missing_idx.size == 0:
        return 0

    gaps = np.flatnonzero(np.diff(missing_idx) > 1)
    run_starts = np.concatenate(([0], gaps + 1))
    run_ends = np.concatenate((gaps, [missing_idx.size - 1]))
    longest = np.argmax(run_ends - run_starts)
    run_start_pos = missing_idx[run_starts[longest]]
    run_end_pos = missing_idx[run_ends[longest]]

    center = (run_start_pos + run_end_pos) // 2
    start = center - window_size // 2
    return int(max(0, min(start, n_timesteps - window_size)))


def visualize_one(dataset: str, pattern: str, rate: float, seed: int = 0) -> None:
    """Plot one scenario's reconstructions for the series at config.PLOT_SERIES_INDEX."""
    data_path = os.path.join(seed_dir(dataset, pattern, rate, seed), "data.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Missing {data_path} - run algo_ranking/build.py for "
            f"dataset={dataset!r} pattern={pattern!r} rate={rate} seed={seed} first."
        )
    with open(data_path) as f:
        built = json.load(f)

    y_true = np.array(built["y_true"])
    mask = np.array(built["mask"])
    reconstructions = {
        name: np.array(built[name]) for name in built if name not in ("y_true", "mask")
    }

    n_series, n_timesteps = y_true.shape
    series_idx = PLOT_SERIES_INDEX if PLOT_SERIES_INDEX < n_series else 0

    window_size = min(PLOT_WINDOW_TIMESTEPS, n_timesteps)
    start = _choose_window(mask[series_idx], window_size, n_timesteps)
    end = min(start + window_size, n_timesteps)
    window = slice(start, end)

    plot_reconstruction(
        {name: arr[series_idx][window] for name, arr in reconstructions.items()},
        y_true[series_idx][window],
        mask[series_idx][window],
        title=(
            f"{dataset} {pattern} {rate:.0%} (series {series_idx + 1} of {n_series}, "
            f"timesteps {start}-{end} of {n_timesteps})"
        ),
        output_path=os.path.join(
            reconstruction_dir(dataset), f"{pattern}_{round(rate * 100):02d}pct.png"
        ),
    )


def visualize_phase(datasets: list[str], patterns: list[str], rates: list[float]) -> None:
    """Draw the reconstruction plots for every (dataset, pattern, rate)."""
    for dataset in datasets:
        print(f"=== dataset: {dataset} " + "=" * 50)
        for pattern in patterns:
            print(f"  -- pattern: {pattern} --")
            for rate in rates:
                visualize_one(dataset, pattern, rate)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algorithm Ranking (Part 2) — reconstruction plots.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--patterns", nargs="+", default=PATTERNS, choices=PATTERNS)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES, choices=RATES)
    args = parser.parse_args()

    visualize_phase(args.datasets, args.patterns, args.rates)
