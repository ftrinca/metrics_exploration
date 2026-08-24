import argparse
import json
import sys

import numpy as np

from algo_ranking import algorithms


def main() -> None:
    """Run one algorithm on a pre-computed (y_true, mask) pair and write its reconstruction.

    Spawned by build.py, one fresh process per algorithm. Invoked as
    `python -m algo_ranking._run_algorithm`, not run directly.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True,
        help="JSON file with {'y_true': [...], 'mask': [...]}, native (n_series, n_timesteps) orientation",
    )
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", required=True,
        help="Where to write {'result': [...]} (native orientation) - key absent if the algorithm failed",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    y_true_t = np.array(data["y_true"])
    mask_t = np.array(data["mask"])

    results = algorithms.build(y_true_t, mask_t, seed=args.seed, only={args.algo})

    out = {}
    if args.algo in results:
        out["result"] = results[args.algo].tolist()
    with open(args.output, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
