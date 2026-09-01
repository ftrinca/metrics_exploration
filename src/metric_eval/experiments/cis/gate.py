import json
import os

from metric_eval.experiments.algorank.config import DATASETS
from metric_eval.experiments.cis.config import FLAT_THRESHOLD, UNSTABLE_THRESHOLD, cache_path

MIN_SURVIVORS = 3


def load_cache(datasets: list[str] = DATASETS) -> dict[tuple, dict]:
    """Every built scenario, keyed by (dataset, pattern, rate percent)."""
    out = {}
    for dataset in datasets:
        path = cache_path(dataset)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m cis.build --datasets {dataset}` first.")
        with open(path) as f:
            for key, payload in json.load(f).items():
                pattern, rate = key.split("|")
                out[(dataset, pattern, int(rate))] = payload
    return out


def passes(std_ratio: float,
           flat: float = FLAT_THRESHOLD,
           unstable: float = UNSTABLE_THRESHOLD) -> bool:
    """Whether a reconstruction keeps a plausible amount of the truth's variation."""
    return flat <= std_ratio <= unstable


def survivors(payload: dict, flat: float = FLAT_THRESHOLD,
              unstable: float = UNSTABLE_THRESHOLD) -> list[str]:
    """The algorithms of one scenario that pass the gate."""
    return [a for a, r in payload["std_ratio"].items() if passes(r, flat, unstable)]


def rankable(cache: dict[tuple, dict], flat: float = FLAT_THRESHOLD,
             unstable: float = UNSTABLE_THRESHOLD) -> list[tuple]:
    """The scenarios left with enough survivors for a ranking to mean anything.

    Below three survivors a rank correlation is either +1 or -1 whatever the
    values are, so those scenarios are reported as unranked.
    """
    return [k for k, p in cache.items()
            if len(survivors(p, flat, unstable)) >= MIN_SURVIVORS]
