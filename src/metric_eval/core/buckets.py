import numpy as np


def bucket_mean(
    raw_scores: dict,
    keys: list,
    metrics: list[str],
    subjects: list[str],
    ) -> dict[str, dict[str, float | None]]:
    """Mean over `keys` per (metric, subject), leaving None values out of the mean.

    A (metric, subject) whose every value is None is itself None. `keys` are the
    rates in one bucket and `subjects` the distortions or algorithms being
    compared.
    """
    out: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        out[metric] = {}
        for subject in subjects:
            vals = [raw_scores[k][metric][subject] for k in keys
                    if raw_scores[k].get(metric, {}).get(subject) is not None]
            out[metric][subject] = float(np.mean(vals)) if vals else None
    return out


def subjects_present(raw_scores: dict, keys: list, metrics: list[str], order: list[str]) -> list[str]:
    """The subjects appearing at any key, in `order`."""
    present: set[str] = set()
    for k in keys:
        for metric in metrics:
            present.update(raw_scores[k].get(metric, {}).keys())
    return [s for s in order if s in present]
