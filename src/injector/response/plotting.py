from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.metric_config import CATEGORIES
from injector.config import DISTORTION_LABEL, DISTORTION_NAMES, METRIC_LABEL


def plot_response(levels, series_by_distortion, category, output_path):
    """Draw one panel per metric in a category, one line per distortion.

    series_by_distortion[distortion][metric] is the list of raw values, one per
    damage level.
    """
    metrics = CATEGORIES[category]
    n = len(metrics)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.3 * nrows), squeeze=False)

    for k, metric in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        for d in DISTORTION_NAMES:
            ys = series_by_distortion.get(d, {}).get(metric)
            if ys is None or all(v is None for v in ys):
                continue
            xs = [x for x, y in zip(levels, ys) if y is not None]
            vs = [y for y in ys if y is not None]
            flat = (max(vs) - min(vs)) <= 1e-12
            ax.plot(xs, vs, marker="o", markersize=3.4, linewidth=1.5,
                    linestyle=":" if flat else "-",
                    label=DISTORTION_LABEL[d] + (" (flat)" if flat else ""))
        ax.set_title(METRIC_LABEL[metric], fontsize=10)
        ax.set_xlabel("damage (MAE / σ)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(f"{category} — every distortion on one damage axis", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   plot -> {output_path}")
