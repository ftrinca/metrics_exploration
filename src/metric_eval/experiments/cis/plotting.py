import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metric_eval.experiments.algorank.config import ALGO_METRICS, ALGO_NAMES, PATTERNS, RATES, label

from metric_eval.experiments.cis.config import (ALGO_COLORS, CIS_METRICS, FLAT_THRESHOLD, PATTERN_COLORS,
                        UNSTABLE_THRESHOLD)
from metric_eval.experiments.cis.experiments import (RATE_BANDS, coverage, damage_sweep,
                             equal_damage_response, variation_preference)
from metric_eval.experiments.cis.gate import MIN_SURVIVORS, survivors

BAND_MARKERS = {"10-30": "o", "40-50": "s", "60-80": "^"}
BAND_COLORS = {"10-30": "#4C72B0", "40-50": "#937860", "60-80": "#C44E52"}
COMPONENT_COLORS = {"mae": "#4C72B0", "wd": "#55A868", "mi": "#DD8452",
                    "dtw": "#8172B2", "rmse": "#937860", "jsd": "#C44E52"}


def _save(fig, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   {path}")


def plot_gate_distribution(cache: dict[tuple, dict], output_path: str) -> None:
    """Standard-deviation ratio of every pair beside the interquartile ratio.

    The right panel is what rules the interquartile range out as the gate's
    instrument: the points on its floor are reconstructions it reads as constant
    while they carry hundreds of distinct values.
    """
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    rng = np.random.default_rng(0)

    for ax, field, title in ((left, "std_ratio", "standard-deviation ratio"),
                             (right, "iqr_ratio", "interquartile ratio")):
        for pi, pattern in enumerate(PATTERNS):
            for ai, algo in enumerate(ALGO_NAMES):
                values = [p[field][algo] for (_, pat, _), p in cache.items()
                          if pat == pattern and algo in p[field]]
                if not values:
                    continue
                x = pi + rng.uniform(-0.3, 0.3, len(values)) + (ai - 2.5) * 0.04
                ax.scatter(x, values, color=ALGO_COLORS[algo], s=14, alpha=0.7,
                           edgecolor="white", linewidth=0.25,
                           label=algo if (pi == 0 and ax is left) else None)
        ax.axhline(FLAT_THRESHOLD, color="gray", ls="--", lw=0.9)
        ax.axhline(UNSTABLE_THRESHOLD, color="gray", ls="--", lw=0.9)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylim(-0.004, 400)
        ax.set_xticks(range(len(PATTERNS)))
        ax.set_xticklabels(PATTERNS)
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, which="both")
    left.set_ylabel("ratio to the truth")
    left.legend(fontsize=6.5, ncol=6, loc="upper center",
                bbox_to_anchor=(1.05, 1.22), frameon=False)
    _save(fig, output_path)


def plot_coverage(cache: dict[tuple, dict], output_path: str) -> None:
    """Scenarios left rankable by the gate, per missing rate and geometry."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.26
    for pi, pattern in enumerate(PATTERNS):
        counts = [sum(1 for (_, pat, r), p in cache.items()
                      if pat == pattern and r == round(rate * 100)
                      and len(survivors(p)) >= MIN_SURVIVORS) for rate in RATES]
        ax.bar(np.arange(len(RATES)) + (pi - 1) * width, counts, width,
               color=PATTERN_COLORS[pattern], label=pattern)
    ax.set_xticks(range(len(RATES)))
    ax.set_xticklabels([f"{int(r * 100)}" for r in RATES])
    ax.set_xlabel("missing rate (%)")
    ax.set_ylabel("rankable scenarios")
    ax.set_ylim(0, 6.5)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _save(fig, output_path)


def plot_variation_axis(cache: dict[tuple, dict], output_path: str) -> None:
    """Where each ranking sits between flattening and keeping the variation."""
    preference = variation_preference(cache, gated=False)
    rankings = list(ALGO_METRICS) + ["CIS"]
    order = sorted(rankings, key=lambda m: preference[m]["60-80"])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for band in RATE_BANDS:
        ax.scatter([preference[m][band] for m in order], range(len(order)),
                   marker=BAND_MARKERS[band], color=BAND_COLORS[band], s=34,
                   label=f"{band}%", zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(["CIS" if m == "CIS" else label(m) for m in order])
    for tick, name in zip(ax.get_yticklabels(), order):
        if name == "CIS":
            tick.set_fontweight("bold")
    ax.set_xlabel("flattening $\\leftarrow$   preference   $\\rightarrow$ keeping the variation")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, output_path)


def plot_known_damage(conditions: dict, sweep: dict, output_path: str) -> None:
    """The composite and its components under damage of a known kind and size.

    The left panel is the design argument made visible: the pointwise component
    reads the eight distortions alike, each of the other two drops to nearly
    nothing on the kinds it cannot see, and the composite stays even.
    """
    response = equal_damage_response(conditions)
    order = sorted(response["per_distortion"], key=response["per_distortion"].get)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                                      gridspec_kw={"width_ratios": [1.35, 1]})

    x = np.arange(len(order))
    width = 0.2
    for i, metric in enumerate(CIS_METRICS):
        left.bar(x + (i - 1.5) * width,
                 [response["per_component"][d][metric] for d in order], width,
                 color=COMPONENT_COLORS.get(metric, "#B0B0B0"), label=label(metric))
    left.bar(x + 1.5 * width, [response["per_distortion"][d] for d in order], width,
             color="black", label="CIS")
    left.set_xticks(x)
    left.set_xticklabels(order, rotation=35, ha="right", fontsize=7.5)
    left.set_ylim(0, 0.88)
    left.set_ylabel("distance to the truth")
    left.legend(fontsize=7.5, ncol=4, loc="upper left")
    left.grid(True, axis="y", alpha=0.3)

    curves = damage_sweep(sweep)
    levels = next(iter(curves.values()))["damage_levels"]
    for distortion, info in curves.items():
        right.plot(levels, info["cis"], marker="o", ms=3, lw=1.1, label=distortion)
    right.set_xlabel("damage ($\\sigma$)")
    right.set_ylabel("CIS")
    right.legend(fontsize=6.5, ncol=2)
    right.grid(True, alpha=0.3)
    _save(fig, output_path)
