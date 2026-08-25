import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from core.ranking import rank_algorithms
from algo_ranking.config import PATTERNS

from cis.config import (ALGO_COLORS, ALGO_NAMES, FLAT_THRESHOLD,
                        PATTERN_COLORS, UNSTABLE_THRESHOLD)


def _scatter_ratio(ax, rows: list[dict], field: str) -> None:
    """Jittered scatter of one field, grouped by pattern and coloured by algorithm."""
    rng = np.random.default_rng(0)
    for pi, pattern in enumerate(PATTERNS):
        for ai, algo in enumerate(ALGO_NAMES):
            vals = [r[field] for r in rows if r["pattern"] == pattern and r["algo"] == algo]
            if not vals:
                continue
            vals = np.array(vals)
            x = pi + rng.uniform(-0.32, 0.32, size=len(vals)) + (ai - 2.5) * 0.045
            ax.scatter(x, vals, color=ALGO_COLORS[algo], s=22, alpha=0.75,
                       edgecolor="white", linewidth=0.3, label=algo if pi == 0 else None)
    ax.set_xticks(range(len(PATTERNS)))
    ax.set_xticklabels(PATTERNS)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(-0.5, len(PATTERNS) - 1 + 0.8)


def plot_gate_distribution(rows: list[dict], output_path: str) -> None:
    """The IQR ratio for every scenario and algorithm, with both thresholds marked."""
    fig, ax = plt.subplots(figsize=(8, 6))

    _scatter_ratio(ax, rows, "iqr_ratio")
    ax.set_ylim(bottom=-0.02)
    ax.axhline(FLAT_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.text(0.05, FLAT_THRESHOLD, f"flat threshold ({FLAT_THRESHOLD})",
            va="bottom", ha="left", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    ax.axhline(UNSTABLE_THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.text(0.05, UNSTABLE_THRESHOLD, f"unstable threshold ({UNSTABLE_THRESHOLD})",
            va="bottom", ha="left", fontsize=8, color="gray",
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("IQR(reconstruction) / IQR(truth), masked positions (symlog scale)")
    ax.set_title("Stability gate across all scenarios and algorithms\n"
                 "(near 0 = flat or collapsed, far above 1 = unstable)")
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {output_path}")


def plot_cis_vs_consensus(rows: list[dict], output_path: str) -> None:
    """CIS rank against consensus rank, one point per (scenario, algorithm)."""
    by_scenario: dict[tuple, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault((r["dataset"], r["pattern"], r["rate"]), []).append(r)

    xs_gated, ys_gated = [], []
    xs_survive, ys_survive, colors_survive = [], [], []

    for key, scenario_rows in by_scenario.items():
        pattern = key[1]
        cis_vals = {r["algo"]: r["cis"] for r in scenario_rows}
        cis_rank = rank_algorithms(cis_vals, direction="higher")
        for r in scenario_rows:
            if r["passes_gate"]:
                xs_survive.append(r["consensus_rank"])
                ys_survive.append(cis_rank[r["algo"]])
                colors_survive.append(PATTERN_COLORS[pattern])
            else:
                xs_gated.append(r["consensus_rank"])
                ys_gated.append(cis_rank[r["algo"]])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs_gated, ys_gated, c="lightgray", s=45, alpha=0.7,
               edgecolor="white", linewidth=0.3)
    ax.scatter(xs_survive, ys_survive, c=colors_survive, s=45, alpha=0.8,
               edgecolor="white", linewidth=0.3)
    lo, hi = 1, len(ALGO_NAMES)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
                       label="fails gate (excluded)", markersize=9)]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                        label=f"survives, {p}", markersize=9)
                for p, c in PATTERN_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    ax.set_xlabel("8-metric global consensus rank (1 = best)")
    ax.set_ylabel("CIS rank (1 = best)")
    ax.set_title("CIS rank vs. full 8-metric consensus rank, gate applied")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {output_path}")


def plot_reactivity_response(reactivity: dict, output_path: str) -> None:
    """CIS and its four components across Experiment 1's eight calibrated distortions.

    Left panel: CIS per distortion, one marker per condition. Right panel: the
    four components, where a component that barely moves shows up as a flat line.
    """
    per_condition = reactivity["per_condition"]
    mean_cis = reactivity["mean_per_distortion"]
    names = sorted(mean_cis, key=mean_cis.get)
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for condition, payload in per_condition.items():
        pattern = condition.split("/")[0]
        ax1.scatter(x, [payload["cis"][d] for d in names],
                    color=PATTERN_COLORS.get(pattern, "gray"), s=26, alpha=0.65,
                    edgecolor="white", linewidth=0.3)
    ax1.plot(x, [mean_cis[d] for d in names], color="black", linewidth=1.6,
             marker="o", markersize=5, label="mean over conditions")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("CIS")
    ax1.set_title("CIS under equal damage\n(every distortion at the same MAE)")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=p,
                      markersize=8) for p, c in PATTERN_COLORS.items()]
    handles.append(Line2D([0], [0], color="black", label="mean"))
    ax1.legend(handles=handles, fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    styles = {"M": ("#4C72B0", "MAE"), "D": ("#DD8452", "WD"),
              "T": ("#55A868", "DTW"), "I": ("#C44E52", "MI")}
    for comp, (color, source) in styles.items():
        vals = [float(np.mean([p["components"][d][comp] for p in per_condition.values()]))
                for d in names]
        ax2.plot(x, vals, color=color, marker="o", markersize=5, linewidth=1.6,
                 label=f"{comp}  (from {source})")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("component value")
    ax2.set_title("The four components over the same eight")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {output_path}")


