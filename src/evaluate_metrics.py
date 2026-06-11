"""Second pipeline step: for every experiment in experiment_config.ALL_SPECS
with an existing data file, writes the metrics report, the imputation plot,
the ranking report, and the ranking heatmap. Run after build_datasets.py.
"""

import os

from experiment_config import ALL_SPECS
from generate_reports import compute_all_scores, generate_metrics_report, load_data
from plot import plot_imputation, plot_ranking
from ranking import build_rank_matrix, generate_ranking_report

# ── paths ─────────────────────────────────────────────────────────────────────
# Which experiments exist (and where their JSON files live) is defined in
# experiment_config.ALL_SPECS - this script only needs to know where to put
# the generated reports/plots.
HERE       = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")
PLOT_DIR   = os.path.join(HERE, "plots")

# Which series (0-indexed) to draw in the imputation plot. Each dataset has
# n_series series (see ExperimentSpec.n_series, default 20); only one is
# plotted at a time for readability. Change this to inspect a different one.
PLOT_SERIES_INDEX = 1


if __name__ == "__main__":

    for spec in ALL_SPECS:
        name = spec.dataset_name

        if not os.path.exists(spec.output_path):
            print(f"-- {name}: no data file at {spec.output_path}")
            print("   Run build_datasets.py to generate it.\n")
            continue

        print(f"-- {name} --------------------------------------------------")

        y_true, imputations, mask = load_data(spec.output_path)

        # metric scores report
        generate_metrics_report(
            y_true, imputations,
            dataset_name=name,
            output_dir=REPORT_DIR,
            mask=mask,
        )

        # imputation plot. y_true/imputations/mask are 2D ([series][timestep])
        # for every dataset now - plot only one series (PLOT_SERIES_INDEX),
        # since plotting all of them on one axis would be unreadable.
        n_series = y_true.shape[0]
        series_idx = PLOT_SERIES_INDEX if PLOT_SERIES_INDEX < n_series else 0
        plot_imputation(
            {algo: arr[series_idx] for algo, arr in imputations.items()},
            y_true[series_idx],
            title=f"{name.replace('_', ' ')} (series {series_idx + 1} of {n_series})",
            output_path=os.path.join(PLOT_DIR, f"{name}_imputation.png"),
            mask=mask[series_idx] if mask is not None else None,
        )

        # ranking report + heatmap (averaged across series, see generate_reports._apply_metric)
        generate_ranking_report(
            y_true, imputations,
            output_dir=REPORT_DIR,
            dataset_name=name,
            mask=mask,
        )

        scores      = compute_all_scores(y_true, imputations, mask=mask)
        rank_matrix = build_rank_matrix(scores)
        plot_ranking(
            rank_matrix,
            title=f"Algorithm Ranking — {name.replace('_', ' ')}",
            output_path=os.path.join(PLOT_DIR, f"{name}_ranking.png"),
        )

        print()
