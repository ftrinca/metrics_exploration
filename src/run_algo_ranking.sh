#!/usr/bin/env bash
# Experiment 2 — Algorithm ranking: do the kept metrics agree on which
# algorithm is best?
#
# The build stage takes hours: six algorithms across 54 scenarios, each
# algorithm in its own subprocess. It caches every reconstruction, so the
# stages after it can be re-run freely, and an interrupted build resumes from
# where it stopped when the script is run again.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"
require_imputegap

stage "algo_ranking.build  — SLOW: 6 algorithms x 54 scenarios x seeds"
"$PYTHON" -m algo_ranking.build $FORCE

stage "algo_ranking.score  — metrics from the cached reconstructions"
"$PYTHON" -m algo_ranking.score $FORCE

stage "algo_ranking.aggregate  — consensus ranks, agreement matrices, heatmaps"
"$PYTHON" -m algo_ranking.aggregate

# Not part of the ranking: this is the human-readable check on HOW an algorithm
# is wrong, which the rank heatmap cannot show.
stage "algo_ranking.visualize  — reconstruction plots"
"$PYTHON" -m algo_ranking.visualize

echo
echo "Experiment 2 done."
echo "  reports/algo_ranking/<dataset>/"
echo "  plots/algo_ranking/<dataset>/"
