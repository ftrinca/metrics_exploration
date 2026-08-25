#!/usr/bin/env bash
# Experiment 2 — Algorithm ranking: do the kept metrics agree on which algorithm is best?
#
# Six algorithms across 144 scenarios
# - The four deterministic ones run once per scenario
# - BRITS and MPIN run once per seed
# giving 1,440 runs in all, each in its own subprocess.
#
# The build takes about 17 hours, dominated by BRITS and MPIN; scoring and
# aggregating the whole cache take seconds by comparison. Every reconstruction
# is cached, so the later stages re-run freely and an interrupted build resumes
# from where it stopped.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"
require_imputegap

stage "algo_ranking.build  — SLOW: 1,440 algorithm runs over 144 scenarios"
"$PYTHON" -m algo_ranking.build

stage "algo_ranking.score  — metrics from the cached reconstructions"
"$PYTHON" -m algo_ranking.score $FORCE

stage "algo_ranking.aggregate  — consensus ranks, agreement matrices, heatmaps"
"$PYTHON" -m algo_ranking.aggregate

stage "algo_ranking.visualize  — reconstruction plots"
"$PYTHON" -m algo_ranking.visualize

echo
echo "Experiment 2 done."
echo "  reports/algo_ranking/<dataset>/"
echo "  plots/algo_ranking/<dataset>/"
