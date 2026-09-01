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

stage "algorank.build  — SLOW: 1,440 algorithm runs over 144 scenarios"
"$PYTHON" -m metric_eval.experiments.algorank.build

stage "algorank.score  — metrics from the cached reconstructions"
"$PYTHON" -m metric_eval.experiments.algorank.score $FORCE

stage "algorank.aggregate  — consensus ranks, agreement matrices, heatmaps"
"$PYTHON" -m metric_eval.experiments.algorank.aggregate

stage "algorank.visualize  — reconstruction plots"
"$PYTHON" -m metric_eval.experiments.algorank.visualize

# The chapter-level tables and figures read the standard-deviation ratios of
# the CIS build, so this stage only runs once that cache exists.
stage "algorank.summarize  — chapter-level tables and figures"
if [ -d "outputs/time_series/cis" ]; then
    "$PYTHON" -m metric_eval.experiments.algorank.summarize
else
    echo "  SKIP: needs the CIS build cache; run ./run_cis.sh, then"
    echo "        python -m metric_eval.experiments.algorank.summarize"
fi

echo
echo "Experiment 2 done."
echo "  outputs/reports/algorank/<dataset>/"
echo "  outputs/plots/algorank/<dataset>/"
echo "  outputs/reports/algorank/algorank_experiments.txt  and  outputs/plots/algorank/summary/"
