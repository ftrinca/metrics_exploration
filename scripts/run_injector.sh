#!/usr/bin/env bash
# Experiment 1 — Injector: can the metrics tell different KINDS of damage apart?
#
# Two pipelines over the same eight distortions. Damage reactivity holds the
# damage fixed at config.TARGET_DAMAGE and varies the kind, across every
# missingness geometry and rate. Damage response fixes the geometry and rate and
# varies the damage across config.DAMAGE_LEVELS instead.
#
# Minutes, not hours — the algorithms are simulated rather than run.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

# Checks that every distortion can be solved to the target and that the declared
# structural invariants hold, on synthetic data under all three geometries.
# Needs neither ImputeGAP nor any cache, so a failure here means nothing
# downstream is trustworthy and set -e stops the run.
if [ -z "$SKIP_SELFTEST" ]; then
    stage "injector.selftest  (synthetic data, no ImputeGAP needed)"
    "$PYTHON" -m metric_eval.experiments.injector.selftest
fi

require_imputegap

stage "reactivity.calibrate  — solve severities to a common damage"
"$PYTHON" -m metric_eval.experiments.injector.reactivity.calibrate $FORCE

stage "reactivity.build  — apply the solved severities"
"$PYTHON" -m metric_eval.experiments.injector.reactivity.build $FORCE

stage "reactivity.score  — every metric, every distortion"
"$PYTHON" -m metric_eval.experiments.injector.reactivity.score $FORCE

stage "reactivity.aggregate  — tables, heatmaps, invariance checks"
"$PYTHON" -m metric_eval.experiments.injector.reactivity.aggregate

stage "response.build  — the same eight across seven damage levels"
"$PYTHON" -m metric_eval.experiments.injector.response.build $FORCE

stage "response.score"
"$PYTHON" -m metric_eval.experiments.injector.response.score $FORCE

stage "response.aggregate"
"$PYTHON" -m metric_eval.experiments.injector.response.aggregate

# The second calibration pass of the thesis: MAE and ND are pinned by the MAE
# target, so their columns in the response grid are read off this one.
stage "reactivity RMSE-target pass  — calibrate, build, score"
"$PYTHON" -m metric_eval.experiments.injector.reactivity.calibrate --damage-metric rmse $FORCE
"$PYTHON" -m metric_eval.experiments.injector.reactivity.build --damage-metric rmse $FORCE
"$PYTHON" -m metric_eval.experiments.injector.reactivity.score --damage-metric rmse $FORCE

stage "injector.summary  — response grid and redundancy correlations"
"$PYTHON" -m metric_eval.experiments.injector.summary

stage "injector.panels  — the eight distortion panels"
"$PYTHON" -m metric_eval.experiments.injector.panels

echo
echo "Experiment 1 done."
echo "  outputs/reports/injector/reactivity/     raw tables, agreement, invariance checks"
echo "  outputs/reports/injector/response/       flat / monotonic / non-monotonic per metric"
echo "  outputs/reports/injector/summary.txt     response grid values, redundancy"
echo "  outputs/plots/injector/reactivity/  and  plots/injector/response/"
