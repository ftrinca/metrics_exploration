#!/usr/bin/env bash
# Experiment 1 — Injector: are the metrics sensitive to the KIND of damage?
#
# Roughly 10 minutes end to end. Two pipelines: the equal-damage experiment,
# where all eight distortions are solved to one damage target, and the sweep,
# where the same eight are solved to each of seven targets in turn.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

# Checks that every distortion can be solved to the target and that the
# declared structural invariants hold, on synthetic data. Neither ImputeGAP nor
# any cached output is involved, so a failure here means nothing downstream is
# trustworthy and the run is stopped rather than continued.
if [ -z "$SKIP_SELFTEST" ]; then
    stage "injector.selftest  (synthetic data, no ImputeGAP needed)"
    "$PYTHON" -m injector.selftest
fi

require_imputegap

stage "injector.calibrate  — solve severities to equal damage"
"$PYTHON" -m injector.calibrate $FORCE

stage "injector.build  — apply the solved severities"
"$PYTHON" -m injector.build $FORCE

stage "injector.score  — every metric, every distortion"
"$PYTHON" -m injector.score $FORCE

stage "injector.aggregate  — tables, heatmaps, invariance checks"
"$PYTHON" -m injector.aggregate

stage "injector.build_sweep  — the same eight across seven damage levels"
"$PYTHON" -m injector.build_sweep $FORCE

stage "injector.score_sweep"
"$PYTHON" -m injector.score_sweep $FORCE

stage "injector.aggregate_sweep"
"$PYTHON" -m injector.aggregate_sweep

echo
echo "Experiment 1 done."
echo "  reports/injector/equal_damage/   raw tables, agreement, invariance checks"
echo "  reports/injector/damage_sweep/   flat / monotonic / non-monotonic per metric"
echo "  plots/injector/"
