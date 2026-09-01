#!/usr/bin/env bash
# Experiment 3 — CIS: can metrics that read different properties be combined
# into one score that no kind of damage escapes?
#
# Reads what Experiment 2 cached for the ranking half and what Experiment 1
# cached for the known-damage half, so run_algorank.sh and run_injector.sh
# both have to have finished first. Nothing here calls an imputation algorithm,
# so ImputeGAP is not required.
#
# The build stage derives the reference reconstruction and the gate ratios for
# all 144 scenarios and takes about a minute. It caches per dataset and skips
# whatever is already there, so an interrupted run resumes.
#
# --force rebuilds that cache.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

if [ -n "$FORCE" ] || [ ! -d "outputs/time_series/cis" ]; then
  stage "cis.build  — reference reconstruction and gate ratios"
  "$PYTHON" -m metric_eval.experiments.cis.build $FORCE
fi

stage "cis  — gate, composite score, variation axis, known damage"
"$PYTHON" -m metric_eval.experiments.cis

echo
echo "Experiment 3 done."
echo "  outputs/reports/cis/cis_experiments.txt"
echo "  outputs/plots/cis/cis_gate.pdf"
echo "  outputs/plots/cis/cis_coverage.pdf"
echo "  outputs/plots/cis/cis_variation_axis.pdf"
echo "  outputs/plots/cis/cis_known_damage.pdf"
