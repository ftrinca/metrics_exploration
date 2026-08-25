#!/usr/bin/env bash
# Experiment 3 — CIS: can complementary metrics be combined into one score
# without losing what made them complementary?
#
# Reads the scores Experiment 2 cached, so run_algo_ranking.sh has to have
# finished first. The supporting analyses also read Experiment 1's cache where
# it exists, and say so in the report when it does not. Nothing here calls an
# imputation algorithm, so ImputeGAP is not required and the whole run takes
# well under a minute.
#
# --force is accepted for symmetry with the other two scripts and does
# nothing: this experiment caches no intermediate state of its own.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

stage "cis  — stability gate, composite score, validation, component sweep"
"$PYTHON" -m cis

echo
echo "Experiment 3 done."
echo "  reports/cis/cis_validation_summary.txt      agreement with the 8-metric panel"
echo "  reports/cis/cis_supporting_experiments.txt  the design choices behind CIS"
echo "  plots/cis/cis_gate_distribution.png"
echo "  plots/cis/cis_gated_vs_consensus.png"
echo "  plots/cis/cis_reactivity_response.png"
