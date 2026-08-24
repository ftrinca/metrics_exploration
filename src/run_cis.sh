#!/usr/bin/env bash
# Experiment 3 — CIS: can complementary metrics be combined into one score
# without losing what made them complementary?
#
# Reads the scores Experiment 2 cached, so run_algo_ranking.sh has to have
# finished first. The supporting analyses also read Experiment 1's cache where
# it exists, and say so in the report when it does not.
#
# --force is accepted for symmetry with the other two scripts and does
# nothing: this experiment caches no intermediate state of its own.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

stage "cis.cis  — gate, score, validation, rejected constructions"
"$PYTHON" -m cis.cis

echo
echo "Experiment 3 done."
echo "  reports/cis/cis_validation_summary.txt"
echo "  reports/cis/cis_supporting_experiments.txt"
echo "  plots/cis/"
