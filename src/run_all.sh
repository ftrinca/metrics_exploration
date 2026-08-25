#!/usr/bin/env bash
# All three experiments, in the order their caches depend on each other:
# Experiment 3 reads Experiment 2's scores, and its supporting analyses read
# Experiment 1's.
#
# Dominated by Experiment 2's build stage.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

# _run_common.sh has already cd'd to src/, so these resolve whatever directory
# this script was invoked from.
./run_injector.sh "$@"
./run_algo_ranking.sh "$@"
./run_cis.sh "$@"

echo
echo "All three experiments done."
