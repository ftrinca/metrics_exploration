#!/usr/bin/env bash
# All three experiments, in the order their caches depend on each other:
# Experiment 3 reads Experiment 2's scores, and its supporting analyses read
# Experiment 1's.
#
# Dominated by Experiment 2's build stage.

source "$(dirname "${BASH_SOURCE[0]}")/_run_common.sh"
parse_args "$@"

# _run_common.sh has already cd'd to the repository root, so these resolve
# whatever directory this script was invoked from.
./scripts/run_injector.sh "$@"
./scripts/run_algorank.sh "$@"
./scripts/run_cis.sh "$@"

# These two read caches the runs above produce, so they come last: the
# chapter-level summary needs the CIS build, the background figures need the
# ranking caches.
"$PYTHON" -m metric_eval.experiments.algorank.summarize
"$PYTHON" -m metric_eval.background.figures

echo
echo "All three experiments done."
