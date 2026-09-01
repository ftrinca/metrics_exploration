# Every script runs from the repository root whatever directory it was
# invoked from. src/ goes on PYTHONPATH so the metric_eval package resolves
# without an install; the output paths are anchored by metric_eval/paths.py
# and do not depend on the working directory.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-$(command -v python || command -v python3 || true)}"
if [ -z "$PYTHON" ] || ! "$PYTHON" -c "" >/dev/null 2>&1; then
    echo "No usable python interpreter (tried: ${PYTHON:-none})." >&2
    echo "Activate the virtualenv, or set PYTHON=/path/to/python and retry." >&2
    exit 1
fi

# The build stages are long, so a missing dependency is checked once up front
# rather than discovered hours in. ImputeGAP supplies both the datasets and the
# algorithms, so nothing beyond injector.selftest runs without it.
require_imputegap() {
    if ! "$PYTHON" -c "import imputegap" >/dev/null 2>&1; then
        echo "ImputeGAP is not importable by $PYTHON." >&2
        echo "Install the dependencies first:  pip install -r ../requirements.txt" >&2
        exit 1
    fi
}

# Announce each stage, so a run that stops somewhere says where.
stage() {
    echo
    echo "=================================================================="
    echo ">>> $*"
    echo "=================================================================="
}

# FORCE is passed on to the stages that cache; the aggregate stages take no
# --force because they recompute from the cached scores every time anyway.
FORCE=""
SKIP_SELFTEST=""
parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --force)         FORCE="--force" ;;
            --skip-selftest) SKIP_SELFTEST="1" ;;
            -h|--help)
                echo "usage: $(basename "$0") [--force] [--skip-selftest]"
                echo
                echo "  --force          redo every stage instead of skipping cached output"
                echo "  --skip-selftest  skip the synthetic-data check before Experiment 1"
                echo
                echo "These scripts run the whole pipeline. To work on a subset, call the"
                echo "stages directly, which take --datasets / --patterns / --rates:"
                echo "  python -m injector.reactivity.build --patterns mcar --rates 0.2 0.5"
                exit 0
                ;;
            *)
                echo "unknown option: $arg (try --help)" >&2
                exit 2
                ;;
        esac
    done
}
