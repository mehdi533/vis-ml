#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

SUITE_CONFIG="${SUITE_CONFIG:-configs/scheduling/suites/legacy/milp_constraint_isolation_local.yaml}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-160}"
mkdir -p results/thesis_optimization_results/local_validation/milp_isolation_local

echo "STARTING MILP CONSTRAINT ISOLATION RUN AT $(date)"
echo "SUITE_CONFIG=${SUITE_CONFIG}"

if [[ "${STOP_ON_ERROR}" == "1" ]]; then
  "${PYTHON_BIN}" scheduling/run_experiment_suite.py \
    --suite "${SUITE_CONFIG}" \
    --log-tail-lines "${LOG_TAIL_LINES}" \
    --stop-on-error \
    "$@"
else
  "${PYTHON_BIN}" scheduling/run_experiment_suite.py \
    --suite "${SUITE_CONFIG}" \
    --log-tail-lines "${LOG_TAIL_LINES}" \
    "$@"
fi

echo "FINISHED MILP CONSTRAINT ISOLATION RUN AT $(date)"
echo "Default output root: results/thesis_optimization_results/local_validation/milp_isolation_local"
