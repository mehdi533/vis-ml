#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

SUITE_CONFIG="${SUITE_CONFIG:-configs/scheduling/suites/01_formulation_comparison.yaml}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"

args=(
  --suite "${SUITE_CONFIG}"
  --log-tail-lines "${LOG_TAIL_LINES}"
)
if [[ "${STOP_ON_ERROR}" == "1" ]]; then
  args+=(--stop-on-error)
fi

"${PYTHON_BIN}" scheduling/run_experiment_suite.py "${args[@]}" "$@"
