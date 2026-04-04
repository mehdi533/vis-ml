#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

SUITE_CONFIG="${SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/zone_mismatch_vis_sensitivity.yaml}"
SUMMARY_JSON="${SUMMARY_JSON:-results/thesis_optimization_results/results/zone_mismatch_vis_sensitivity_summary.json}"
GLOBAL_SCENARIO_ID="${GLOBAL_SCENARIO_ID:-global_uniform}"
STEM="${STEM:-zone_mismatch_vis_sensitivity}"
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

"${PYTHON_BIN}" results/thesis_optimization_results/src/zone_mismatch_vis_analysis.py \
  --summary-json "${SUMMARY_JSON}" \
  --global-scenario-id "${GLOBAL_SCENARIO_ID}" \
  --stem "${STEM}"
