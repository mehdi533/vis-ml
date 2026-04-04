#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

SUITE_CONFIG="${SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/area_vis_comparison.yaml}"
SUMMARY_JSON="${SUMMARY_JSON:-results/thesis_optimization_results/results/area_vis_comparison_summary.json}"
BASELINE_ID="${BASELINE_ID:-she_method_i_rted}"
STEM="${STEM:-area_vis_comparison}"
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

"${PYTHON_BIN}" results/thesis_optimization_results/src/area_vis_analysis.py \
  --summary-json "${SUMMARY_JSON}" \
  --baseline-id "${BASELINE_ID}" \
  --stem "${STEM}"
