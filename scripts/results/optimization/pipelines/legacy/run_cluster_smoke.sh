#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"

echo "STARTING THESIS OPTIMIZATION CLUSTER SMOKE TEST AT $(date)"

SUITE_CONFIG="configs/scheduling/smoke/formulation_comparison_smoke.yaml" \
STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_formulations.sh"

SUITE_CONFIG="configs/scheduling/smoke/security_checks_smoke.yaml" \
STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_security_checks.sh"

SUITE_CONFIG="configs/scheduling/smoke/redispatch_sensitivity_smoke.yaml" \
STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_redispatch_sensitivity.sh"

SUITE_CONFIG="configs/scheduling/smoke/she_vis_rted_style_comparison_smoke.yaml" \
STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_she_style_comparison.sh"

SUITE_CONFIG="configs/scheduling/smoke/area_vis_comparison_smoke.yaml" \
SUMMARY_JSON="results/thesis_optimization_results/local_validation/smoke/area_vis_comparison_smoke_summary.json" \
BASELINE_ID="she_method_i_rted" \
STEM="area_vis_comparison_smoke" \
STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_area_vis_comparison.sh"

SUITE_CONFIG="configs/scheduling/smoke/zone_mismatch_vis_sensitivity_smoke.yaml" \
SUMMARY_JSON="results/thesis_optimization_results/local_validation/smoke/zone_mismatch_vis_sensitivity_smoke_summary.json" \
GLOBAL_SCENARIO_ID="global_uniform" \
STEM="zone_mismatch_vis_sensitivity_smoke" \
STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_zone_mismatch_vis_sensitivity.sh"

REPLAY_CONFIG="configs/scheduling/smoke/replay_validation_smoke.yaml" \
  bash "${SCRIPT_DIR}/run_replay_validation.sh"

echo "FINISHED THESIS OPTIMIZATION CLUSTER SMOKE TEST AT $(date)"
echo "Smoke artifacts root: results/thesis_optimization_results/local_validation/smoke"
