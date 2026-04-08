#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

FORMULATION_SUITE_CONFIG="${FORMULATION_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/formulation_comparison.yaml}"
SECURITY_SUITE_CONFIG="${SECURITY_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/security_checks.yaml}"
REDISPATCH_SUITE_CONFIG="${REDISPATCH_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/redispatch_sensitivity.yaml}"
ZONE_MISMATCH_SUITE_CONFIG="${ZONE_MISMATCH_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/zone_mismatch_vis_sensitivity.yaml}"
ZONE_MISMATCH_SUMMARY_JSON="${ZONE_MISMATCH_SUMMARY_JSON:-results/thesis_optimization_results/results/zone_mismatch_vis_sensitivity_summary.json}"
ZONE_MISMATCH_GLOBAL_SCENARIO_ID="${ZONE_MISMATCH_GLOBAL_SCENARIO_ID:-global_uniform}"
ZONE_MISMATCH_STEM="${ZONE_MISMATCH_STEM:-zone_mismatch_vis_sensitivity}"
REPLAY_CONFIG="${REPLAY_CONFIG:-results/thesis_optimization_results/configs/replay/replay_validation.yaml}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-results/thesis_optimization_results/configs/analysis/results_pack.yaml}"
RUN_REPLAY="${RUN_REPLAY:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
REQUIRE_REPLAY="${REQUIRE_REPLAY:-${RUN_REPLAY}}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"
RESULTS_ROOT_LABEL="${RESULTS_ROOT_LABEL:-results/thesis_optimization_results/results}"
OUTPUTS_ROOT_LABEL="${OUTPUTS_ROOT_LABEL:-results/thesis_optimization_results/outputs}"

run_step() {
  local label="$1"
  shift
  echo "[working_formulations] START ${label} AT $(date)"
  "$@"
  echo "[working_formulations] DONE ${label} AT $(date)"
}

echo "STARTING THESIS WORKING FORMULATIONS RUN AT $(date)"
echo "Included suites: formulation_comparison, security_checks, redispatch_sensitivity, zone_mismatch_vis_sensitivity"
echo "Skipped suites: she_vis_rted_style_comparison, area_vis_comparison"

run_step "formulation_comparison" \
  env SUITE_CONFIG="${FORMULATION_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_formulations.sh"

run_step "security_checks" \
  env SUITE_CONFIG="${SECURITY_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_security_checks.sh"

run_step "redispatch_sensitivity" \
  env SUITE_CONFIG="${REDISPATCH_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_redispatch_sensitivity.sh"

run_step "zone_mismatch_vis_sensitivity" \
  env SUITE_CONFIG="${ZONE_MISMATCH_SUITE_CONFIG}" SUMMARY_JSON="${ZONE_MISMATCH_SUMMARY_JSON}" \
  GLOBAL_SCENARIO_ID="${ZONE_MISMATCH_GLOBAL_SCENARIO_ID}" STEM="${ZONE_MISMATCH_STEM}" \
  STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_zone_mismatch_vis_sensitivity.sh"

if [[ "${RUN_REPLAY}" == "1" ]]; then
  run_step "replay_validation" \
    env REPLAY_CONFIG="${REPLAY_CONFIG}" \
    bash "${SCRIPT_DIR}/run_replay_validation.sh"
else
  echo "[working_formulations] SKIP replay_validation (RUN_REPLAY=${RUN_REPLAY})"
fi

if [[ "${RUN_POSTPROCESS}" == "1" ]]; then
  run_step "postprocess" \
    env ANALYSIS_CONFIG="${ANALYSIS_CONFIG}" REQUIRE_REPLAY="${REQUIRE_REPLAY}" \
    bash "${SCRIPT_DIR}/run_postprocess.sh"
else
  echo "[working_formulations] SKIP postprocess (RUN_POSTPROCESS=${RUN_POSTPROCESS})"
fi

echo "FINISHED THESIS WORKING FORMULATIONS RUN AT $(date)"
echo "Optimization results root: ${RESULTS_ROOT_LABEL}"
echo "Postprocessed thesis outputs: ${OUTPUTS_ROOT_LABEL}"
