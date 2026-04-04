#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

FORMULATION_SUITE_CONFIG="${FORMULATION_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/formulation_comparison.yaml}"
SHE_SUITE_CONFIG="${SHE_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/she_vis_rted_style_comparison.yaml}"
AREA_VIS_SUITE_CONFIG="${AREA_VIS_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/area_vis_comparison.yaml}"
AREA_VIS_SUMMARY_JSON="${AREA_VIS_SUMMARY_JSON:-results/thesis_optimization_results/results/area_vis_comparison_summary.json}"
AREA_VIS_BASELINE_ID="${AREA_VIS_BASELINE_ID:-she_method_i_rted}"
AREA_VIS_STEM="${AREA_VIS_STEM:-area_vis_comparison}"
ZONE_MISMATCH_SUITE_CONFIG="${ZONE_MISMATCH_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/zone_mismatch_vis_sensitivity.yaml}"
ZONE_MISMATCH_SUMMARY_JSON="${ZONE_MISMATCH_SUMMARY_JSON:-results/thesis_optimization_results/results/zone_mismatch_vis_sensitivity_summary.json}"
ZONE_MISMATCH_GLOBAL_SCENARIO_ID="${ZONE_MISMATCH_GLOBAL_SCENARIO_ID:-global_uniform}"
ZONE_MISMATCH_STEM="${ZONE_MISMATCH_STEM:-zone_mismatch_vis_sensitivity}"
SECURITY_SUITE_CONFIG="${SECURITY_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/security_checks.yaml}"
REDISPATCH_SUITE_CONFIG="${REDISPATCH_SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/redispatch_sensitivity.yaml}"
REPLAY_CONFIG="${REPLAY_CONFIG:-results/thesis_optimization_results/configs/replay/replay_validation.yaml}"
ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-results/thesis_optimization_results/configs/analysis/results_pack.yaml}"
REQUIRE_REPLAY="${REQUIRE_REPLAY:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"

SUITE_CONFIG="${FORMULATION_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_formulations.sh"

SUITE_CONFIG="${SHE_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_she_style_comparison.sh"

SUITE_CONFIG="${AREA_VIS_SUITE_CONFIG}" SUMMARY_JSON="${AREA_VIS_SUMMARY_JSON}" BASELINE_ID="${AREA_VIS_BASELINE_ID}" STEM="${AREA_VIS_STEM}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_area_vis_comparison.sh"

SUITE_CONFIG="${ZONE_MISMATCH_SUITE_CONFIG}" SUMMARY_JSON="${ZONE_MISMATCH_SUMMARY_JSON}" GLOBAL_SCENARIO_ID="${ZONE_MISMATCH_GLOBAL_SCENARIO_ID}" STEM="${ZONE_MISMATCH_STEM}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_zone_mismatch_vis_sensitivity.sh"

SUITE_CONFIG="${SECURITY_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_security_checks.sh"

SUITE_CONFIG="${REDISPATCH_SUITE_CONFIG}" STOP_ON_ERROR="${STOP_ON_ERROR}" LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_redispatch_sensitivity.sh"

REPLAY_CONFIG="${REPLAY_CONFIG}" bash "${SCRIPT_DIR}/run_replay_validation.sh"

ANALYSIS_CONFIG="${ANALYSIS_CONFIG}" REQUIRE_REPLAY="${REQUIRE_REPLAY}" bash "${SCRIPT_DIR}/run_postprocess.sh"
