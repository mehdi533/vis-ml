#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/scheduling/thesis_optimization_benchmark.yaml}"
FORMULATION_SUITE="${FORMULATION_SUITE:-configs/scheduling/suites/01_formulation_comparison.yaml}"
SCENARIO_FAMILY="${SCENARIO_FAMILY:-global}"
TOP_K_VALUES="${TOP_K_VALUES:-1,3,5,all}"
GENERATED_SUITE="${GENERATED_SUITE:-results/thesis_optimization_results/configs/generated/mtlsh_topk_screening_${SCENARIO_FAMILY}.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
SCENARIO_IDS="${SCENARIO_IDS:-}"
MAX_SCENARIOS="${MAX_SCENARIOS:-0}"
RETAINED_FORMULATION_ID="${RETAINED_FORMULATION_ID:-}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"

gen_args=(
  --benchmark-config "${BENCHMARK_CONFIG}"
  --formulation-suite "${FORMULATION_SUITE}"
  --scenario-family "${SCENARIO_FAMILY}"
  --top-k-values "${TOP_K_VALUES}"
  --out-suite "${GENERATED_SUITE}"
)

if [[ -n "${RESULTS_ROOT}" ]]; then
  gen_args+=(--results-root "${RESULTS_ROOT}")
fi
if [[ -n "${SCENARIO_IDS}" ]]; then
  gen_args+=(--scenario-ids "${SCENARIO_IDS}")
fi
if [[ "${MAX_SCENARIOS}" != "0" ]]; then
  gen_args+=(--max-scenarios "${MAX_SCENARIOS}")
fi
if [[ -n "${RETAINED_FORMULATION_ID}" ]]; then
  gen_args+=(--retained-formulation-id "${RETAINED_FORMULATION_ID}")
fi

"${PYTHON_BIN}" results/thesis_optimization_results/scripts/generate_mtlsh_topk_screening_suite.py "${gen_args[@]}"

if [[ "${GENERATE_ONLY}" == "1" ]]; then
  exit 0
fi

run_args=(
  --suite "${GENERATED_SUITE}"
  --log-tail-lines "${LOG_TAIL_LINES}"
)
if [[ "${STOP_ON_ERROR}" == "1" ]]; then
  run_args+=(--stop-on-error)
fi

"${PYTHON_BIN}" scheduling/run_experiment_suite.py "${run_args[@]}"
