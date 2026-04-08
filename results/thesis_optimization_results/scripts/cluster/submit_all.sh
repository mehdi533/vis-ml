#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_ENV_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
unset SCRIPT_DIR
source "${COMMON_ENV_DIR}/common_env.sh"

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-results/thesis_optimization_results/configs/thesis_optimization_benchmark.yaml}"
FORMULATION_SUITE="${FORMULATION_SUITE:-results/thesis_optimization_results/configs/suites/01_formulation_comparison.yaml}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_benchmark_manifest.py" --benchmark-config "${BENCHMARK_CONFIG}" --formulation-suite "${FORMULATION_SUITE}" --dry-run

echo "[submit_all] submit main benchmark"
bash "${SCRIPT_DIR}/launch_main_benchmark.sh"

echo "[submit_all] submit cross-method subset"
bash "${SCRIPT_DIR}/launch_cross_method_subset.sh"

echo "[submit_all] submit replay for main benchmark after optimization outputs exist"
BENCHMARK_GROUP=main bash "${SCRIPT_DIR}/launch_replay_validation.sh"

echo "[submit_all] submit replay for cross-method subset after optimization outputs exist"
BENCHMARK_GROUP=cross_method_subset bash "${SCRIPT_DIR}/launch_replay_validation.sh"
