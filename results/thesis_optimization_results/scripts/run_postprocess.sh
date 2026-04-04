#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-results/thesis_optimization_results/configs/analysis/results_pack.yaml}"
REQUIRE_REPLAY="${REQUIRE_REPLAY:-0}"

args=(--config "${ANALYSIS_CONFIG}")
if [[ "${REQUIRE_REPLAY}" == "1" ]]; then
  args+=(--require-replay)
fi

"${PYTHON_BIN}" scheduling/build_outputs.py "${args[@]}" "$@"
