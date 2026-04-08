#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=(mtlsh picnn)
fi

"${PYTHON_BIN}" results/thesis_optimization_results/src/build_model_comparison_outputs.py \
  --models "${MODELS[@]}"
