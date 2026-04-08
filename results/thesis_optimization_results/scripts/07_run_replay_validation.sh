#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

REPLAY_CONFIG="${REPLAY_CONFIG:-results/thesis_optimization_results/configs/replay/replay_validation.yaml}"
"${PYTHON_BIN}" scheduling/replay_validation.py \
  --config "${REPLAY_CONFIG}" \
  "$@"
