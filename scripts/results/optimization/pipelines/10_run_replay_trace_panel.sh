#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

TRACE_CONFIG="${TRACE_CONFIG:-configs/scheduling/replay/focused_trace_stressed_case.yaml}"
"${PYTHON_BIN}" results/thesis_optimization_results/scripts/export_replay_trace_panel.py \
  --config "${TRACE_CONFIG}" \
  "$@"
