#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

CONFIGS=(
  "configs/scheduling/replay/focused_trace_topk_global_b0p900_s1p030.yaml"
  "configs/scheduling/replay/focused_trace_topk_global_b0p900_s1p070.yaml"
  "configs/scheduling/replay/focused_trace_topk_global_b0p900_s1p200.yaml"
)

for cfg in "${CONFIGS[@]}"; do
  echo "[topk-replay-suite] Running ${cfg}"
  TRACE_CONFIG="${cfg}" "${PYTHON_BIN}" results/thesis_optimization_results/scripts/export_replay_trace_panel.py --config "${cfg}"
done
