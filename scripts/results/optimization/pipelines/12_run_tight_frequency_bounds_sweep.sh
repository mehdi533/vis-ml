#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

SUITE_CONFIG="${SUITE_CONFIG:-configs/scheduling/suites/07_tight_frequency_bounds_sweep.yaml}"
SUMMARY_JSON="${SUMMARY_JSON:-results/thesis_optimization_results/results/tight_frequency_bounds_sweep_summary.json}"
REPLAY_CONFIG="${REPLAY_CONFIG:-configs/scheduling/replay/tight_frequency_bounds_sweep.yaml}"
REPLAY_OUTDIR="${REPLAY_OUTDIR:-results/thesis_optimization_results/local_validation/replay_trace_tight_frequency_bounds_sweep}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-120}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

run_args=(
  --suite "${SUITE_CONFIG}"
  --log-tail-lines "${LOG_TAIL_LINES}"
)
if [[ "${STOP_ON_ERROR}" == "1" ]]; then
  run_args+=(--stop-on-error)
fi

"${PYTHON_BIN}" scheduling/run_experiment_suite.py "${run_args[@]}"

"${PYTHON_BIN}" results/thesis_optimization_results/scripts/build_tight_frequency_replay_config.py \
  --summary-json "${SUMMARY_JSON}" \
  --out-config "${REPLAY_CONFIG}" \
  --out-directory "${REPLAY_OUTDIR}" \
  --strictest-limit-lines

"${PYTHON_BIN}" results/thesis_optimization_results/scripts/export_replay_trace_panel.py \
  --config "${REPLAY_CONFIG}"
