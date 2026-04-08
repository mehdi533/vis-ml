#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_REPLAY="${RUN_REPLAY:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
REQUIRE_REPLAY="${REQUIRE_REPLAY:-${RUN_REPLAY}}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
LOG_TAIL_LINES="${LOG_TAIL_LINES:-80}"

echo "STARTING THESIS WORKING FORMULATIONS LOCAL RUN AT $(date)"
echo "This uses the full production suites and writes into results/thesis_optimization_results/results."

RUN_REPLAY="${RUN_REPLAY}" \
RUN_POSTPROCESS="${RUN_POSTPROCESS}" \
REQUIRE_REPLAY="${REQUIRE_REPLAY}" \
STOP_ON_ERROR="${STOP_ON_ERROR}" \
LOG_TAIL_LINES="${LOG_TAIL_LINES}" \
  bash "${SCRIPT_DIR}/run_working_formulations.sh" "$@"
