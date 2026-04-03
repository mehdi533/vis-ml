#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
SUITE_CONFIG="${SUITE_CONFIG:-results/thesis_optimization_results/configs/suites/formulation_comparison_debug.yaml}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

mkdir -p results/thesis_optimization_results/results/debug

echo "STARTING OPTIMIZATION DEBUG RUN AT $(date)"
echo "SUITE_CONFIG=${SUITE_CONFIG}"

"${PYTHON_BIN}" scheduling/run_experiment_suite.py \
  --suite "${SUITE_CONFIG}" \
  --stop-on-error \
  "$@"

echo "FINISHED OPTIMIZATION DEBUG RUN AT $(date)"
echo "Summary CSV: results/thesis_optimization_results/results/debug/formulation_comparison_debug_summary.csv"
