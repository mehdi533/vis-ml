#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

"${PYTHON_BIN}" final_optimization_folder/replay_validation.py \
  --config results/thesis_optimization_results/configs/replay/replay_validation.yaml \
  "$@"
