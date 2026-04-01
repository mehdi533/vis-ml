#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"

"${PYTHON_BIN}" final_optimization_folder/run_experiment_suite.py \
  --suite thesis_optimization_results/configs/suites/formulation_comparison.yaml
