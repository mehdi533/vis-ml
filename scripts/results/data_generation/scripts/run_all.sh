#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-../venv/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"

"${PYTHON_BIN}" data_generation/run_sims.py \
  --config configs/data_generation/load_mismatch_only.yaml

"${PYTHON_BIN}" data_generation/run_sims.py \
  --config configs/data_generation/line_outages_only.yaml

"${PYTHON_BIN}" data_generation/run_sims.py \
  --config configs/data_generation/line_outages_plus_global_load_mismatch.yaml

"${PYTHON_BIN}" data_generation/run_sims.py \
  --config configs/data_generation/zone_based_load_mismatch.yaml

"${PYTHON_BIN}" data_generation/run_sims.py \
  --config configs/data_generation/line_outages_plus_zone_based_load_mismatch.yaml

"${PYTHON_BIN}" data_generation/run_sims.py \
  --config configs/data_generation/no_mismatch.yaml
