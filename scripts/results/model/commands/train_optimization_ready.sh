#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/../venv/bin/python}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"

cd "${REPO_ROOT}"

CONFIGS=(
  "optimization_ready_mtlsh_no_dispatch.yaml"
  "optimization_ready_mtlsh.yaml"
  "optimization_ready_mtlsh_with_dispatch.yaml"
  "optimization_ready_mtlsh_all_sched.yaml"
)

for config_name in "${CONFIGS[@]}"; do
  echo "=== Training ${config_name%.yaml} ==="
  "${PYTHON_BIN}" models/train_sweep.py --config "configs/model/${config_name}"
  echo ""
done

echo "=== Done ==="
