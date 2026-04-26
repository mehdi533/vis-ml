#!/usr/bin/env bash

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="${SCRIPT_DIR}"
  while [[ ! -d "${REPO_ROOT}/.git" && "${REPO_ROOT}" != "/" ]]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
  done
fi

cd "${REPO_ROOT}"

DEFAULT_PYTHON_BIN="${REPO_ROOT}/../venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"
