#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/results/optimization/common_env.sh"

# Prefer currently active virtualenv python over common_env defaults.
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
  export PYTHON_BIN
fi

DEFAULT_CONFIG="configs/presentation/presentation_vis_case.yaml"
if [[ "$#" -eq 0 ]]; then
  set -- --config "${DEFAULT_CONFIG}"
fi

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/results/optimization/run_std_ieee39_response.py" "$@"
