#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_VARIANT=mtlsh \
BASE_CONFIG_SOURCE=configs/scheduling/base_optimization_mtlsh_no_dispatch.yaml \
RUN_LABEL=mtlsh_no_dispatch \
  bash "${SCRIPT_DIR}/run_working_formulations_model.sh" "$@"
