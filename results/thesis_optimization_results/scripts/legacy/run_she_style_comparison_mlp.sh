#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_VARIANT=mlp \
  bash "${SCRIPT_DIR}/run_she_style_comparison_model.sh" "$@"
