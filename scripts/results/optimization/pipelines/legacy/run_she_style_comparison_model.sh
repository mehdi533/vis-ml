#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

MODEL_VARIANT="${MODEL_VARIANT:-}"
case "${MODEL_VARIANT}" in
  mlp)
    MODEL_LABEL="MLP"
    RUN_LABEL="${RUN_LABEL:-mlp}"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-configs/scheduling/base_optimization_mlp.yaml}"
    ;;
  mtlsh)
    MODEL_LABEL="MTLSH"
    RUN_LABEL="${RUN_LABEL:-mtlsh}"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-configs/scheduling/base_optimization_mtlsh.yaml}"
    ;;
  mtlsh_no_dispatch)
    MODEL_LABEL="MTLSH no dispatch"
    RUN_LABEL="${RUN_LABEL:-mtlsh_no_dispatch}"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-configs/scheduling/base_optimization_mtlsh_no_dispatch.yaml}"
    ;;
  *)
    echo "MODEL_VARIANT must be one of: mlp, mtlsh, mtlsh_no_dispatch" >&2
    exit 1
    ;;
esac

SUITE_SOURCE="${SUITE_SOURCE:-configs/scheduling/suites/02_she_vis_rted_style_comparison.yaml}"
RESULTS_MODEL_ROOT="${RESULTS_MODEL_ROOT:-results/thesis_optimization_results/results/by_model/${RUN_LABEL}}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-results/thesis_optimization_results/generated_configs/${RUN_LABEL}_she_style}"

mkdir -p "${RESULTS_MODEL_ROOT}" "${GENERATED_CONFIG_DIR}"

export REPO_ROOT
export BASE_CONFIG_SOURCE
export SUITE_SOURCE
export RESULTS_MODEL_ROOT
export GENERATED_CONFIG_DIR

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml


def resolve(path_like: str, repo_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (repo_root / path).resolve()


repo_root = Path(os.environ["REPO_ROOT"]).resolve()
suite_source = resolve(os.environ["SUITE_SOURCE"], repo_root)
base_config = resolve(os.environ["BASE_CONFIG_SOURCE"], repo_root)
results_root = resolve(os.environ["RESULTS_MODEL_ROOT"], repo_root)
generated_dir = resolve(os.environ["GENERATED_CONFIG_DIR"], repo_root)

with suite_source.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg["results_root"] = str((results_root / "she_style_formulations").resolve())
output_cfg = dict(cfg.get("output") or {})
output_cfg["summary_csv"] = str((results_root / "she_style_comparison_summary.csv").resolve())
output_cfg["summary_markdown"] = str((results_root / "she_style_comparison_summary.md").resolve())
output_cfg["summary_json"] = str((results_root / "she_style_comparison_summary.json").resolve())
cfg["output"] = output_cfg

for run in list(cfg.get("runs") or []):
    run["base_config"] = str(base_config)

generated_dir.mkdir(parents=True, exist_ok=True)
out_path = generated_dir / "she_style_comparison.yaml"
with out_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(out_path)
PY

GENERATED_SUITE="${GENERATED_CONFIG_DIR}/she_style_comparison.yaml"

echo "STARTING ${MODEL_LABEL} SHE-STYLE COMPARISON AT $(date)"
echo "Base optimization config: ${BASE_CONFIG_SOURCE}"
echo "Generated suite config: ${GENERATED_SUITE}"
echo "Results root: ${RESULTS_MODEL_ROOT}"

env SUITE_CONFIG="${GENERATED_SUITE}" \
  bash "${SCRIPT_DIR}/run_she_style_comparison.sh" "$@"
