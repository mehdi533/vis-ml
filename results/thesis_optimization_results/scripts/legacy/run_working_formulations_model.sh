#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

MODEL_VARIANT="${MODEL_VARIANT:-}"
case "${MODEL_VARIANT}" in
  mlp)
    MODEL_TAG="mlp"
    MODEL_LABEL="MLP"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-results/thesis_optimization_results/configs/base_optimization_mlp.yaml}"
    ;;
  mtlsh)
    MODEL_TAG="mtlsh"
    MODEL_LABEL="MTLSH"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-results/thesis_optimization_results/configs/base_optimization_mtlsh.yaml}"
    ;;
  mtlsh_all_sched)
    MODEL_TAG="mtlsh_all_sched"
    MODEL_LABEL="MTLSH all sched"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-results/thesis_optimization_results/configs/base_optimization_mtlsh_all_sched.yaml}"
    ;;
  picnn)
    MODEL_TAG="picnn"
    MODEL_LABEL="PICNN"
    BASE_CONFIG_SOURCE="${BASE_CONFIG_SOURCE:-results/thesis_optimization_results/configs/base_optimization_picnn.yaml}"
    ;;
  *)
    echo "MODEL_VARIANT must be one of: mlp, mtlsh, mtlsh_all_sched, picnn" >&2
    exit 1
    ;;
esac

RUN_LABEL="${RUN_LABEL:-${MODEL_TAG}}"
RESULTS_MODEL_ROOT="${RESULTS_MODEL_ROOT:-results/thesis_optimization_results/results/by_model/${RUN_LABEL}}"
OUTPUTS_MODEL_ROOT="${OUTPUTS_MODEL_ROOT:-results/thesis_optimization_results/outputs/by_model/${RUN_LABEL}}"
GENERATED_CONFIG_DIR="${GENERATED_CONFIG_DIR:-results/thesis_optimization_results/generated_configs/${RUN_LABEL}}"

FORMULATION_SUITE_SOURCE="${FORMULATION_SUITE_SOURCE:-results/thesis_optimization_results/configs/suites/formulation_comparison.yaml}"
SECURITY_SUITE_SOURCE="${SECURITY_SUITE_SOURCE:-results/thesis_optimization_results/configs/suites/security_checks.yaml}"
REDISPATCH_SUITE_SOURCE="${REDISPATCH_SUITE_SOURCE:-results/thesis_optimization_results/configs/suites/redispatch_sensitivity.yaml}"
ZONE_MISMATCH_SUITE_SOURCE="${ZONE_MISMATCH_SUITE_SOURCE:-results/thesis_optimization_results/configs/suites/zone_mismatch_vis_sensitivity.yaml}"
REPLAY_CONFIG_SOURCE="${REPLAY_CONFIG_SOURCE:-results/thesis_optimization_results/configs/replay/replay_validation.yaml}"
ANALYSIS_CONFIG_SOURCE="${ANALYSIS_CONFIG_SOURCE:-results/thesis_optimization_results/configs/analysis/results_pack.yaml}"

mkdir -p "${RESULTS_MODEL_ROOT}" "${OUTPUTS_MODEL_ROOT}" "${GENERATED_CONFIG_DIR}"

export REPO_ROOT
export GENERATED_CONFIG_DIR
export BASE_CONFIG_SOURCE
export RESULTS_MODEL_ROOT
export FORMULATION_SUITE_SOURCE
export SECURITY_SUITE_SOURCE
export REDISPATCH_SUITE_SOURCE
export ZONE_MISMATCH_SUITE_SOURCE
export REPLAY_CONFIG_SOURCE

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml

repo_root = Path(os.environ["REPO_ROOT"]).resolve()
cfg_path = Path(os.environ["BASE_CONFIG_SOURCE"])
cfg_path = cfg_path if cfg_path.is_absolute() else (repo_root / cfg_path).resolve()

with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

model_type = str((cfg.get("model") or {}).get("type", "")).strip()
if model_type in {"PICNN", "PICNN_MTLSH"}:
    raise SystemExit(
        "PICNN thesis runner is configured, but the current scheduler still rejects "
        f"{model_type} embeddings in scheduling/constraints_nn.py. "
        "Implement the PICNN embedding path first, then rerun this script."
    )
PY

"${PYTHON_BIN}" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml


def resolve(path_like: str, repo_root: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


repo_root = Path(os.environ["REPO_ROOT"]).resolve()
generated_dir = resolve(os.environ["GENERATED_CONFIG_DIR"], repo_root)
base_config = resolve(os.environ["BASE_CONFIG_SOURCE"], repo_root)
results_root = resolve(os.environ["RESULTS_MODEL_ROOT"], repo_root)

suite_specs = [
    ("formulation_comparison", "formulations", os.environ["FORMULATION_SUITE_SOURCE"]),
    ("security_checks", "security_checks", os.environ["SECURITY_SUITE_SOURCE"]),
    ("redispatch_sensitivity", "redispatch_sensitivity", os.environ["REDISPATCH_SUITE_SOURCE"]),
    ("zone_mismatch_vis_sensitivity", "zone_mismatch_vis_sensitivity_formulations", os.environ["ZONE_MISMATCH_SUITE_SOURCE"]),
]

for stem, results_subdir, source_raw in suite_specs:
    source = resolve(source_raw, repo_root)
    cfg = load_yaml(source)
    cfg["results_root"] = str((results_root / results_subdir).resolve())
    output_cfg = dict(cfg.get("output") or {})
    output_cfg["summary_csv"] = str((results_root / f"{stem}_summary.csv").resolve())
    output_cfg["summary_markdown"] = str((results_root / f"{stem}_summary.md").resolve())
    output_cfg["summary_json"] = str((results_root / f"{stem}_summary.json").resolve())
    cfg["output"] = output_cfg

    for run in list(cfg.get("runs") or []):
        run["base_config"] = str(base_config)

    write_yaml(generated_dir / f"{stem}.yaml", cfg)

replay_source = resolve(os.environ["REPLAY_CONFIG_SOURCE"], repo_root)
replay_cfg = load_yaml(replay_source)
for run in list(replay_cfg.get("runs") or []):
    if "suite_summary_json" in run:
        run["suite_summary_json"] = str((results_root / "formulation_comparison_summary.json").resolve())
output_cfg = dict(replay_cfg.get("output") or {})
output_cfg["summary_csv"] = str((results_root / "replay_validation_summary.csv").resolve())
output_cfg["detail_csv"] = str((results_root / "replay_validation_detail.csv").resolve())
output_cfg["summary_json"] = str((results_root / "replay_validation_summary.json").resolve())
replay_cfg["output"] = output_cfg
write_yaml(generated_dir / "replay_validation.yaml", replay_cfg)
PY

FORMULATION_SUITE_CONFIG="${GENERATED_CONFIG_DIR}/formulation_comparison.yaml"
SECURITY_SUITE_CONFIG="${GENERATED_CONFIG_DIR}/security_checks.yaml"
REDISPATCH_SUITE_CONFIG="${GENERATED_CONFIG_DIR}/redispatch_sensitivity.yaml"
ZONE_MISMATCH_SUITE_CONFIG="${GENERATED_CONFIG_DIR}/zone_mismatch_vis_sensitivity.yaml"
ZONE_MISMATCH_SUMMARY_JSON="${RESULTS_MODEL_ROOT}/zone_mismatch_vis_sensitivity_summary.json"
REPLAY_CONFIG="${GENERATED_CONFIG_DIR}/replay_validation.yaml"

echo "STARTING ${MODEL_LABEL} THESIS WORKING FORMULATIONS RUN AT $(date)"
echo "Base optimization config: ${BASE_CONFIG_SOURCE}"
echo "Generated suite configs: ${GENERATED_CONFIG_DIR}"
echo "Optimization results root: ${RESULTS_MODEL_ROOT}"
echo "Postprocessed outputs root: ${OUTPUTS_MODEL_ROOT}"

THESIS_OPT_RESULTS_DIR="${RESULTS_MODEL_ROOT}" \
THESIS_OPT_OUTPUTS_DIR="${OUTPUTS_MODEL_ROOT}" \
THESIS_OPT_BASE_CONFIG="${BASE_CONFIG_SOURCE}" \
THESIS_OPT_FORMULATION_SUITE_CONFIG="${FORMULATION_SUITE_CONFIG}" \
THESIS_OPT_REPLAY_CONFIG="${REPLAY_CONFIG}" \
THESIS_OPT_ANALYSIS_CONFIG="${ANALYSIS_CONFIG_SOURCE}" \
FORMULATION_SUITE_CONFIG="${FORMULATION_SUITE_CONFIG}" \
SECURITY_SUITE_CONFIG="${SECURITY_SUITE_CONFIG}" \
REDISPATCH_SUITE_CONFIG="${REDISPATCH_SUITE_CONFIG}" \
ZONE_MISMATCH_SUITE_CONFIG="${ZONE_MISMATCH_SUITE_CONFIG}" \
ZONE_MISMATCH_SUMMARY_JSON="${ZONE_MISMATCH_SUMMARY_JSON}" \
REPLAY_CONFIG="${REPLAY_CONFIG}" \
ANALYSIS_CONFIG="${ANALYSIS_CONFIG_SOURCE}" \
RESULTS_ROOT_LABEL="${RESULTS_MODEL_ROOT}" \
OUTPUTS_ROOT_LABEL="${OUTPUTS_MODEL_ROOT}" \
  bash "${SCRIPT_DIR}/run_working_formulations.sh" "$@"
