#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p thesis_model_results/exports

python3 models/train_sweep.py --config thesis_model_results/configs/architecture_comparison_core.yaml

python3 models/export_retained_model.py \
  --summary-csv thesis_model_results/outputs/architecture_comparison_core/sweep_run_summary.csv \
  --dest-dir thesis_model_results/exports/retained_core_model \
  --metric agg_rmse_mean \
  --note "Automatically selected best embeddable surrogate from architecture_comparison_core."
