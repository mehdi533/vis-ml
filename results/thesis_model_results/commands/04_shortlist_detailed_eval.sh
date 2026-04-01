#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p thesis_model_results/tables

python3 models/train_sweep.py --config thesis_model_results/configs/shortlist_detailed_eval.yaml

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/shortlist_detailed_eval \
  --source run \
  --group-by model \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv thesis_model_results/tables/shortlist_detailed_eval_overview.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/shortlist_detailed_eval \
  --source label \
  --group-by model label \
  --value-cols rmse mae \
  --output-csv thesis_model_results/tables/shortlist_detailed_eval_by_label.csv
