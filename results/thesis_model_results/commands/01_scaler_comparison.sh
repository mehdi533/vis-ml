#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p thesis_model_results/tables

python3 models/train_sweep.py --config thesis_model_results/configs/scaler_comparison_mlp.yaml
python3 models/train_sweep.py --config thesis_model_results/configs/scaler_comparison_mtlsh.yaml
python3 models/train_sweep.py --config thesis_model_results/configs/scaler_comparison_mtlgsh.yaml
python3 models/train_sweep.py --config thesis_model_results/configs/scaler_comparison_picnn.yaml

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/scaler_comparison_mlp \
  --source run \
  --group-by scaler \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv thesis_model_results/tables/scaler_comparison_mlp.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/scaler_comparison_mlp \
  --source label \
  --group-by scaler label \
  --value-cols rmse mae \
  --output-csv thesis_model_results/tables/scaler_comparison_mlp_by_label.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/scaler_comparison_mtlsh \
  --source run \
  --group-by scaler \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv thesis_model_results/tables/scaler_comparison_mtlsh.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/scaler_comparison_mtlgsh \
  --source run \
  --group-by scaler \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv thesis_model_results/tables/scaler_comparison_mtlgsh.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/scaler_comparison_picnn \
  --source run \
  --group-by scaler \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv thesis_model_results/tables/scaler_comparison_picnn.csv
