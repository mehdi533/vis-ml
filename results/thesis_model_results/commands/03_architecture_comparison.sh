#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

mkdir -p thesis_model_results/tables

python3 models/train_sweep.py --config thesis_model_results/configs/architecture_comparison_core.yaml
python3 models/train_sweep.py --config thesis_model_results/configs/architecture_comparison_exploratory.yaml

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/architecture_comparison_core \
  --source run \
  --group-by model \
  --value-cols agg_rmse_mean agg_mae_mean n_parameters_trainable \
  --output-csv thesis_model_results/tables/architecture_comparison_core.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/architecture_comparison_core \
  --source label \
  --group-by model label \
  --value-cols rmse mae \
  --output-csv thesis_model_results/tables/architecture_comparison_core_by_label.csv

python3 models/summarize_sweep.py \
  --input-dir thesis_model_results/outputs/architecture_comparison_exploratory \
  --source run \
  --group-by model \
  --value-cols agg_rmse_mean agg_mae_mean n_parameters_trainable \
  --output-csv thesis_model_results/tables/architecture_comparison_exploratory.csv
