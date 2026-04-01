#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

python3 models/train_sweep.py --config thesis_model_results/configs/feature_relevance_attention.yaml
python3 models/train_sweep.py --config thesis_model_results/configs/shortlist_detailed_eval.yaml

python3 models/feature_relevance.py \
  --config thesis_model_results/configs/feature_relevance_attention.yaml \
  --model-dir thesis_model_results/outputs/feature_relevance_attention/MTLGSH_ATT__mse__minmax__seed42 \
  --mode both \
  --group-config thesis_model_results/configs/feature_groups.yaml

python3 models/feature_relevance.py \
  --config thesis_model_results/configs/shortlist_detailed_eval.yaml \
  --model-dir thesis_model_results/outputs/shortlist_detailed_eval/MTLSH__mse__minmax__seed42 \
  --mode permutation \
  --group-config thesis_model_results/configs/feature_groups.yaml \
  --out-dir thesis_model_results/outputs/feature_relevance_permutation_mtlsh
