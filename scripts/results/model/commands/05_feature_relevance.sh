#!/bin/bash
#SBATCH --job-name=tm_features
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/feature_relevance_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/feature_relevance_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p results/thesis_model_results/logs

echo "STARTING AT $(date)"

python models/train_sweep.py --config configs/model/feature_relevance_attention.yaml
python models/train_sweep.py --config configs/model/shortlist_detailed_eval.yaml

python models/feature_relevance.py \
  --config configs/model/feature_relevance_attention.yaml \
  --model-dir results/thesis_model_results/outputs/feature_relevance_attention/MTLGSH_ATT__mse__minmax__seed42 \
  --mode both \
  --group-config configs/model/feature_groups.yaml

python models/feature_relevance.py \
  --config configs/model/shortlist_detailed_eval.yaml \
  --model-dir results/thesis_model_results/outputs/shortlist_detailed_eval/MTLSH__mse__minmax__seed42 \
  --mode permutation \
  --group-config configs/model/feature_groups.yaml \
  --out-dir results/thesis_model_results/outputs/feature_relevance_permutation_mtlsh

echo "FINISHED AT $(date)"
