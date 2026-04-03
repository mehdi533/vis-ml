#!/bin/bash
#SBATCH --job-name=tm_she_mlp
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/mlp_she_style_comparison_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/mlp_she_style_comparison_job_err%j.out
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

mkdir -p results/thesis_model_results/logs results/thesis_model_results/tables

echo "STARTING AT $(date)"

python models/train_sweep.py --config results/thesis_model_results/configs/mlp_she_style_comparison.yaml

python models/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/mlp_she_style_comparison \
  --source run \
  --group-by data_target_cols \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv results/thesis_model_results/tables/mlp_she_style_comparison.csv

python models/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/mlp_she_style_comparison \
  --source label \
  --group-by data_target_cols label \
  --value-cols rmse mae \
  --output-csv results/thesis_model_results/tables/mlp_she_style_comparison_by_label.csv

echo "FINISHED AT $(date)"
