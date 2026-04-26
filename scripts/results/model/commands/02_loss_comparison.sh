#!/bin/bash
#SBATCH --job-name=tm_losses
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/loss_comparison_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/loss_comparison_job_err%j.out
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

python models/train_sweep.py --config configs/model/loss_comparison_mlp.yaml
python models/train_sweep.py --config configs/model/loss_comparison_mtlsh.yaml
python models/train_sweep.py --config configs/model/loss_comparison_mtlgsh.yaml
python models/train_sweep.py --config configs/model/loss_comparison_picnn.yaml

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/loss_comparison_mlp \
  --source run \
  --group-by loss \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv results/thesis_model_results/tables/loss_comparison_mlp.csv

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/loss_comparison_mlp \
  --source label \
  --group-by loss label \
  --value-cols rmse mae \
  --output-csv results/thesis_model_results/tables/loss_comparison_mlp_by_label.csv

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/loss_comparison_mtlsh \
  --source run \
  --group-by loss \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv results/thesis_model_results/tables/loss_comparison_mtlsh.csv

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/loss_comparison_mtlgsh \
  --source run \
  --group-by loss \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv results/thesis_model_results/tables/loss_comparison_mtlgsh.csv

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/loss_comparison_picnn \
  --source run \
  --group-by loss \
  --value-cols agg_rmse_mean agg_mae_mean best_val_loss \
  --output-csv results/thesis_model_results/tables/loss_comparison_picnn.csv

echo "FINISHED AT $(date)"
