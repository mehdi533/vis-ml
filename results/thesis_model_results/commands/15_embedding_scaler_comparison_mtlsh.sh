#!/bin/bash
#SBATCH --job-name=tm_embed_scaler
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/embedding_scaler_comparison_mtlsh_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/embedding_scaler_comparison_mtlsh_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p results/thesis_model_results/logs results/thesis_model_results/tables

echo "STARTING AT $(date)"

python models/train_sweep.py --config results/thesis_model_results/configs/embedding_scaler_comparison_mtlsh.yaml

python results/thesis_model_results/src/scaler_bigm_summary.py \
  --sweep-dir results/thesis_model_results/outputs/embedding_scaler_comparison_mtlsh \
  --optimization-config results/thesis_optimization_results/configs/base_optimization.yaml \
  --output-csv results/thesis_model_results/tables/embedding_scaler_bigm_comparison_mtlsh.csv

echo "FINISHED AT $(date)"
