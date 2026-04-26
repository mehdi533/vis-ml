#!/bin/bash
#SBATCH --job-name=tm_relu_family
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/relu_size_family_comparison_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/relu_size_family_comparison_job_err%j.out
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

python models/train_sweep.py --config configs/model/relu_size_mlp_picnn.yaml
python models/train_sweep.py --config configs/model/relu_size_mtlgsh.yaml

python results/thesis_model_results/src/relu_family_summary.py \
  --mtlsh-tradeoff-csv results/thesis_model_results/tables/mtlsh_embeddability_tradeoff.csv \
  --mlp-picnn-dir results/thesis_model_results/outputs/relu_size_mlp_picnn \
  --mtlgsh-dir results/thesis_model_results/outputs/relu_size_mtlgsh \
  --she-dir results/thesis_model_results/outputs/mlp_she_style_comparison \
  --output-csv results/thesis_model_results/tables/relu_family_comparison.csv

echo "FINISHED AT $(date)"
