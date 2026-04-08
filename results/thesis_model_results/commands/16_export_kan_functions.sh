#!/bin/bash
#SBATCH --job-name=tm_kan_funcs
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/kan_functions_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/kan_functions_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p results/thesis_model_results/logs results/thesis_model_results/tables LaTeX/figures

echo "STARTING AT $(date)"

python results/thesis_model_results/src/export_kan_functions.py \
  --model-dir results/thesis_model_results/outputs/architecture_comparison_exploratory/MTLGSH_KAN_SHARED__mse__minmax__seed42

echo "FINISHED AT $(date)"
