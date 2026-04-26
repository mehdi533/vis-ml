#!/bin/bash
#SBATCH --job-name=tm_boundary
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/boundary_region_eval_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/boundary_region_eval_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=18:00:00

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p results/thesis_model_results/logs results/thesis_model_results/tables

echo "STARTING AT $(date)"

python models/train_sweep.py --config configs/model/shortlist_detailed_eval.yaml

python results/thesis_model_results/src/boundary_region_eval.py \
  --config configs/model/boundary_region_eval.yaml \
  --subset-csv results/thesis_model_results/tables/boundary_region_eval_subset_metrics.csv \
  --by-label-csv results/thesis_model_results/tables/boundary_region_eval_by_label.csv \
  --comparison-csv results/thesis_model_results/tables/boundary_region_eval_global_vs_stress.csv

echo "FINISHED AT $(date)"
