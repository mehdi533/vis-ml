#!/bin/bash
#SBATCH --job-name=dg_load_only
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_data_generation_results/logs/load_mismatch_only_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_data_generation_results/logs/load_mismatch_only_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=3:00:00

set -euo pipefail

echo "STARTING AT $(date)"

python data_generation/run_sims.py \
  --config configs/data_generation/load_mismatch_only.yaml

echo "FINISHED AT $(date)"
