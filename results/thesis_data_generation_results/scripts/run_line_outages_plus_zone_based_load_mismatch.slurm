#!/bin/bash
#SBATCH --job-name=dg_line_zone
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_data_generation_results/logs/line_outages_plus_zone_based_load_mismatch_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_data_generation_results/logs/line_outages_plus_zone_based_load_mismatch_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=3:00:00

set -euo pipefail

echo "STARTING AT $(date)"

python data_generation/run_sims.py \
  --config results/thesis_data_generation_results/configs/line_outages_plus_zone_based_load_mismatch.yaml

echo "FINISHED AT $(date)"
