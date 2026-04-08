#!/bin/bash
#SBATCH --job-name=tm_opt_ready_models
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/optimization_ready_models_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/optimization_ready_models_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/12_optimization_ready_models.sh" "$@"
