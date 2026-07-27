#!/usr/bin/env bash
#SBATCH --job-name=vis_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/logs/train_%j.out
#SBATCH --error=results/logs/train_%j.err
## #SBATCH --account=<your_account>
## #SBATCH --partition=<partition>          # add a GPU partition + --gpus=1 to accelerate
#
# Trains the model sweep and exports the optimization-ready MTLSH surrogate.
# Assumes the full dataset has been generated (merge the per-scenario CSVs first,
# or point train_sweep configs at the desired simulation_results.csv).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PWD"

echo "START $(date)"
# Architecture / loss / scaler sweep (thesis Chapter 5 model side):
python -m models.train_sweep --config configs/model/train_sweep.yaml
# Retained optimization-ready surrogate (178-feature M/D contract):
python -m models.train_sweep --config configs/model/optimization_ready_mtlsh.yaml
echo "DONE  $(date)"
