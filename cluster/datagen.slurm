#!/usr/bin/env bash
#SBATCH --job-name=vis_datagen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128          # Python multiprocessing needs the cores on ONE node
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/logs/datagen_%x_%j.out
#SBATCH --error=results/logs/datagen_%x_%j.err
## --- EDIT for your cluster ---
## #SBATCH --account=<your_account>
## #SBATCH --partition=<partition>
#
# Usage:  sbatch cluster/datagen.slurm configs/data_generation/load_mismatch_only.yaml
# NOTE: set `workers:` in the config <= --cpus-per-task above.
set -euo pipefail
CONFIG="${1:?usage: sbatch cluster/datagen.slurm <config.yaml>}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # 1 thread/worker

echo "START $(date) | config=$CONFIG | cpus=${SLURM_CPUS_PER_TASK:-?}"
python -m data_generation.run_sims --config "$CONFIG"
echo "DONE  $(date)"
