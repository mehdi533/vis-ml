#!/usr/bin/env bash
#SBATCH --job-name=vis_optimize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=results/logs/optimize_%j.out
#SBATCH --error=results/logs/optimize_%j.err
## #SBATCH --account=<your_account>
## #SBATCH --partition=<partition>
#
# Solves the preventive VIS dispatch (ED + line + N-1 + embedded surrogate).
# Uses the free SCIP solver by default (no Gurobi licence). To use Gurobi, set
# solver.name: GUROBI in the config and ensure a licence is available on the node.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p results/logs
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$PWD"

CONFIG="${1:-configs/scheduling/base_optimization.yaml}"
echo "START $(date) | config=$CONFIG"
python -m scheduling.problem --config "$CONFIG"
echo "DONE  $(date)"
