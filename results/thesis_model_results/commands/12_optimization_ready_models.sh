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

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export KMP_AFFINITY="${KMP_AFFINITY:-disabled}"
export KMP_INIT_AT_FORK="${KMP_INIT_AT_FORK:-FALSE}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p results/thesis_model_results/logs results/thesis_model_results/exports

echo "STARTING AT $(date)"

"${PYTHON_BIN}" models/train_sweep.py --config results/thesis_model_results/configs/optimization_ready_models.yaml

"${PYTHON_BIN}" models/export_retained_model.py \
  --summary-csv results/thesis_model_results/outputs/optimization_ready_models/sweep_run_summary.csv \
  --dest-dir results/thesis_model_results/exports/optimization_ready_mlp \
  --filters model=MLP \
  --metric agg_rmse_mean \
  --note "Optimization-ready MLP surrogate trained on the explicit scheduling feature contract with M_i, D_i, M_agg, and D_agg retained."

"${PYTHON_BIN}" models/export_retained_model.py \
  --summary-csv results/thesis_model_results/outputs/optimization_ready_models/sweep_run_summary.csv \
  --dest-dir results/thesis_model_results/exports/optimization_ready_mtlsh \
  --filters model=MTLSH \
  --metric agg_rmse_mean \
  --note "Optimization-ready MTLSH surrogate trained on the explicit scheduling feature contract with M_i, D_i, M_agg, and D_agg retained."

echo "FINISHED AT $(date)"
