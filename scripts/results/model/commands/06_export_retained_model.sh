#!/bin/bash
#SBATCH --job-name=tm_export
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/export_retained_model_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/export_retained_model_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p results/thesis_model_results/logs results/thesis_model_results/exports

echo "STARTING AT $(date)"

python models/train_sweep.py --config configs/model/architecture_comparison_core.yaml

python - <<'PY'
from models.utils import export_retained_model_bundle

export_retained_model_bundle(
    summary_csv="results/thesis_model_results/outputs/architecture_comparison_core/sweep_run_summary.csv",
    dest_dir="results/thesis_model_results/exports/retained_core_model",
    metric="agg_rmse_mean",
    note="Automatically selected best embeddable surrogate from architecture_comparison_core.",
)
print("Exported retained_core_model bundle.")
PY

echo "FINISHED AT $(date)"
