#!/bin/bash
#SBATCH --job-name=tm_arch_ks
#SBATCH --mail-user=mehdi.abdallahi@epfl.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/architecture_comparison_kendall_standard_job_out%j.out
#SBATCH --error=/cluster/home/mabdallahi/vis-ml/results/thesis_model_results/logs/architecture_comparison_kendall_standard_job_err%j.out
#SBATCH --chdir=/cluster/home/mabdallahi/vis-ml
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p results/thesis_model_results/logs results/thesis_model_results/tables

echo "STARTING AT $(date)"

python models/train_sweep.py --config configs/model/architecture_comparison_core_kendall_standard.yaml
python models/train_sweep.py --config configs/model/architecture_comparison_exploratory_kendall_standard.yaml

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/architecture_comparison_core_kendall_standard \
  --source run \
  --group-by model \
  --value-cols agg_rmse_mean agg_mae_mean n_parameters_trainable \
  --output-csv results/thesis_model_results/tables/architecture_comparison_core_kendall_standard.csv

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/architecture_comparison_core_kendall_standard \
  --source label \
  --group-by model label raw_unit display_unit display_scale \
  --value-cols rmse mae rmse_display mae_display \
  --output-csv results/thesis_model_results/tables/architecture_comparison_core_kendall_standard_by_label.csv

python scripts/results/model/summarize_sweep.py \
  --input-dir results/thesis_model_results/outputs/architecture_comparison_exploratory_kendall_standard \
  --source run \
  --group-by model \
  --value-cols agg_rmse_mean agg_mae_mean n_parameters_trainable \
  --output-csv results/thesis_model_results/tables/architecture_comparison_exploratory_kendall_standard.csv

python results/thesis_model_results/src/complexity_summary.py \
  --input-dir results/thesis_model_results/outputs/architecture_comparison_core_kendall_standard \
  --output-csv results/thesis_model_results/tables/architecture_complexity_fairness_core_kendall_standard.csv

echo "FINISHED AT $(date)"
