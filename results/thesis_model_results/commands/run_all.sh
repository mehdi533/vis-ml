#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export KMP_USE_SHM="${KMP_USE_SHM:-0}"

bash results/thesis_model_results/commands/01_scaler_comparison.sh
bash results/thesis_model_results/commands/02_loss_comparison.sh
bash results/thesis_model_results/commands/03_architecture_comparison.sh
bash results/thesis_model_results/commands/04_shortlist_detailed_eval.sh
bash results/thesis_model_results/commands/05_feature_relevance.sh
bash results/thesis_model_results/commands/06_export_retained_model.sh
bash results/thesis_model_results/commands/07_mtlsh_embeddability_tradeoff.sh
bash results/thesis_model_results/commands/08_mlp_she_style_comparison.sh
bash results/thesis_model_results/commands/09_seed_robustness_shortlist.sh
bash results/thesis_model_results/commands/10_boundary_region_eval.sh
bash results/thesis_model_results/commands/11_architecture_comparison_kendall_standard.sh
bash results/thesis_model_results/commands/12_optimization_ready_mtlsh.sh
bash results/thesis_model_results/commands/13_convex_family_favorable.sh
