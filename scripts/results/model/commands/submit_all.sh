#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

mkdir -p results/thesis_model_results/logs

sbatch results/thesis_model_results/commands/01_scaler_comparison.sh
sbatch results/thesis_model_results/commands/02_loss_comparison.sh
sbatch results/thesis_model_results/commands/03_architecture_comparison.sh
sbatch results/thesis_model_results/commands/04_shortlist_detailed_eval.sh
sbatch results/thesis_model_results/commands/05_feature_relevance.sh
sbatch results/thesis_model_results/commands/06_export_retained_model.sh
job07=$(sbatch results/thesis_model_results/commands/07_mtlsh_embeddability_tradeoff.sh | awk '{print $4}')
job08=$(sbatch results/thesis_model_results/commands/08_mlp_she_style_comparison.sh | awk '{print $4}')
sbatch --dependency=afterok:${job07}:${job08} results/thesis_model_results/commands/14_relu_size_family_comparison.sh
sbatch results/thesis_model_results/commands/15_embedding_scaler_comparison_mtlsh.sh
sbatch results/thesis_model_results/commands/09_seed_robustness_shortlist.sh
sbatch results/thesis_model_results/commands/10_boundary_region_eval.sh
sbatch results/thesis_model_results/commands/11_architecture_comparison_kendall_standard.sh
sbatch results/thesis_model_results/commands/12_optimization_ready_mtlsh.sh
sbatch results/thesis_model_results/commands/13_convex_family_favorable.sh
sbatch results/thesis_model_results/commands/18_train_optimization_ready_variants.sh
sbatch results/thesis_model_results/commands/19_kan_feature_function_study.sh
