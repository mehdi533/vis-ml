# Thesis Model Results Pack

This folder packages the **model/surrogate-only** experiments for Chapter 5 / *Numerical Results and Discussion*.

It does **not** cover optimization tables, scheduling comparisons, or ANDES replay validation.

## Structure

- `configs/`: YAML configs for each model-side experiment family
- `commands/`: cluster-oriented launcher scripts for the experiments and thesis table exports
- `scripts/`: additional cluster-oriented launcher scripts for thesis analysis sweeps
- `src/`: small thesis-specific aggregation and analysis utilities
- `tables/`: recommended destination for aggregated CSV tables created by `models/summarize_sweep.py`
- `exports/`: recommended destination for the retained-model bundle exported by `models/export_retained_model.py`
- `logs/`: SLURM stdout/stderr logs on the cluster

## Assumptions

- Training CSV path defaults to `results/to_export/simulation_results.csv`.
- The configs assume the full Chapter 5 target vector:
  - `rocof_COI`
  - `dev_COI`
  - `Delta_P_IBR_1`
  - `Delta_P_IBR_2`
  - `Delta_P_IBR_3`
  - `Delta_P_IBR_4`
- `ignore_missing_drop_cols: true` is enabled so optional metadata columns can be listed safely.
- If your exported dataset uses different control-feature names for PICNN/PICNN\_MTLSH schedule variables, adjust the `u_feature_spec` blocks before training.
- The current PICNN/PICNN\_MTLSH configs treat the full `x_sched` block as the convex `u` family: scheduled inertia, damping, generator dispatch, converter dispatch, and reserve-related fields.

## Experiment map

| Thesis need | Config(s) | Command file |
|---|---|---|
| Training/testing protocol and standard artifacts | Any `train_sweep` config here | all command files |
| Preprocessing comparison (main MLP table) | `configs/scaler_comparison_mlp.yaml` | `commands/01_scaler_comparison.sh` |
| Preprocessing comparison (appendix variants) | `configs/scaler_comparison_mtlsh.yaml`, `configs/scaler_comparison_mtlgsh.yaml`, `configs/scaler_comparison_picnn.yaml` | `commands/01_scaler_comparison.sh` |
| Loss comparison (main MLP table) | `configs/loss_comparison_mlp.yaml` | `commands/02_loss_comparison.sh` |
| Loss comparison (appendix variants) | `configs/loss_comparison_mtlsh.yaml`, `configs/loss_comparison_mtlgsh.yaml`, `configs/loss_comparison_picnn.yaml` | `commands/02_loss_comparison.sh` |
| Embeddable architecture comparison | `configs/architecture_comparison_core.yaml` | `commands/03_architecture_comparison.sh` |
| Exploratory architecture comparison (attention / KAN) | `configs/architecture_comparison_exploratory.yaml` | `commands/03_architecture_comparison.sh` |
| Shortlisted detailed error analysis | `configs/shortlist_detailed_eval.yaml` | `commands/04_shortlist_detailed_eval.sh` |
| Feature relevance | `configs/feature_relevance_attention.yaml`, `configs/feature_groups.yaml` | `commands/05_feature_relevance.sh` |
| Retained surrogate export | `configs/architecture_comparison_core.yaml` + its sweep outputs | `commands/06_export_retained_model.sh` |
| MTLSH embeddability / complexity tradeoff | `configs/mtlsh_embeddability_tradeoff.yaml`, `configs/mtlsh_embeddability_tradeoff_small.yaml` | `commands/07_mtlsh_embeddability_tradeoff.sh` |
| Two-MLP comparison in the style of She et al. | `configs/mlp_she_style_comparison.yaml` | `commands/08_mlp_she_style_comparison.sh` |
| Small ReLU-family comparison (She-style baseline, MLP, PICNN, MTLSH, MTLGSH) | `configs/relu_size_mlp_picnn.yaml`, `configs/relu_size_mtlgsh.yaml` plus outputs from commands 07 and 08 | `commands/14_relu_size_family_comparison.sh` |
| Optimizer-contract scaler comparison for big-M tightness | `configs/embedding_scaler_comparison_mtlsh.yaml` | `commands/15_embedding_scaler_comparison_mtlsh.sh` |
| Shortlist seed robustness | `configs/seed_robustness_shortlist.yaml` | `scripts/09_seed_robustness_shortlist.sh` |
| Boundary / stress-region evaluation | `configs/boundary_region_eval.yaml` + `configs/shortlist_detailed_eval.yaml` outputs | `scripts/10_boundary_region_eval.sh` |
| Architecture complexity / fairness summary | reuses `configs/architecture_comparison_core.yaml` outputs | produced by `commands/03_architecture_comparison.sh` via `src/complexity_summary.py` |
| Alternate architecture comparison under retained preprocessing/loss candidate | `configs/architecture_comparison_core_kendall_standard.yaml`, `configs/architecture_comparison_exploratory_kendall_standard.yaml` | `commands/11_architecture_comparison_kendall_standard.sh` |
| Optimization-ready surrogate handoff | `configs/optimization_ready_mtlsh.yaml` | `commands/12_optimization_ready_mtlsh.sh` |
| Convex-family favorable comparison | `configs/convex_family_favorable.yaml` | `commands/13_convex_family_favorable.sh` |

## Recommended run order

1. Submit one job at a time with `sbatch results/thesis_model_results/commands/01_scaler_comparison.sh`
2. Continue with `sbatch results/thesis_model_results/commands/02_loss_comparison.sh`
3. Then `sbatch results/thesis_model_results/commands/03_architecture_comparison.sh`
4. Then `sbatch results/thesis_model_results/commands/04_shortlist_detailed_eval.sh`
5. Then `sbatch results/thesis_model_results/commands/05_feature_relevance.sh`
6. Finish with `sbatch results/thesis_model_results/commands/06_export_retained_model.sh`
7. Run `sbatch results/thesis_model_results/commands/07_mtlsh_embeddability_tradeoff.sh` for the MTLSH complexity figure, now including the additional `32|16` and `16|8` points.
8. Run `sbatch results/thesis_model_results/commands/08_mlp_she_style_comparison.sh` for the two-MLP comparison.
9. Run `sbatch results/thesis_model_results/commands/14_relu_size_family_comparison.sh` for the small-family ReLU-size comparison that combines the She-style baseline with small MLP, PICNN, MTLSH, and MTLGSH runs.
10. Run `sbatch results/thesis_model_results/commands/15_embedding_scaler_comparison_mtlsh.sh` for the optimizer-contract scaler sweep and the big-M tightness summary under the exact scheduling input box.
11. Run `sbatch results/thesis_model_results/commands/09_seed_robustness_shortlist.sh` for ranking robustness across seeds.
12. Run `sbatch results/thesis_model_results/commands/10_boundary_region_eval.sh` for stressed-subset evaluation.
13. Run `sbatch results/thesis_model_results/commands/11_architecture_comparison_kendall_standard.sh` to compare architectures under `kendall + standard`.
14. Run `sbatch results/thesis_model_results/commands/12_optimization_ready_mtlsh.sh` to train and export the optimizer-compatible surrogate bundle.
15. Run `sbatch results/thesis_model_results/commands/13_convex_family_favorable.sh` to compare the convex families under a broader schedule-based `u` split and larger widths.

For bulk submission, use `results/thesis_model_results/commands/submit_all.sh`.

## Notes on the outputs

- Every run directory produced by `models/train_sweep.py` contains:
  - resolved config
  - fitted scalers
  - model text + model stats
  - best/final state dicts
  - per-target metrics
  - aggregate metrics
  - plots
- Each sweep root contains:
  - `sweep_results.csv` for per-target metrics
  - `sweep_run_summary.csv` for per-run aggregate metrics

The table-generation commands use `models/summarize_sweep.py` so the thesis tables can be rebuilt from the run outputs without manual editing.

The retained-family ablation is not packaged as a separate sweep because `mtlsh_embeddability_tradeoff.yaml` already provides the retained-family size sweep; the new complexity summary and tradeoff table are intended to present that result directly.

`configs/optimization_ready_mtlsh.yaml` is intentionally different from the full thesis shortlist configs: it uses the schema-defined `x_op`, `x_cont`, and `x_sched` feature families from `configs/data_generation_feature_names.yaml`, while dropping the `x0__*` initial-state channels and non-feature extras.

For the current architecture/radius comparison reruns, the main chapter-facing sweeps now use `training.epochs: 1000` rather than `200`. That includes the core/exploratory architecture comparisons, the MTLSH embeddability sweep, and the optimizer-ready MTLSH handoff config.

`configs/embedding_scaler_comparison_mtlsh.yaml` now follows the same broader optimization-ready training contract as `configs/optimization_ready_mtlsh.yaml`: schema-defined `x_op`, `x_cont`, and `x_sched`, with `x0__*` excluded. The associated big-M diagnostic remains useful, but it now depends on the optimization-side feature builder being extended to construct that broader fixed-feature block too. Until that is done, `scaler_bigm_summary.py` fails explicitly instead of silently pretending the MILP box matches the trained feature contract. In that table, affine scalers (`minmax`, `standard`, `robust`) can still be compared directly once the feature builder is aligned, while `log1p` is flagged as not directly compatible with the current affine input-link formulation.
