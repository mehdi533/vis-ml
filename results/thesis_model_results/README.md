# Thesis Model Results Pack

This folder packages the **model/surrogate-only** experiments for Chapter 5 / *Numerical Results and Discussion*.

It does **not** cover optimization tables, scheduling comparisons, or ANDES replay validation.

## Structure

- `configs/model/`: canonical source configs for model-side experiments (the local `configs/` path is a compatibility symlink)
- `commands/`: cluster-oriented launcher scripts for the experiments and thesis table exports
- `scripts/`: additional cluster-oriented launcher scripts for thesis analysis sweeps
- `src/`: small thesis-specific aggregation and analysis utilities
- `tables/`: recommended destination for aggregated CSV tables created by `scripts/model/summarize_sweep.py`
- `exports/`: recommended destination for retained-model bundles exported via `models.utils.export_retained_model_bundle`
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
| Preprocessing comparison (main MLP table) | `configs/model/scaler_comparison_mlp.yaml` | `commands/01_scaler_comparison.sh` |
| Preprocessing comparison (appendix variants) | `configs/model/scaler_comparison_mtlsh.yaml`, `configs/model/scaler_comparison_mtlgsh.yaml`, `configs/model/scaler_comparison_picnn.yaml` | `commands/01_scaler_comparison.sh` |
| Loss comparison (main MLP table) | `configs/model/loss_comparison_mlp.yaml` | `commands/02_loss_comparison.sh` |
| Loss comparison (appendix variants) | `configs/model/loss_comparison_mtlsh.yaml`, `configs/model/loss_comparison_mtlgsh.yaml`, `configs/model/loss_comparison_picnn.yaml` | `commands/02_loss_comparison.sh` |
| Embeddable architecture comparison | `configs/model/architecture_comparison_core.yaml` | `commands/03_architecture_comparison.sh` |
| Exploratory architecture comparison (attention) | `configs/model/architecture_comparison_exploratory.yaml` | `commands/03_architecture_comparison.sh` |
| Shortlisted detailed error analysis | `configs/model/shortlist_detailed_eval.yaml` | `commands/04_shortlist_detailed_eval.sh` |
| Feature relevance | `configs/model/feature_relevance_attention.yaml`, `configs/model/feature_groups.yaml` | `commands/05_feature_relevance.sh` |
| Retained surrogate export | `configs/model/architecture_comparison_core.yaml` + its sweep outputs | `commands/06_export_retained_model.sh` |
| MTLSH embeddability / complexity tradeoff | `configs/model/mtlsh_embeddability_tradeoff.yaml`, `configs/model/mtlsh_embeddability_tradeoff_small.yaml` | `commands/07_mtlsh_embeddability_tradeoff.sh` |
| Two-MLP comparison in the style of She et al. | `configs/model/mlp_she_style_comparison.yaml` | `commands/08_mlp_she_style_comparison.sh` |
| Small ReLU-family comparison (She-style baseline, MLP, PICNN, MTLSH, MTLGSH) | `configs/model/relu_size_mlp_picnn.yaml`, `configs/model/relu_size_mtlgsh.yaml` plus outputs from commands 07 and 08 | `commands/14_relu_size_family_comparison.sh` |
| Optimizer-contract scaler comparison for big-M tightness | `configs/model/embedding_scaler_comparison_mtlsh.yaml` | `commands/15_embedding_scaler_comparison_mtlsh.sh` |
| Shortlist seed robustness | `configs/model/seed_robustness_shortlist.yaml` | `scripts/09_seed_robustness_shortlist.sh` |
| Boundary / stress-region evaluation | `configs/model/boundary_region_eval.yaml` + `configs/model/shortlist_detailed_eval.yaml` outputs | `scripts/10_boundary_region_eval.sh` |
| Architecture complexity / fairness summary | reuses `configs/model/architecture_comparison_core.yaml` outputs | produced by `commands/03_architecture_comparison.sh` via `src/complexity_summary.py` |
| Alternate architecture comparison under retained preprocessing/loss candidate | `configs/model/architecture_comparison_core_kendall_standard.yaml`, `configs/model/architecture_comparison_exploratory_kendall_standard.yaml` | `commands/11_architecture_comparison_kendall_standard.sh` |
| Optimization-ready MTLSH surrogate handoff | `configs/model/optimization_ready_mtlsh.yaml` | `commands/12_optimization_ready_mtlsh.sh` |
| Convex-family favorable comparison | `configs/model/convex_family_favorable.yaml` | `commands/13_convex_family_favorable.sh` |
| Compact KAN feature-function study | `configs/model/compact_kan_feature_function_study.yaml` | `commands/19_kan_feature_function_study.sh` |
| Exploratory KAN learned-function export | existing compact `MTLGSH_KAN_SHARED` run directory + `src/export_kan_functions.py` | `commands/16_export_kan_functions.sh` |
| Exploratory KAN spline-shape study | existing compact `MTLGSH_KAN_SHARED` run directory + `src/kan_spline_study.py` | `commands/17_kan_spline_study.sh` |

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
14. Run `sbatch results/thesis_model_results/commands/12_optimization_ready_mtlsh.sh` to train and export the optimizer-compatible MTLSH surrogate bundle.
15. Run `sbatch results/thesis_model_results/commands/13_convex_family_favorable.sh` to compare the convex families under a broader schedule-based `u` split and larger widths.
16. If you want a dedicated appendix-style KAN run, first submit `sbatch results/thesis_model_results/commands/19_kan_feature_function_study.sh`. This trains a compact one-layer shared-KAN model for feature-function interpretation and then exports the spline diagnostics.
17. You can rerun `sbatch results/thesis_model_results/commands/16_export_kan_functions.sh` or `sbatch results/thesis_model_results/commands/17_kan_spline_study.sh` afterward if you want the export steps separately from training.

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

The table-generation commands use `scripts/model/summarize_sweep.py` so the thesis tables can be rebuilt from the run outputs without manual editing.

The retained-family ablation is not packaged as a separate sweep because `mtlsh_embeddability_tradeoff.yaml` already provides the retained-family size sweep; the new complexity summary and tradeoff table are intended to present that result directly.

`configs/model/optimization_ready_mtlsh.yaml` is intentionally different from the full thesis shortlist configs: it uses the schema-defined `x_op` and `x_cont` feature families, keeps the schedulable local channels `{M_i, D_i}`, and now also retains the aggregate scheduling features `{M_agg, D_agg}` from `x_sched`, while still dropping the `x0__*` initial-state channels and non-feature extras.

For the current architecture/radius comparison reruns, the main chapter-facing sweeps still use `training.epochs: 1000` rather than `200`, but the optimizer-ready handoff configs are capped at `training.epochs: 700` to fit the shorter retraining budget while rebuilding the retained artifacts.

`configs/model/embedding_scaler_comparison_mtlsh.yaml` still follows the reduced optimization-ready training contract focused on directly embeddable scheduling inputs. If you want its big-M diagnostic to match the retained optimizer-facing artifacts exactly, rerun it after the optimizer-side feature builder is updated to handle the reintroduced aggregate features `{M_agg, D_agg}` consistently. Until that is done, `scaler_bigm_summary.py` fails explicitly instead of silently pretending the MILP box matches the trained feature contract. In that table, affine scalers (`minmax`, `standard`, `robust`) can still be compared directly once the feature builder is aligned, while `log1p` is flagged as not directly compatible with the current affine input-link formulation.
