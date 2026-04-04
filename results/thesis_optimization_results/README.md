# Thesis Optimization Results Pack

This folder contains the thesis-oriented optimization workflow for Chapter 5, focused on the preventive scheduling results, surrogate-embedded formulations, and ANDES replay validation.

## Scope

Included:
- formulation comparison across the thesis formulations
- structured per-run exports for dispatch, VIS, predicted metrics, and line-security checks
- replay validation of surrogate-embedded schedules in ANDES
- thesis-ready postprocessing into tables, merged CSVs, and figures
- lightweight notebooks that reuse the script-generated outputs

Not included:
- surrogate training and architecture benchmarking
- generic dashboard infrastructure

## Folder Layout

- `configs/base_optimization.yaml`
  Shared optimization configuration for the retained surrogate, solver defaults, and equation-map metadata.

- `configs/suites/formulation_comparison.yaml`
  Main formulation comparison suite. The runner is now scenario-aware: add a top-level `scenarios:` list here when final thesis scenarios are fixed.

- `configs/suites/security_checks.yaml`
  Security-check run for the retained preventive formulation.

- `configs/suites/redispatch_sensitivity.yaml`
  Optional redispatch sensitivity run.

- `configs/suites/area_vis_comparison.yaml`
  She-style comparison augmented with an area-tied VIS formulation.

- `configs/suites/zone_mismatch_vis_sensitivity.yaml`
  Zone-based load-mismatch sensitivity suite for scheduled IBR inertia and damping.

- `configs/smoke/`
  Reduced smoke-test configs that exercise the cluster launchers without writing into the main production `results/` tree.

- `configs/replay/replay_validation.yaml`
  Replay-validation config. It can read summary JSONs directly or pull all relevant runs from the suite summary.

- `configs/analysis/results_pack.yaml`
  Thesis postprocessing settings: formulation order, retained formulation, and plotting defaults.

- `scripts/run_formulations.sh`
  Runs the formulation comparison suite.

- `scripts/run_debug.sh`
  Runs the full debug suite (includes feasibility block checks; slower on laptops).

- `scripts/run_debug_local.sh`
  Runs a faster local smoke-test suite for optimization wiring and exports.

- `scripts/run_security_checks.sh`
  Runs the retained preventive formulation for security-check reporting.

- `scripts/run_redispatch_sensitivity.sh`
  Runs the optional redispatch sensitivity.

- `scripts/run_replay_validation.sh`
  Runs ANDES replay validation for the configured surrogate-embedded formulations.

- `scripts/run_she_style_comparison.sh`
  Runs a four-method optimization comparison aligned with She et al. (Method I-IV style).

- `scripts/run_area_vis_comparison.sh`
  Runs the area-VIS comparison suite and builds area-wise VIS allocation tables and figures.

- `scripts/run_zone_mismatch_vis_sensitivity.sh`
  Runs owner/zone-targeted load-mismatch scenarios and builds VIS allocation sensitivity tables and figures.

- `scripts/run_cluster_smoke.sh`
  Runs a reduced end-to-end smoke pass for the production launchers and writes outputs under `local_validation/smoke/`.

- `scripts/run_postprocess.sh`
  Builds thesis tables and figures from available optimization and replay outputs.

- `scripts/run_all.sh`
  Runs the full pack end-to-end, including postprocessing.

- `src/build_outputs.py`
  Main thesis postprocessing entry point.

- `src/analysis_utils.py`
  Loaders and aggregation helpers for formulation, dispatch, replay, and security data.

- `src/validation_utils.py`
  Predicted-vs-replayed feasibility classification helpers.

- `notebooks/01_formulation_comparison.ipynb`
- `notebooks/02_cost_impact.ipynb`
- `notebooks/03_dispatch_vis_analysis.ipynb`
- `notebooks/04_predicted_vs_replayed.ipynb`
- `notebooks/05_constraint_satisfaction.ipynb`
  Thin notebooks for inspecting the script-generated thesis artifacts.

## Exact Commands

Run from repository root.

1. Formulation comparison:
```bash
results/thesis_optimization_results/scripts/run_formulations.sh
```

2. Cluster smoke test:
```bash
results/thesis_optimization_results/scripts/run_cluster_smoke.sh
```

3. Local debug smoke test:
```bash
results/thesis_optimization_results/scripts/run_debug.sh
```

3b. Fast local debug smoke test:
```bash
results/thesis_optimization_results/scripts/run_debug_local.sh
```

4. Security checks:
```bash
results/thesis_optimization_results/scripts/run_security_checks.sh
```

5. She et al. style method comparison:
```bash
results/thesis_optimization_results/scripts/run_she_style_comparison.sh
```

6. Area-wise VIS comparison:
```bash
results/thesis_optimization_results/scripts/run_area_vis_comparison.sh
```

7. Zone-based load-mismatch VIS sensitivity:
```bash
results/thesis_optimization_results/scripts/run_zone_mismatch_vis_sensitivity.sh
```

8. Optional redispatch sensitivity:
```bash
results/thesis_optimization_results/scripts/run_redispatch_sensitivity.sh
```

9. Replay validation:
```bash
results/thesis_optimization_results/scripts/run_replay_validation.sh
```

10. Build thesis tables and figures from the generated outputs:
```bash
results/thesis_optimization_results/scripts/run_postprocess.sh
```

11. Full pipeline:
```bash
results/thesis_optimization_results/scripts/run_all.sh
```

If needed, set the interpreter explicitly:
```bash
PYTHON_BIN=../venv/bin/python results/thesis_optimization_results/scripts/run_all.sh
```

## Main Outputs

Raw suite outputs:
- `results/thesis_optimization_results/results/formulation_comparison_summary.csv`
- `results/thesis_optimization_results/results/formulation_comparison_summary.json`
- `results/thesis_optimization_results/results/she_style_comparison_summary.csv`
- `results/thesis_optimization_results/results/she_style_comparison_summary.json`
- `results/thesis_optimization_results/results/area_vis_comparison_summary.csv`
- `results/thesis_optimization_results/results/area_vis_comparison_summary.json`
- `results/thesis_optimization_results/results/zone_mismatch_vis_sensitivity_summary.csv`
- `results/thesis_optimization_results/results/zone_mismatch_vis_sensitivity_summary.json`
- `results/thesis_optimization_results/results/replay_validation_summary.csv`
- `results/thesis_optimization_results/results/replay_validation_detail.csv`

Per-run outputs:
- `results/thesis_optimization_results/results/formulations/<run or scenario>/<...>_summary.json`
- `results/thesis_optimization_results/results/formulations/<run or scenario>/<...>_dispatch_impact.csv`
- `results/thesis_optimization_results/results/formulations/<run or scenario>/<...>_predicted_metrics.csv`
- `results/thesis_optimization_results/results/formulations/<run or scenario>/<...>_constraint_blocks.csv`

Thesis-ready postprocessed artifacts:
- `results/thesis_optimization_results/outputs/tables/formulation_catalog.csv`
- `results/thesis_optimization_results/outputs/tables/formulation_kpis.csv`
- `results/thesis_optimization_results/outputs/tables/dispatch_generator_comparison.csv`
- `results/thesis_optimization_results/outputs/tables/dispatch_ibr_comparison.csv`
- `results/thesis_optimization_results/outputs/tables/area_vis_comparison_unit_allocations.csv`
- `results/thesis_optimization_results/outputs/tables/area_vis_comparison_by_area.csv`
- `results/thesis_optimization_results/outputs/tables/zone_mismatch_vis_sensitivity_unit_allocations.csv`
- `results/thesis_optimization_results/outputs/tables/zone_mismatch_vis_sensitivity_by_scenario.csv`
- `results/thesis_optimization_results/outputs/tables/zone_mismatch_vis_sensitivity_by_area.csv`
- `results/thesis_optimization_results/outputs/tables/replay_metric_summary.csv`
- `results/thesis_optimization_results/outputs/tables/constraint_satisfaction_by_run.csv`
- `results/thesis_optimization_results/outputs/tables/constraint_satisfaction_by_formulation.csv`
- `results/thesis_optimization_results/outputs/merged_results/formulation_run_catalog.csv`
- `results/thesis_optimization_results/outputs/merged_results/predicted_vs_replayed_metrics.csv`
- `results/thesis_optimization_results/outputs/figures/cost_impact_by_formulation.png`
- `results/thesis_optimization_results/outputs/figures/cost_impact_by_formulation.pdf`
- `results/thesis_optimization_results/outputs/figures/dispatch_vis_comparison_<scenario>.png`
- `results/thesis_optimization_results/outputs/figures/dispatch_vis_comparison_<scenario>.pdf`
- `results/thesis_optimization_results/outputs/figures/area_vis_comparison_totals.png`
- `results/thesis_optimization_results/outputs/figures/area_vis_comparison_units.png`
- `results/thesis_optimization_results/outputs/figures/zone_mismatch_vis_sensitivity_unit_deltas.png`
- `results/thesis_optimization_results/outputs/figures/zone_mismatch_vis_sensitivity_area_totals.png`
- `results/thesis_optimization_results/outputs/figures/predicted_vs_replayed_metrics.png`
- `results/thesis_optimization_results/outputs/figures/predicted_vs_replayed_metrics.pdf`
- `results/thesis_optimization_results/outputs/figures/constraint_satisfaction_breakdown.png`
- `results/thesis_optimization_results/outputs/figures/constraint_satisfaction_breakdown.pdf`

CSV, Markdown, and LaTeX variants are written for the main thesis tables under `outputs/tables/`.

Local validation outputs:
- `results/thesis_optimization_results/local_validation/debug/`
- `results/thesis_optimization_results/local_validation/debug_local/`
- `results/thesis_optimization_results/local_validation/milp_isolation_local/`
- `results/thesis_optimization_results/local_validation/smoke/`

## Mapping to Thesis Sections

1. Compared optimization formulations
- `outputs/tables/formulation_catalog.*`
- `outputs/tables/formulation_kpis.*`

2. Cost impact of security constraints
- `outputs/tables/formulation_kpis.*`
- `outputs/figures/cost_impact_by_formulation.*`

3. Dispatch, headroom, and VIS allocation
- `outputs/tables/dispatch_generator_comparison.*`
- `outputs/tables/dispatch_ibr_comparison.*`
- `outputs/figures/dispatch_vis_comparison_<scenario>.*`

4. Predicted versus replayed metric comparison
- `outputs/merged_results/predicted_vs_replayed_metrics.csv`
- `outputs/tables/replay_metric_summary.*`
- `outputs/figures/predicted_vs_replayed_metrics.*`

5. Constraint satisfaction and violation analysis
- `outputs/tables/constraint_satisfaction_by_run.*`
- `outputs/tables/constraint_satisfaction_by_formulation.*`
- `outputs/figures/constraint_satisfaction_breakdown.*`

## She et al. Style Comparison Mapping

Configuration:
- `configs/suites/she_vis_rted_style_comparison.yaml`

Method mapping in this repo:
- Method I: `she_method_i_rted` (ED only)
- Method II: `she_method_ii_dyn_fixed_md_no_reserve` (dynamic surrogate constraints, fixed M/D, no Delta-P dispatch-output link)
- Method III: `she_method_iii_dyn_fixed_md_with_reserve` (dynamic surrogate constraints, fixed M/D, with Delta-P dispatch-output link)
- Method IV: `she_method_iv_vis_rted_full` (dynamic surrogate constraints, optimized M/D, with Delta-P dispatch-output link)

Outputs:
- `results/she_style_comparison_summary.csv`
- `results/she_style_comparison_summary.md`
- `results/she_style_comparison_summary.json`
- `results/she_style_formulations/<formulation>/<run>_summary.json`

## Area-Wise VIS Comparison Mapping

Configuration:
- `configs/suites/area_vis_comparison.yaml`

Method mapping in this repo:
- Method I: `she_method_i_rted` (ED only)
- Method II: `she_method_ii_dyn_fixed_md_no_reserve` (dynamic surrogate constraints, fixed M/D, no Delta-P dispatch-output link)
- Method III: `she_method_iii_dyn_fixed_md_with_reserve` (dynamic surrogate constraints, fixed M/D, with Delta-P dispatch-output link)
- Method IV: `she_method_iv_vis_rted_full` (dynamic surrogate constraints, optimized per-unit M/D)
- Method IV-A: `she_method_iv_vis_rted_area_tied` (dynamic surrogate constraints, optimized M/D tied within each geographic area)

Derived area split for IEEE 39 REGCV1 units:
- `WEST`: `REGCV1_1`, `REGCV1_3`
- `EAST`: `REGCV1_2`, `REGCV1_4`

Outputs:
- `results/area_vis_comparison_summary.csv`
- `results/area_vis_comparison_summary.md`
- `results/area_vis_comparison_summary.json`
- `outputs/tables/area_vis_comparison_unit_allocations.*`
- `outputs/tables/area_vis_comparison_by_area.*`
- `outputs/figures/area_vis_comparison_totals.*`
- `outputs/figures/area_vis_comparison_units.*`

## Zone-Based Load Mismatch VIS Sensitivity

Configuration:
- `configs/suites/zone_mismatch_vis_sensitivity.yaml`

Scenario mapping:
- `global_uniform`: uniform load mismatch across all PQ loads
- `zone_owner_1`: mismatch only in owner/zone bucket `1`
- `zone_owner_2`: mismatch only in owner/zone bucket `2`
- `zone_owner_3`: mismatch only in owner/zone bucket `3`
- `zone_owner_4`: mismatch only in owner/zone bucket `4`

Compared formulations:
- `retained_vis`: final retained preventive formulation
- `retained_vis_area_tied`: same formulation with area-tied `M_i, D_i`

Outputs:
- `results/zone_mismatch_vis_sensitivity_summary.csv`
- `results/zone_mismatch_vis_sensitivity_summary.md`
- `results/zone_mismatch_vis_sensitivity_summary.json`
- `outputs/tables/zone_mismatch_vis_sensitivity_unit_allocations.*`
- `outputs/tables/zone_mismatch_vis_sensitivity_by_scenario.*`
- `outputs/tables/zone_mismatch_vis_sensitivity_by_area.*`
- `outputs/figures/zone_mismatch_vis_sensitivity_unit_deltas.*`
- `outputs/figures/zone_mismatch_vis_sensitivity_area_totals.*`

## Debug Smoke Test

Debug suite config:
- `configs/suites/formulation_comparison_debug.yaml`

Fast local suite config:
- `configs/suites/formulation_comparison_debug_local.yaml`

Default debug launcher:
- `scripts/run_debug.sh`

Fast local launcher:
- `scripts/run_debug_local.sh`

Both suites run four formulations without N-1:
- `ed_debug`
- `ed_line_debug`
- `ed_surrogate_debug`
- `ed_line_surrogate_debug`

Key difference:
- `run_debug.sh` keeps feasibility block checks enabled (`feasibility_checks: true`) to diagnose conflicts.
- `run_debug_local.sh` disables feasibility block checks, uses shorter time limits, and keeps the surrogate in MILP mode to mirror the local isolation checks.

Override the suite if needed:
```bash
SUITE_CONFIG=results/thesis_optimization_results/configs/suites/formulation_comparison.yaml \
results/thesis_optimization_results/scripts/run_debug.sh
```

## Scenario Handling

The formulation-suite runner now supports a top-level `scenarios:` list in the suite YAML. Each scenario can provide:

```yaml
scenarios:
  - id: chapter5_nominal
    name: Chapter 5 nominal load-step case
    description: Base thesis scenario
    overrides:
      scenario:
        base_scale: 0.6
        step_scale: 1.2
        load_step_time: 3.0
```

If `scenarios:` is omitted, the runner still records a derived `scenario_id` from the resolved optimization config and keeps the legacy single-scenario result layout.

## Missing Inputs

The pack is designed to fail explicitly when required inputs are still missing. Typical examples:
- no formulation summary yet
  rerun `results/thesis_optimization_results/scripts/run_formulations.sh`
- no replay outputs yet
  rerun `results/thesis_optimization_results/scripts/run_replay_validation.sh`
- missing retained surrogate artifacts
  check `model.model_dir` in `configs/base_optimization.yaml`
- missing ANDES case
  check `system.case` in `configs/base_optimization.yaml`

## Surrogate Feature Contract

The optimization config must use an optimization-ready surrogate, not just any retained thesis model.

- The current full thesis shortlist MTLSH model is trained on `1206` inputs from `results/to_export/simulation_results.csv`.
- That feature set includes roughly `998` `x0__*` initial-state channels and `83` line-contingency fields.
- The current scheduling optimizer does not build those inputs, so changing `x_features` to mirror the full dataset is not sufficient.

Use one of these two paths:

- Train and export a dedicated optimization surrogate on the buildable scheduling subset, then set `model.model_dir`, `model.in_dim`, and `features.x_features` to that artifact exactly.
- Or extend the optimization feature builder first so it can reproduce the larger training contract before embedding the full thesis surrogate.

`configs/base_optimization.yaml` now defaults to `allow_missing_features: false` so a bad feature contract fails explicitly instead of silently zero-filling unsupported inputs.

`run_postprocess.sh` will still build the optimization-only tables and figures if replay outputs are not available yet. `run_all.sh` requires replay outputs and will fail if replay validation cannot complete.

## Notes

- The main retained formulation remains `ED + Line + N-1 + Surrogate`.
- Redispatch is kept as an optional sensitivity, not the main thesis baseline.
- Replay validation currently covers the dynamic-security metrics tracked by the retained surrogate. Line-security remains reported from the optimization-side DC checks unless replay-side line metrics are added later.
