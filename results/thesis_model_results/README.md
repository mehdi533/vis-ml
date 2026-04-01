# Thesis Model Results Pack

This folder packages the **model/surrogate-only** experiments for Chapter 5 / *Numerical Results and Discussion*.

It does **not** cover optimization tables, scheduling comparisons, or ANDES replay validation.

## Structure

- `configs/`: YAML configs for each model-side experiment family
- `commands/`: exact shell commands to run the experiments and export thesis tables
- `tables/`: recommended destination for aggregated CSV tables created by `models/summarize_sweep.py`
- `exports/`: recommended destination for the retained-model bundle exported by `models/export_retained_model.py`

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
- If your exported dataset uses different control-feature names for PICNN `u_feature_cols`, adjust the PICNN configs before training.

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

## Recommended run order

1. `commands/01_scaler_comparison.sh`
2. `commands/02_loss_comparison.sh`
3. `commands/03_architecture_comparison.sh`
4. `commands/04_shortlist_detailed_eval.sh`
5. `commands/05_feature_relevance.sh`
6. `commands/06_export_retained_model.sh`

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
