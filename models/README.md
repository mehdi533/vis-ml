# Models Workflow

This folder contains the surrogate-model training, evaluation, and reporting workflow used for the thesis model-side results.

## Main entrypoints

- `models/train_sweep.py`
  Train one or more runs from a YAML config and write a structured run directory plus sweep summaries.
- `models/eval_on_csv.py`
  Re-evaluate a trained run on another CSV using the saved scalers and architecture config.
- `models/summarize_sweep.py`
  Aggregate `sweep_run_summary.csv` or `sweep_results.csv` into thesis-ready comparison tables.
- `models/feature_relevance.py`
  Export attention-based and/or permutation-based feature relevance diagnostics.
- `models/export_retained_model.py`
  Copy the retained model artifacts and selection metadata into a clean bundle.

## Standard run artifacts

Each `train_sweep.py` run directory now contains:

- `run_config.yaml`: resolved config used for that run
- `model.txt`: string representation of the instantiated model
- `model_stats.json`: basic model metadata and parameter counts
- `artifact_manifest.json`: artifact naming metadata, including the primary checkpoint filenames
- `x_scaler.pkl`, `y_scaler.pkl`: fitted scalers
- `<model_name>_state_dict_best.pt`: best validation checkpoint, named from the model family
- `<model_name>_state_dict.pt`: final restored model state (best checkpoint loaded before saving)
- `training_summary.txt`: epoch-by-epoch training log
- `metrics_by_target.csv`: per-output RMSE/MAE/MSE in physical and normalized units
- `metrics_summary.json`: aggregate metrics across outputs
- `rmse_results.txt`: compact human-readable RMSE report
- `loss_curve.png`: train/val/test loss plot
- `scatter_<target>.png`: true-vs-predicted scatter plot per output
- `test_predictions.csv`: optional per-row predictions when `training.save_test_predictions=true`

At the sweep root:

- `sweep_results.csv`: one row per output per run
- `sweep_run_summary.csv`: one row per run with aggregate metrics and parameter counts

## Config notes

- Default config path for `train_sweep.py` is `models/train_sweep.yaml`.
- Sweep grids can now be defined in:
  - `sweep.model`
  - `sweep.training`
  - `sweep.data`
- For nested list-valued sweep options such as `shared_sizes`, use:

```yaml
sweep:
  model:
    shared_sizes:
      values:
        - [128, 64]
        - [256, 128]
```

- Use `data.drop_prefixes` to remove whole families of columns and avoid target leakage.

## Thesis pack

See [thesis_model_results/README.md](/Users/cloud9/Desktop/ETH%20Project/Working%20folder/03_Code/vis-ml/thesis_model_results/README.md) for the curated Chapter 5 model-side experiment pack.
