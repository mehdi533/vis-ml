# Debug Model Configs

These YAMLs mirror the main thesis model experiment configs in
`results/thesis_model_results/configs/`, but they are wired to the small sample
dataset at `__simpqareas/exaple_40_samples.csv`.

They are intended for quick smoke tests of the model workflow before launching
the full runs on the larger exported dataset.

Common debug adjustments:
- `data.csv_path` points to `__simpqareas/exaple_40_samples.csv`
- `output_dir` points to `results/thesis_model_results/debug_outputs/...`
- `training.epochs` is reduced to `5`
- `training.batch_train` and `training.batch_eval` are reduced to `8`
- `skip_if_exists` is set to `false`

The experiment naming matches the full pack, with `_debug` appended.
