# Thesis Data Generation Results Pack

This folder contains reproducible data-generation runs for the main disturbance families used in the thesis workflow.

## Structure

- `configs/`: backward-compatible symlink to canonical source configs in `configs/data_generation/`
- `notebooks/`: analysis notebooks for loading scenario CSVs and building thesis figures
- `scripts/run_all.sh`: runs all five scenarios sequentially on one machine
- `scripts/*.slurm`: cluster-ready SLURM job files, one per scenario
- `scripts/submit_all.sh`: submits all five SLURM jobs
- `results/`: recommended destination for generated CSV files

## Notebook Entry Points

For dataset understanding and surrogate-learning diagnostics, start with:

- `notebooks/05_learning_task_domain_and_mapping.ipynb`
  Integrated notebook for label distributions, scenario-conditioned coverage, physical correlations, and simple linear-regression probes that show what low-order mappings are already learnable from the retained dataset.

For the results section, start with these editable notebooks:

- `notebooks/10_results_section_data_loader.ipynb`
  Loads the scenario CSVs, checks counts, and builds simple custom tables.

- `notebooks/11_results_section_load_mismatch_figures.ipynb`
  Focuses on the four load-mismatch scenarios and gives direct-edit cells for scatter plots, histograms, and bin summaries.

- `notebooks/12_results_section_line_outage_figures.ipynb`
  Focuses on outage scenarios and mixed outage-plus-load scenarios, including line-identity counts and severity plots.

- `notebooks/13_results_section_figure_sandbox.ipynb`
  Scratch notebook for trying custom figure ideas without touching the more structured notebooks.

## Scenarios

The pack includes five 10k-sample scenarios:

1. `load_mismatch_only.yaml`
   Global load mismatch only.

2. `line_outages_only.yaml`
   Line outages only.

3. `line_outages_plus_global_load_mismatch.yaml`
   Line outages with a simultaneous global load mismatch.

4. `zone_based_load_mismatch.yaml`
   Load mismatch applied only to the PQ loads whose `owner` labels match the configured list.

5. `line_outages_plus_zone_based_load_mismatch.yaml`
   Line outages with a simultaneous zone-filtered load mismatch.

## How To Run

Run locally from the repository root:

```bash
results/thesis_data_generation_results/scripts/run_all.sh
```

Or run one scenario directly:

```bash
python3 data_generation/run_sims.py \
  --config configs/data_generation/load_mismatch_only.yaml
```

For the cluster workflow, submit all five jobs with:

```bash
results/thesis_data_generation_results/scripts/submit_all.sh
```

The individual SLURM job files are:

- `results/thesis_data_generation_results/scripts/run_load_mismatch_only.slurm`
- `results/thesis_data_generation_results/scripts/run_line_outages_only.slurm`
- `results/thesis_data_generation_results/scripts/run_line_outages_plus_global_load_mismatch.slurm`
- `results/thesis_data_generation_results/scripts/run_zone_based_load_mismatch.slurm`
- `results/thesis_data_generation_results/scripts/run_line_outages_plus_zone_based_load_mismatch.slurm`

## What You Will Likely Edit

- `contingency.load_step.scale`
  Load mismatch range and optional bins.

- `contingency.line_n1.line_ids`
  Explicit line list for the outage scenarios. Leave empty to sample from all in-service lines.

- `load.owners`
  Zone filter for the zone-based scenarios. This uses the current pipeline behavior, where zones are recovered through the `owner` field on the PQ devices.

- `seed`, `workers`, `output_dir`
  Run-level settings.

## Notes

- Each scenario writes to its own subdirectory under `results/thesis_data_generation_results/results/`.
- The current pipeline still uses the existing data-generation logic in `data_generation/run_sims.py`; these configs do not add new sampling logic on top of it.
