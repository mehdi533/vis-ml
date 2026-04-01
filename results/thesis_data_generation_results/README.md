# Thesis Data Generation Results Pack

This folder contains reproducible data-generation runs for the main disturbance families used in the thesis workflow.

## Structure

- `configs/`: scenario-specific YAML configs for `data_generation/run_sims.py`
- `scripts/run_all.sh`: runs all five scenarios sequentially on one machine
- `scripts/*.slurm`: cluster-ready SLURM job files, one per scenario
- `scripts/submit_all.sh`: submits all five SLURM jobs
- `results/`: recommended destination for generated CSV files

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
  --config results/thesis_data_generation_results/configs/load_mismatch_only.yaml
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
