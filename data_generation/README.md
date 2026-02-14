# Data Generation (ANDES TDS → Supervised Dataset)

This part of the repo is used to generate supervised datasets for **Virtual Inertia Scheduling (VIS)** in **IBR-dominated power systems**. It runs **ANDES time-domain simulations (TDS)** on a benchmark grid (IEEE39), applies disturbances (load steps, trips), varies IBR virtual inertia/damping setpoints (M, D), and extracts labels such as **COI frequency nadir**, **RoCoF**, and **IBR power peaks**.

The output is a **flat CSV dataset** that can be used directly for training surrogate models and for end-to-end scheduling evaluation.

Parallel runs are supported via multiprocessing. Set `workers` in `data_generation/generation.yaml` to the number of processes; each worker writes a CSV shard that is merged into the final `output_csv`.

### Load-step selection (owners or explicit PQ list)
- `load.pq_names`: optional list of PQ device names to receive the load step (defaults to all PQs in the case).
- `load.owners`: optional list of owner labels; if provided, only PQs whose `ss.PQ.owner` matches one of these labels will receive the step.  
  Base scaling still applies to **all** PQs/PVs; the owner filter only affects the step disturbance.

### Stratified sampling knobs
- `load_step_scale`: supports `bins` with `low/high/prob/label` to bias small/medium/large disturbances (falls back to uniform if omitted).
- `ibr.M_range` / `ibr.D_range`: support `bins` and `log_uniform` flags to emphasize low-inertia regimes.

### Economic pre-dispatch (optional)
- `ed.enable`: if true, sample two shared cost triples (GENROU vs REGCV1), solve ED, and set generator setpoints before PFlow/TDS.
- `ed.genrou_costs` / `ed.regcv1_costs`: each has `a/b/c` ranges (same for all units in that class).
- `ed.ibr_idx`: indices (PV+Slack order) treated as IBRs for costing; others use the GENROU triple.
- Results (sampled costs, dispatch setpoints) are logged in the output CSV.

### Feature/label outputs (updated)
- Features now drop all ΔQ terms; keep ΔP per PQ and add ΔP aggregated per owner.
- Dispatch features: `P_GENROU_i` (and `P_REGCV1_i` when available) record pre-disturbance setpoints.
- Frequency labels are signed: `dev_COI` (largest signed deviation from f0) and `rocof_COI` (RoCoF with largest magnitude, signed), plus `Delta_P_IBR_i` peaks.
