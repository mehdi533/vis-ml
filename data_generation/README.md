# Data Generation (ANDES TDS -> Supervised Dataset)

This module runs ANDES simulations and writes a flat dataset for VIS model training/evaluation.

Code entrypoint: `data_generation/run_sims.py`  
Config file: `configs/data_generation/generation.yaml`

Core extension points (Phase 1+2 refactor):

- Disturbance dispatcher: `data_generation/disturbance_dispatch.py`
  - `DisturbanceSpec`
  - `DisturbanceHandler`
  - `HANDLERS` and `DisturbanceDispatcher`
- Stable API surface: `data_generation/run_sims.py`
  - `load_config(path)`
  - `run_generation(config_path)`
  - `run_one_sim(config, sim_id, rng=None)`

## YAML configuration

Top-level keys:

- `case`: direct path to the ANDES case file (no resolver/fallback is used).
- `output_dir`: folder for dataset artifacts.
- `output_csv`: final merged CSV filename.
- `seed`: global random seed.
- `n_sims`: number of sampled scenarios.
- `workers`: number of multiprocessing workers.
- `stream_level`: ANDES logger level.
- `features.include_initial_state` (default true): include initial dynamic-state (`x0__*`) columns.

Sampling/disturbance:

- `base_load_scale.low/high`: uniform scale applied to all PQ/PV base loads.
- `contingency.load_step.enable`: enable load step disturbance.
- `contingency.load_step.time`: step application time.
- `contingency.load_step.scale`: scalar or range/bins for step multiplier.
- `contingency.line_n1.enable`: enable line outage contingencies.
- `contingency.line_n1.trip_time`: line trip time.
- `contingency.line_n1.line_ids`: optional subset (line idx/name/uid).
- `contingency.line_n1.max_lines`: cap on sampled lines per sim.

IBR parameters:

- `ibr.n_ibr` / `ibr.indices`: how many REGCV1 devices are sampled.
- `ibr.M_range`, `ibr.D_range`: sampled inertia/damping ranges (supports bins).

Load targeting:

- `load.pq_names`: optional list of PQ names for load-step targeting.
- `load.owners`: optional owner filter for load-step targeting.

Economic dispatch:

- `ed.enable`: run pre-disturbance ED.
- `ed.ibr_idx`: generator indices (PV+Slack order) treated as IBR cost class.
- `ed.genrou_costs.{a,b,c}`: sampled cost ranges for non-IBR class.
- `ed.regcv1_costs.{a,b,c}`: sampled cost ranges for IBR class.
- `ed.solver`: CVXPY solver name.
- `ed.verbose`: solver verbosity.
- `ed.line_limits_enable` (default true): enforce PTDF line constraints in ED.

TDS:

- `tds.t_end`, `tds.t_step`, `tds.method`, `tds.max_iter`, `tds.tol`, `tds.honest`, `tds.fixt`, `tds.shrinkt`, `tds.criteria`, `tds.no_tqdm`.

Plot export:

- `plotter.export`: if true, export one-row plotter snapshot per run.
- `plotter.subdir`: subfolder under `output_dir`.

## What is saved in CSV

Column schema is built via `data_generation.extract_metrics.simulation_row_fieldnames(...)`.

Core metadata:

- `sim_id`, `seed`, `success`
- `cont_type` in `{none, load, line, line_plus_load}`
- `contingency_time`
- `line_uid`, `line_from_bus`, `line_to_bus`
- line metadata is set to `-1` when no line toggle is applied

Base line metrics:

- `line_rating` (p.u.; raw limit divided by system base MVA)
- `pre_fault_flow` (p.u.)
- `pre_fault_loading` (%)

Feature inputs:

- `base_load_scale`, `load_step_scale`, `load_step_time`
- `base_load_p_total`, `base_load_q_total`
- `DELTA_PQ_tot`
- `M_agg`, `D_agg`
- `M_i`, `D_i` for sampled IBRs
- `DELTA_P_<pq_name>` for each selected PQ
- `DELTA_P_OWNER_<owner>` for each owner bucket
- `P_GENROU_i`, `P_REGCV1_i` (dispatch/setpoints)

Frequency/response labels:

- `time_max_dev`
- `rocof_COI` (signed largest-magnitude ROCOF)
- `dev_COI` (signed largest-magnitude frequency deviation)
- `Delta_P_IBR_i` (peak signed REGCV1 active-power delta)
- Per-bus dynamics: `bus_freq_max_abs_dev_<bus>`, `bus_v_max_abs_dev_<bus>`, `bus_rocof_max_abs_<bus>`

Extended line/topology/DC metrics (`line_utils.line_extra_fieldnames(...)`):

- Line params: `line_fn`, `line_Vn1`, `line_Vn2`, `line_r`, `line_x`, `line_b`, `line_g`, `line_b1`, `line_g1`, `line_b2`, `line_g2`, `line_trans`, `line_tap`, `line_phi`
- Ratio: `line_x_over_r` (invalid -> `-1`)
- Prefault directional flows: `pre_p_from`, `pre_p_to` (p.u.)
- Directional loadings: `pre_loading_from`, `pre_loading_to` (%)
- `pre_flow_direction_p` (`sign(pre_p_from)`)
- Bus states: `pre_v_from`, `pre_v_to`, `pre_theta_from`, `pre_theta_to`, `pre_delta_theta`
- Graph criticality: `bus_degree_from`, `bus_degree_to`, `is_bridge`, `n_components_after_trip`, `largest_component_fraction_after_trip`
- System stress: `total_load_p_prefault`, `total_gen_p_prefault`, `reserve_p_total_prefault`, `reserve_q_total_prefault`, `system_max_loading_prefault`, `system_mean_loading_prefault`, `system_top5_loading_mean_prefault`
- DC sensitivity: `ptdf_l1_norm_outaged_line`, `max_abs_lodf_row`, `predicted_max_post_cont_loading_dc` (%, invalid -> `-1`)
- One-hot outage identity: `line_oh_uid_<uid>` in `{0,1}`

Sampling/ED diagnostics:

- `step_bin_label`, `M_bin_label`, `D_bin_label`
- `ed_enabled`, `ed_solver`, `ed_status`
- `ed_total_cost`, `ed_constant_cost`, `ed_energy_cost`, `ed_reserve_cost`, `ed_quadratic_cost`
- Prefault reserve aggregates: `reserve_p_genrou`, `reserve_p_ibr`, `reserve_q_genrou`, `reserve_q_ibr`
- Per-unit schedulable active reserves: `P_GENROU_RESERVE_i`, `P_REGCV1_RESERVE_i`
- Initial dynamic-state columns are written inline to the main CSV with the configured `x0__` prefix when `features.include_initial_state=true`.

## How key quantities are computed

- `pre_fault` line metrics are extracted after `PFlow.run()` and before `TDS.run()`.
- `line_rating` source order: `Line.rate_a/rateA/RATE_A`, fallback to `Line.Sn`, then pandapower branch `RATE_A`.
- p.u. conversion: `rating_pu = rating_raw / system_base_mva`.
- Loading conversion (all exported loading fields): `abs(flow_pu / rating_pu) * 100`.
- `line_x_over_r`: `line_x / line_r`; non-finite mapped to `-1`.
- `predicted_max_post_cont_loading_dc`:
  - Builds DC PTDF/LODF on intact topology.
  - Projects post-contingency flows with outage column.
  - Returns max projected loading in `%`.
  - If unavailable/non-finite, fallback path is used; final non-finite is forced to `-1`.

## ED line limits behavior

When `ed.line_limits_enable=true`:

- PTDF is built from the pandapower-converted pypower case.
- Branch limit comes from `RATE_A`.
- Placeholder/unlimited values are rejected (`<=0`, `NaN`, or `>=1e4`).
- Invalid entries are backfilled from ANDES `Line.rate_a`, then `Line.Sn` when branch/line alignment is available.
- If any branch still has invalid limits, ED fails fast with an explicit error.

This avoids silently accepting fake limits such as `99999`.
