# Scheduling Module

This folder contains the optimization and validation workflow used for the thesis power-system scheduling experiments.

## What Happens Inside

1. `problem.py` loads one optimization config, builds the system state in ANDES, constructs CVXPY constraint blocks (ED, surrogate, line, N-1, redispatch), solves, and writes run artifacts.
2. `run_experiment_suite.py` loops over multiple formulations/scenarios, launches `problem.py` (default: subprocess mode), and aggregates summary/failure diagnostics.
3. `replay_validation.py` replays optimized schedules in time-domain simulation (ANDES TDS) and compares predicted vs replayed metrics.
4. `build_outputs.py` calls the thesis post-processing builder to generate final tables/figures.

## Main Files

- `problem.py`: core optimization engine and artifact writer.
- `constraints.py`: input/output/ED/PTDF/N-1 constraint builders.
- `constraints_nn.py`: MILP embedding for supported ReLU surrogate models.
- `utils.py`: config parsing, model/scaler loading, feature extraction bridge, logger setup.
- `run_experiment_suite.py`: suite orchestration, preflight checks, failure categorization, summary exports.
- `replay_validation.py`: simulation replay and safety-parity checks.

## Quick Start

Run from repo root.

```bash
# Smoke suite (recommended first run)
../venv/bin/python scheduling/run_experiment_suite.py --suite configs/scheduling/smoke/formulation_comparison_smoke.yaml

# Single optimization run (strict base config; may be infeasible for some bound sets)
../venv/bin/python scheduling/problem.py --config configs/scheduling/base_optimization.yaml

# Replay validation
../venv/bin/python scheduling/replay_validation.py --config configs/scheduling/smoke/replay_validation_smoke.yaml

# Build post-processed outputs
../venv/bin/python scheduling/build_outputs.py --config configs/scheduling/analysis/results_pack.yaml
```

## Artifact Pattern

- Per-run artifacts from `problem.py`:
  - `<run_tag>_summary.json`
  - `<run_tag>_summary.csv`
  - `<run_tag>_decisions.csv`
  - `<run_tag>_constraint_blocks.csv`
  - `<run_tag>_predicted_metrics.csv`
  - `<run_tag>_dispatch_impact.csv`
- Suite-level artifacts from `run_experiment_suite.py`:
  - `*_summary.csv`, `*_summary.md`, `*_summary.json`
  - `*_failures.csv`, `*_failures.md`, `*_diagnostics.json`
- Replay artifacts from `replay_validation.py`:
  - replay summary CSV/JSON and per-metric detail CSV

## Full Optimization Problem

The scheduler solves a mixed-integer optimization problem assembled block-wise in `problem.py`.

### Decision variables

- `pg in R^{n_g}`: generator dispatch (PV + Slack order used by ANDES/Pandapower bridge).
- `x in R^{n_x}`: surrogate input vector in **scaled space**.
- `y in R^{n_y}`: surrogate output vector in **scaled space**.
- `delta_pg[c, :] in R^{n_g}`: outage-wise redispatch recourse (only when `use_n1_redispatch=true`).
- Binary ReLU indicators from NN embedding (only for uncertain ReLU states).

Raw/scaled mapping used throughout:

- `x_raw[k] = (x[k] - x_scaler.min_[k]) / x_scaler.scale_[k]`
- `y_raw[j] = (y[j] - y_scaler.min_[j]) / y_scaler.scale_[j]`

### Objective

The minimized objective is:

- `C = C_dispatch + C_reserve_up + C_tie`

with:

- `C_dispatch = sum_g (c_g + b_g * pg_g + a_g * pg_g^2)`
- `C_reserve_up = sum_g b_r,g * (pg_max,g - pg_g)`
- `C_tie = tie_breaker_active * sum_j y_j`
  - `tie_breaker_active = tie_breaker` only when:
    - `tie_breaker > 0`, and
    - (`constraints.use_nn = true` and `model.convex = true`)
  - otherwise `tie_breaker_active = 0`

### Constraint blocks

The active model is the concatenation of enabled blocks:
`input`, `output`, `nn`, `line`, `n1` or `n1_redispatch`, `ed`.

#### 1) Input block (`build_input_feature_constraints`)

- Bounds on schedulable surrogate channels:
  - `m_min_sc <= x[M_i] <= m_max_sc`
  - `d_min_sc <= x[D_i] <= d_max_sc`
- Optional VIS tie groups (`vis_tie_groups`):
  - tie selected IBR indices to a group anchor:
    - `x[M_i] = x[M_anchor]`, `x[D_i] = x[D_anchor]`
- Optional dispatch-feature link (if dispatch features exist in surrogate input):
  - `x[P_dispatch_features] = scale .* pg(order) + min`
- Derived-feature equalities (when feature names are present), including:
  - `M_agg`, `D_agg`
  - `P_GENROU_TOTAL`, `P_REGCV1_TOTAL`, `P_DISPATCH_TOTAL`
  - `P_REGCV1_SHARE = P_REGCV1_TOTAL / sum(pd)`
  - reserve channels `P_GENROU_RESERVE_i = pmax_i - pg_i`
  - reserve channels `P_REGCV1_RESERVE_i = pmax_i - pg_i`
  - aggregate reserves `reserve_p_genrou`, `reserve_p_ibr`, `reserve_p_total_prefault`
- Contract safety check:
  - all scheduling-related features must be either free decision-linked or derived-linked
- All remaining non-sched, non-derived features are fixed to seed:
  - `x[k] = x_seed_sc[k]`

#### 2) Output block (`build_output_feature_constraints`)

- Active output bounds:
  - `y_min_sc[j] <= y[j] <= y_max_sc[j]` for selected active outputs
- Optional dispatch-output consistency (`enforce_dispatch_output_link=true`):
  - for mapped IBR pairs `(j, i)`:
  - `y[j] <= scale_j * (pg_max[i] - pg[i]) + min_j`
  - `y[j] >= scale_j * (pg_min[i] - pg[i]) + min_j`

#### 3) NN block (`build_nn_constraints`)

Mode policy is strict:

- non-convex model (`model.convex=false`) -> `constraints.nn_mode=milp`
- convex model (`model.convex=true`) -> `constraints.nn_mode=convex`
- no other mode values are allowed

The scheduler validates this mapping and raises a config error on mismatch.

For each affine layer: `z = W h + b`.

- Interval propagation for each neuron:
  - `z_min`, `z_max`
- ReLU exact MILP with pruning:
  - always-active neurons (`z_min >= 0`): `h_next = z`
  - always-inactive neurons (`z_max <= 0`): `h_next = 0`
  - undecided neurons use binary `a` and:
    - `h_next >= 0`
    - `h_next >= z`
    - `h_next <= z - z_min * (1 - a)`
    - `h_next <= z_max * a`
- Final link to scheduler outputs:
  - `y[j] = NN_j(x)` for enabled heads / subblocks

#### 4) Base-case line block (`build_basecase_line_constraints`)

- DC injections:
  - `inj = Cg * pg - Cd * pd`
- Base-case flows:
  - `f = PTDF * inj`
- Line limits:
  - `-fmax <= f <= fmax`

#### 5) Preventive N-1 block (`build_n1_constraints`)

For each outage candidate `c` (filtered by candidate mask, optional islanding-critical filter, finite LODF):

- `f_post^c = f + LODF[:, c] * f[c]`
- `-fmax <= f_post^c <= fmax`

This is preventive (no redispatch recourse inside outage constraints).

#### 6) N-1 redispatch block (`build_n1_redispatch_constraints`)

If `use_n1_redispatch=true`, outage-wise recourse `delta_pg[c, :]` is introduced.

For each active outage `c`:

- Power balance in recourse:
  - `sum_g delta_pg[c,g] = 0`
- Optional redispatch limits:
  - `delta_pg[c,g] <= redispatch_up[g]`
  - `delta_pg[c,g] >= -redispatch_down[g]`
- Post-redispatch generator bounds:
  - `pg_min <= pg + delta_pg[c,:] <= pg_max`
- Pre-outage redispatched flow:
  - `f_pre^c = PTDF * (Cg * (pg + delta_pg[c,:]) - Cd * pd)`
- Post-outage flow:
  - `f_post^c = f_pre^c + LODF[:, c] * f_pre^c[c]`
- Line limits on monitored lines (`l != c`):
  - `-fmax[l] <= f_post^c[l] <= fmax[l]`

#### 7) ED block (`build_ed_constraints`)

- Generator bounds:
  - `pg_min <= pg <= pg_max`
- Power balance:
  - `sum_g pg_g = sum_d pd_d`

### Block activation logic

Constraint switches in config (`use_input`, `use_output`, `use_nn`, `use_line`, `use_n1`, `use_n1_redispatch`, `use_ed`) decide which blocks are added.

- `use_n1_redispatch=true` replaces plain `n1` block with redispatch recourse block.
- `line`, `n1`, `n1_redispatch`, and `ed` are only meaningful with dispatch variable `pg`.

## Quick Diagnostic and Improvements

High-value improvements (in order):

1. Split `problem.py` into smaller modules (`data_prep`, `constraint_assembly`, `solve`, `artifact_export`).
   - Reason: `problem.py` is very large and currently mixes orchestration, modeling, and reporting.
2. Remove duplicated helpers across scripts.
   - `_sanitize_token`, scenario-id building, path resolution, and CSV writers are repeated in `problem.py`, `run_experiment_suite.py`, and `replay_validation.py`.
3. Centralize ANDES system setup/reset logic.
   - Load scaling and PQ setup logic is repeated in `problem.py` and `replay_validation.py`.
4. Add config schema validation (dataclass/Pydantic) before runtime.
   - Most configs are parsed as nested dicts, so type/key mistakes are caught late.
5. Add automated smoke tests for entrypoints.
   - Basic tests for `problem.py`, `run_experiment_suite.py`, and `replay_validation.py` would reduce regression risk.
