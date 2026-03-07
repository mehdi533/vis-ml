# YAML Config Examples

Copy any example to your working location and tweak values. Each example matches a script/entrypoint in the repo.

- `configs/sim_generation.example.yaml` → used by `data_generation/run_sims.py`, `experiments/run_forecast_opt_sim.py`, `scheduling/problem.py` (flag `--config`).
- `configs/cost_mtlsh_convex.example.yaml` → cost/model config for convex scheduling (flag `--cost-config` in `experiments/*` and `scheduling/*`).
- `configs/cost_icnn_convex.example.yaml` → cost/model config for ICNN-based scheduling.
- `configs/train_sweep.example.yaml` → training sweep for `models/train_sweep.py` and similar jobs.

Usage:
1. Copy the example: `cp configs/sim_generation.example.yaml experiments/generation.yaml` (or another path).
2. Adjust paths so they are *relative to repo root*.
3. Keep seeds/paths stable for reproducibility.
