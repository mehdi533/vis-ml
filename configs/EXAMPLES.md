# YAML Config Examples

Copy any example to your working location and tweak values. Each example matches a script/entrypoint in the repo.

Canonical example location: `configs/examples/`

- `configs/examples/sim_generation.example.yaml` → used by `data_generation/run_sims.py`.
- `configs/examples/cost_mtlsh_convex.example.yaml` → cost/model config for convex scheduling helpers in `scheduling/`.
- `configs/examples/cost_icnn_convex.example.yaml` → cost/model config for ICNN-based scheduling.
- `configs/examples/train_sweep.example.yaml` → training sweep for `models/train_sweep.py`.
- `configs/model/*.yaml` → curated Chapter 5 model-side experiment pack.
- Shared runtime registry/cost tables live under `configs/shared/`.
- Standard IEEE39 (sync-gen) response runner:
  `scripts/report/run_std_ieee39_response.sh --config configs/presentation/presentation_vis_case.yaml`
  This maps presentation disturbance settings to `data_generation/run_sims.py` using
  `data_generation/andes_cases/ieee39_full.xlsx` and writes COI frequency/RoCoF traces to
  `results/presentation_vis/<case_label>/std_ieee39/`.

Usage:
1. Copy from canonical examples, e.g.:
   `cp configs/examples/sim_generation.example.yaml tmp/my_sim_generation.yaml`
2. Adjust paths so they are *relative to repo root*.
3. Keep seeds/paths stable for reproducibility.
