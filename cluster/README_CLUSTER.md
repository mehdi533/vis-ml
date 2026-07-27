# Cluster runbook — full VIS-ML campaign

Everything needed to reproduce the full-scale results on a SLURM cluster
(ETH Euler / EPFL SCITAS). The design principle is **fail fast**: a small
*canary* run must pass physical-correctness checks before any large job starts,
so you never spend a node-day on a broken configuration.

> **Model note.** All simulations use the **REGCV1** grid-forming converter
> (swing-equation VSG with schedulable inertia `M`=2H and damping `D`). This is
> the correct model for virtual-inertia scheduling; see
> `docs/model_choice/model_choice.pdf` for the full justification.

## 0. What gets produced
- **6 data-gen scenarios × 20,000 sims = 120,000** ANDES simulations on the
  modified IEEE 39-bus case → per-scenario `simulation_results.csv`.
- A trained model sweep + the retained **optimization-ready MTLSH** surrogate.
- The solved preventive VIS dispatch artifacts.

## 1. Get the code
```bash
git clone <repo-url> vis-ml && cd vis-ml
git checkout claude          # or the merged branch
```

## 2. One-time environment setup (login node)
```bash
bash cluster/setup_env.sh
```
This loads a Python **3.10–3.12** module (edit the `module load` line for your
cluster), creates `.venv`, installs `requirements-core.txt`, and **verifies**
that `andes`, `torch`, `cvxpy` import and that **SCIP** is available. If it prints
`OK: all imports + SCIP present`, you are ready. (No Gurobi licence is needed;
SCIP is the default MILP solver.)

## 3. Validate before you spend compute (the canary)
```bash
sbatch --cpus-per-task=8 --time=00:40:00 --wrap="bash cluster/canary.sh"
```
This generates 300 sims and **asserts** (exits non-zero otherwise): 100% PF
success, no non-finite metrics, load-increase ⇒ frequency drop, disturbance ⇒
larger |RoCoF|, and **higher inertia ⇒ smaller |RoCoF|** (the core physics).
Only launch the full campaign if the canary log ends with `CANARY PASSED`.

## 4. Launch the full campaign (gated + chained)
```bash
bash cluster/submit_all.sh
```
Submits, with SLURM `afterok` dependencies:
`canary → {6 data-gen jobs} → train → optimize`. Nothing heavy runs unless the
canary passes. Monitor with `squeue -u $USER`.

### Or run stages manually
```bash
sbatch cluster/datagen.slurm configs/data_generation/load_mismatch_only.yaml
# ... the other five scenarios ...
sbatch cluster/train.slurm
sbatch cluster/optimize.slurm configs/scheduling/base_optimization.yaml
```

## 5. Cluster-specific edits (do these once)
- `cluster/setup_env.sh` — the `module load` line (Python 3.10–3.12).
- `cluster/*.slurm` — uncomment/set `--account` and `--partition`; the
  `--mail-user` is preset. For a GPU training node, add a GPU partition + `--gpus=1`.
- **Cores vs workers:** each `datagen.slurm` requests `--cpus-per-task=128`.
  Set `workers:` in each data-gen config equal to that value (some configs ship
  with `workers: 256` — reduce to 128, or raise `--cpus-per-task`, to avoid
  oversubscription).

## 6. Expected outputs & sanity
- Each scenario writes `results/thesis_data_generation_results/results/<scenario>/simulation_results.csv`
  with 20,000 rows and `success == True` for (essentially) all of them.
- Re-run the physics assertion on any produced CSV:
  ```bash
  python scripts/validate_andes_physics.py --csv <path>/simulation_results.csv
  ```
  It must print `ALL PHYSICS CHECKS PASS`.
- Merge the per-scenario CSVs (or point the training configs at the desired one)
  before `train.slurm`.

## 7. Wall-clock (order of magnitude)
Per 20,000-sim scenario: a few hours on 128 cores (ANDES TDS dominates). Six
scenarios run in parallel as independent jobs. Training and optimisation are
comparatively short. Budget one cluster-day end-to-end.

## Troubleshooting
- `ModuleNotFoundError: data_generation` → `PYTHONPATH` unset; the provided
  scripts export it. If running by hand: `PYTHONPATH=$PWD python -m data_generation.run_sims ...`
  (run as a **module**, never `python data_generation/run_sims.py`).
- Canary fails a physics check → **stop**; inspect the CSV; do not launch the
  campaign. This is the check that saves you a wasted day.
