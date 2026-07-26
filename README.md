# vis-ml

Code repository accompanying the thesis *Optimising VIS with ML for Low-Inertia Power Systems*.

Repository for virtual inertia studies with three core workflow blocks:

1. `data_generation/`: ANDES simulations and dataset extraction.
2. `models/`: surrogate model training/evaluation/export.
3. `scheduling/`: optimization and replay validation.

## Support Folders

- `scripts/results/`: reusable orchestration/post-processing scripts.

## Config Layout

- `configs/shared/`: shared registries/tables used across workflows.
- `configs/data_generation/`: data-generation scenario configs.
- `configs/model/`: model-training/evaluation configs (`base/`, `studies/`, `debug/`, plus compatibility wrappers).
- `configs/scheduling/`: optimization/replay/postprocess source configs.

## Development setup

The pinned stack needs **Python 3.10–3.12** (not 3.13+). CPU-only works.

```bash
python3.12 -m venv .venv
.venv/Scripts/python -m pip install -U pip wheel        # POSIX: .venv/bin/python
.venv/Scripts/python -m pip install -r requirements-core.txt
```

`requirements-core.txt` is `requirements.txt` minus the unused `torch-geometric`
C++ extensions. The MILP solver defaults to **SCIP** (free, via PySCIPOpt) — no
Gurobi licence required; set `solver.name: GUROBI` in a config to use Gurobi.

Run modules with the repo root on `PYTHONPATH`, as modules (running a script by
path can shadow the package, e.g. `models/models.py`):

```bash
PYTHONPATH=. .venv/Scripts/python -m models.train_sweep --config <cfg>
```

- **End-to-end smoke** (data → train → export → SCIP solve): `bash scripts/run_smoke.sh`
- **Tests**: `.venv/Scripts/python -m pytest`

New research results and modules: see `RESULTS.md` and `research/README.md`.

## Fast Entry Points

Run with `PYTHONPATH=.` and the `-m` module form (see above):

- Data generation: `python -m data_generation.run_sims --config configs/data_generation/generation.yaml`
- Model training: `python -m models.train_sweep --config configs/model/train_sweep.yaml`
- Scheduling smoke (SCIP, recommended first run): `python -m scheduling.problem --config configs/scheduling/smoke/optimization_smoke.yaml`
- Single optimization run (strict base config): `python -m scheduling.problem --config configs/scheduling/base_optimization.yaml`

## Acknowledgements

This work builds on open-source tools and libraries, especially ANDES, pandapower, CVXPY, and PyTorch. I would like to thank again Dr. James Ciyu Qin and Mr. Renyou Xie for their support during the development of this project. 
