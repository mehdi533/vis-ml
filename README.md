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

## Fast Entry Points

- Data generation: `python data_generation/run_sims.py --config configs/data_generation/generation.yaml`
- Model training: `python models/train_sweep.py --config configs/model/train_sweep.yaml`
- Scheduling/optimization smoke (recommended first run): `python scheduling/run_experiment_suite.py --suite configs/scheduling/smoke/formulation_comparison_smoke.yaml --execution-mode subprocess`
- Single optimization run (strict base config): `python scheduling/problem.py --config configs/scheduling/base_optimization.yaml`

## Acknowledgements

This work builds on open-source tools and libraries, especially ANDES, pandapower, CVXPY, and PyTorch. I would like to thank again Dr. James Ciyu Qin and Mr. Renyou Xie for their support during the development of this project. 
