# Config Index

This folder stores canonical YAML configs for all repository workflows.

## Top-Level Layout

- `configs/shared/`: shared registries and tables.
- `configs/examples/`: runnable templates and example configs.
- `configs/data_generation/`: simulation/data-generation scenarios.
- `configs/model/`: model training/evaluation study configs.
- `configs/scheduling/`: optimization, suites, replay, and analysis configs.
- `configs/figure/`: figure-pack and plotting style configs.
- `configs/presentation/`: end-to-end presentation/report pipeline cases.

## Model Configs (Phase 1)

Model config canonical files now start moving under:

- `configs/model/base/`: reusable defaults.
- `configs/model/studies/`: canonical study definitions.

Legacy files in `configs/model/*.yaml` are kept as compatibility wrappers so
existing script entry points continue to work.
