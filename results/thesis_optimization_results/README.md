# Thesis Optimization Results Pack

This folder contains the reproducible optimization-side experiments for Chapter 5 (Numerical Results and Discussion), restricted to the optimization workflow.

## Scope

Included:
- formulation comparison (`ED`, `ED+Line`, `ED+Line+N-1`, `ED+Surrogate`, `ED+Line+N-1+Surrogate`, optional redispatch sensitivity)
- structured KPI export for thesis tables
- security checks before replay (line loading and predicted output limit checks)
- replay validation (predicted vs ANDES replayed metrics)

Not included:
- model-only architecture benchmarking/training tables (handled elsewhere)

## Folder Layout

- `configs/base_optimization.yaml`  
  Shared base optimization configuration (system, surrogate, solver defaults, equation map metadata).

- `configs/suites/formulation_comparison.yaml`  
  Main thesis formulation comparison suite.

- `configs/suites/security_checks.yaml`  
  Full preventive formulation run focused on security-check reporting.

- `configs/suites/redispatch_sensitivity.yaml`  
  Optional sensitivity: full preventive baseline vs `+N-1 redispatch`.

- `configs/replay/replay_validation.yaml`  
  Replay validation settings and run list.

- `scripts/run_formulations.sh`  
  Runs the main formulation comparison suite.

- `scripts/run_security_checks.sh`  
  Runs the full preventive security-check suite.

- `scripts/run_redispatch_sensitivity.sh`  
  Runs optional redispatch sensitivity suite.

- `scripts/run_replay_validation.sh`  
  Runs ANDES replay validation for selected surrogate-embedded runs.

- `scripts/run_all.sh`  
  Runs all optimization-side experiments end-to-end.

## Exact Commands

Run from repository root.

1. Main formulation comparison (Chapter 5 optimization baseline table):
```bash
thesis_optimization_results/scripts/run_formulations.sh
```

2. Security checks before replay (max base/post-contingency loadings, violation counts):
```bash
thesis_optimization_results/scripts/run_security_checks.sh
```

3. Optional redispatch sensitivity:
```bash
thesis_optimization_results/scripts/run_redispatch_sensitivity.sh
```

4. Replay validation (predicted vs replayed):
```bash
thesis_optimization_results/scripts/run_replay_validation.sh
```

5. Full pipeline:
```bash
thesis_optimization_results/scripts/run_all.sh
```

If needed, set interpreter explicitly:
```bash
PYTHON_BIN=../venv/bin/python thesis_optimization_results/scripts/run_all.sh
```

## Output Mapping to Thesis Needs

### 1) Formulations compared
Produced by:
- `configs/suites/formulation_comparison.yaml`

Main aggregate outputs:
- `results/formulation_comparison_summary.csv`
- `results/formulation_comparison_summary.md`
- `results/formulation_comparison_summary.json`

### 2) Exact optimization setup
Per-run metadata includes:
- objective and equation map references
- active constraint switches
- solver settings and effective kwargs
- model/surrogate path and mode
- scenario and system case

Per-run file:
- `results/formulations/<run_id>/<run_id>_summary.json`

### 3) Core optimization KPIs
Per-run summary JSON/CSV includes:
- status
- objective
- solve time
- problem size (variables, binaries, constraints, nnz)
- block-wise constraint counts

Comparison + `% cost increase vs ED`:
- `results/formulation_comparison_summary.csv`

### 4) Constraint impact analysis
Per-run export:
- `.../<run_id>_dispatch_impact.csv`

Contains:
- generator dispatch base/opt/delta
- IBR `M`, `D`
- `Delta P` dispatch and predicted values
- headroom up/down and margins

### 5) Security checks before replay
Per-run summary JSON `security_checks` section includes:
- base-case max loading / violations
- N-1 max loading / violations / outage statistics
- predicted-output limit satisfaction and violation count

### 6) Replay validation
Produced by:
- `configs/replay/replay_validation.yaml`

Outputs:
- `results/replay_validation_summary.csv`
- `results/replay_validation_detail.csv`
- `results/replay_validation_summary.json`

These contain predicted vs replayed metrics, absolute/relative errors, and replay-limit violations.

## Notes

- The thesis baseline remains preventive (`ED + Line + N-1 + Surrogate`).
- Redispatch is included only as optional sensitivity.
- Each suite stores resolved per-run configs under each run results directory for traceability.
