#!/usr/bin/env bash
# End-to-end smoke of the full VIS-ML pipeline on a tiny dataset, using the free
# SCIP solver (no Gurobi license needed). Run from the repo root:
#
#   bash scripts/run_smoke.sh
#
# It exercises all four links: data generation -> model training ->
# optimization-ready training -> MILP scheduling. Intended for verifying the
# environment and catching regressions, NOT for producing thesis results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# Prefer the project venv if present, else fall back to `python`.
if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
  PY="$ROOT/.venv/Scripts/python.exe"      # Windows venv layout
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"              # POSIX venv layout
else
  PY="python"
fi

echo "== [1/4] Data generation (80 sims) =="
"$PY" -m data_generation.run_sims --config configs/data_generation/smoke.yaml

echo "== [2/4] Train MTLSH surrogate =="
"$PY" -m models.train_sweep --config configs/model/smoke_train.yaml

echo "== [3/4] Train optimization-ready MTLSH (178-feature contract) =="
"$PY" -m models.train_sweep --config configs/model/smoke_optimization_ready.yaml

echo "== [4/4] Solve MILP schedule with SCIP =="
"$PY" -m scheduling.problem --config configs/scheduling/smoke/optimization_smoke.yaml

echo "== SMOKE OK =="
