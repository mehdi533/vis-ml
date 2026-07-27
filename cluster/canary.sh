#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CANARY: generate a small dataset and assert the physics is correct BEFORE
# committing a node-day to the 120k-sim campaign. Exits non-zero on any failure,
# so submit_all.sh gates the big jobs on this passing.
#   bash cluster/canary.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$REPO_ROOT"

echo "== [canary 1/2] generating 300-sim validation dataset =="
python -m data_generation.run_sims --config configs/data_generation/canary.yaml

echo "== [canary 2/2] asserting physical correctness (50 Hz) =="
# validate_andes_physics.py exits 1 if any check fails:
#  success rate, no non-finite metrics, load-increase -> freq drop,
#  disturbance increases |RoCoF|, inertia reduces |RoCoF|.
python scripts/validate_andes_physics.py --csv results/canary/data/simulation_results.csv

echo "== CANARY PASSED: simulations are physically correct; safe to launch the full campaign. =="
