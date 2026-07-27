#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time environment setup on the cluster (login node).
# Creates a Python 3.10-3.12 virtual env at <repo>/.venv and installs deps.
# Run from the repository root:  bash cluster/setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== Repo: $REPO_ROOT =="

# --- 1. Get a Python 3.10-3.12 interpreter ---------------------------------
# EDIT for your cluster. Examples:
#   ETH Euler:   module load stack/2024-06 python/3.11.6
#   EPFL SCITAS: module load gcc python/3.10.4
# If a suitable `python3.11`/`python3.12` is already on PATH, the fallback below
# is used automatically.
if command -v module >/dev/null 2>&1; then
  module load stack/2024-06 python/3.11.6 2>/dev/null || \
  module load python/3.11 2>/dev/null || \
  echo "WARN: could not 'module load' Python; relying on PATH." >&2
fi

PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")
    case "$v" in 3.10|3.11|3.12) PY="$c"; break;; esac
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: need Python 3.10-3.12. Load a module or install one, then re-run." >&2
  exit 1
fi
echo "== Using $($PY --version) ($PY) =="

# --- 2. Virtual environment + dependencies ---------------------------------
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-core.txt

# --- 3. Verify the install (fail fast if anything is missing) --------------
python - <<'PYCHECK'
import importlib, sys
mods = ["numpy","torch","pandas","sklearn","cvxpy","andes","pandapower","yaml"]
for m in mods:
    importlib.import_module(m)
import cvxpy as cp
solvers = cp.installed_solvers()
assert "SCIP" in solvers, f"SCIP not available; got {solvers}"
print("OK: all imports + SCIP present. python =", sys.version.split()[0])
PYCHECK

echo "== Setup complete. Activate later with: source .venv/bin/activate =="
