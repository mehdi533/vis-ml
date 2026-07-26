#!/usr/bin/env bash
# Reproduce the research-result analyses (RESULTS.md) from trained artifacts.
#
# Prereq: the surrogates/datasets under results/ must exist. Produce them with:
#   PYTHONPATH=. <py> -m data_generation.run_sims  --config configs/data_generation/conformal.yaml
#   PYTHONPATH=. <py> -m models.train_sweep         --config configs/model/conformal_opt_ready.yaml
#   PYTHONPATH=. <py> -m models.train_sweep         --config configs/model/pareto_sweep.yaml
#   PYTHONPATH=. <py> -m models.train_sweep         --config configs/model/pareto_convex.yaml
#
# Then: bash scripts/run_research_demos.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"; export PYTHONPATH="$ROOT"
if   [ -x "$ROOT/.venv/Scripts/python.exe" ]; then PY="$ROOT/.venv/Scripts/python.exe"
elif [ -x "$ROOT/.venv/bin/python" ];        then PY="$ROOT/.venv/bin/python"
else PY="python"; fi

echo "== Conformal safety margins (coverage before/after) =="
"$PY" scripts/run_conformal_demo.py --alpha 0.1 >/dev/null && "$PY" - <<'PY'
import json; d=json.load(open("results/conformal/conformal_demo.json"))
for m,v in d["metrics"].items():
    print(f"  {m}: before={v['coverage_before_mean']:.2f} after={v['coverage_after_mean']:.2f} margin={v['margin_mean']:.3f}")
PY

echo "== Accuracy vs embeddability Pareto =="
"$PY" scripts/run_pareto.py | "$PY" -c "import sys,json;[print(f\"  {r['model']:>7} ({r.get('family','?')}): RMSE={r['agg_rmse']}  box-binaries={r['binaries_schedulable_box']}\") for r in json.load(sys.stdin)['models']]"

echo "== Embeddability (full domain vs schedulable box) =="
"$PY" scripts/analyze_embeddability.py | "$PY" -c "import sys,json; d=json.load(sys.stdin); print(f\"  full={d['full_domain']['binaries_needed']} box={d['schedulable_box']['binaries_needed']} reduction={d.get('binary_reduction_pct')}%\")"

echo "== DONE: research analyses reproduced =="
