#!/usr/bin/env python3
"""N-1 feasibility-boundary stress study on IEEE 118.

Sweeps the base load level and records whether the full preventive-N-1 VIS
dispatch stays feasible, its cost, and solve time -- mapping the security-cost
cliff. Single-process (safe on a laptop). Emits JSON.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/scheduling/ieee118/optimization_ieee118.yaml"
LEVELS = [0.6, 0.8, 1.0, 1.15]


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    rows = []
    import os
    for lvl in LEVELS:
        cfg = json.loads(json.dumps(base))
        cfg["scenario"]["base_scale"] = lvl
        cfg["output"] = dict(cfg.get("output", {}))
        cfg["output"]["run_tag"] = f"stress_b{lvl}"
        cfg["output"]["results_dir"] = "results/ieee118/stress"
        cfg["output"]["log_file"] = f"results/ieee118/stress/stress_b{lvl}.log"
        cfg["output"]["fail_on_infeasible"] = False
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=str(ROOT), encoding="utf-8") as tf:
            yaml.safe_dump(cfg, tf, sort_keys=False)
            tmp = tf.name
        subprocess.run([sys.executable, "-m", "scheduling.problem", "--config", tmp],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)},
                       capture_output=True, text=True)
        Path(tmp).unlink(missing_ok=True)
        sp = ROOT / f"results/ieee118/stress/stress_b{lvl}_summary.json"
        if sp.exists():
            s = json.loads(sp.read_text(encoding="utf-8"))
            rows.append({
                "base_scale": lvl, "status": s.get("status"),
                "objective": None if s.get("objective") is None else round(float(s["objective"]), 1),
                "solve_time_sec": round(float(s.get("solver_stats", {}).get("solve_time_sec", 0)), 1),
                "n1_active": s.get("n1_stats", {}).get("n_active_outages"),
            })
        else:
            rows.append({"base_scale": lvl, "status": "no_summary"})
        print(json.dumps(rows[-1]))
    (ROOT / "results/ieee118/stress_study.json").write_text(
        json.dumps({"study": "ieee118_n1_feasibility_vs_load", "runs": rows}, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
