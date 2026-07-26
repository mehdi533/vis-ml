#!/usr/bin/env python3
"""Measure embedded-MILP solve time vs. binary count as the M/D box widens.

Wider schedulable bounds -> more ReLU neurons are 'unstable' over the box ->
more binaries in the exact encoding -> longer solve. Sweeps the M/D bound width,
runs the (NN-only) scheduling MILP with SCIP at each, and records the binary
count and solve time reported in each run summary. Emits JSON.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/scheduling/conformal/solvetime_base.yaml"
WIDTHS = [1.0, 2.0, 4.0, 6.0, 8.0]


def main() -> None:
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    results = []
    for w in WIDTHS:
        cfg = json.loads(json.dumps(base))  # deep copy
        cfg["bounds"]["M_bounds"] = [0.0, w]
        cfg["bounds"]["D_bounds"] = [0.0, 0.75 * w]
        cfg["output"]["run_tag"] = f"st_w{w}"
        cfg["output"]["results_dir"] = "results/conformal/solvetime"
        cfg["output"]["log_file"] = f"results/conformal/solvetime/st_w{w}.log"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=str(ROOT), encoding="utf-8") as tf:
            yaml.safe_dump(cfg, tf, sort_keys=False)
            tmp = tf.name
        env = {"PYTHONPATH": str(ROOT)}
        import os
        env = {**os.environ, **env}
        subprocess.run([sys.executable, "-m", "scheduling.problem", "--config", tmp],
                       cwd=str(ROOT), env=env, capture_output=True, text=True)
        Path(tmp).unlink(missing_ok=True)
        spath = ROOT / f"results/conformal/solvetime/st_w{w}_summary.json"
        if not spath.exists():
            results.append({"m_width": w, "error": "no summary"}); continue
        s = json.loads(spath.read_text(encoding="utf-8"))
        ps = s.get("problem_size", {}); st = s.get("solver_stats", {})
        results.append({
            "m_width": w,
            "status": s.get("status"),
            "n_binary": ps.get("n_variables_binary"),
            "n_constraints": ps.get("n_constraints_total"),
            "solve_time_sec": round(float(st.get("solve_time_sec", 0.0)), 4),
        })
    out = {"experiment": "solve_time_vs_binaries", "model": "conformal MTLSH (IEEE 39, NN-only, SCIP)", "runs": results}
    print(json.dumps(out, indent=2))
    (ROOT / "results/conformal/solvetime_experiment.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
