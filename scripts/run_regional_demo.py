#!/usr/bin/env python3
"""COI-only vs regional (worst-bus) security: does the regional constraint bite?

Runs the embedded-surrogate dispatch twice: (a) bounding only the COI metrics
(worst-bus free), (b) additionally bounding the worst-bus metrics. Reports each
schedule's predicted metrics and M/D, so we can see whether the COI-only optimum
would violate the per-bus limit that the regional constraint enforces.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/scheduling/regional/optimization_regional.yaml"
# y order: [rocof_COI, dev_COI, rocof_worst_bus, dev_worst_bus]
SCENARIOS = {
    "coi_only":  {"y_min": [-1.2, -2.0, -5.0, -5.0], "y_max": [1.2, 2.0, 5.0, 5.0]},
    "regional":  {"y_min": [-1.2, -2.0, -1.2, -1.5], "y_max": [1.2, 2.0, 1.2, 1.5]},
}


def main() -> None:
    import os
    base = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    out = {}
    for tag, bnds in SCENARIOS.items():
        cfg = json.loads(json.dumps(base))
        cfg["bounds"]["y_min"] = bnds["y_min"]
        cfg["bounds"]["y_max"] = bnds["y_max"]
        cfg["output"] = dict(cfg.get("output", {}))
        cfg["output"]["run_tag"] = f"regional_{tag}"
        cfg["output"]["results_dir"] = "results/regional/optimization"
        cfg["output"]["log_file"] = f"results/regional/optimization/regional_{tag}.log"
        cfg["output"]["fail_on_infeasible"] = False
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=str(ROOT), encoding="utf-8") as tf:
            yaml.safe_dump(cfg, tf, sort_keys=False)
            tmp = tf.name
        subprocess.run([sys.executable, "-m", "scheduling.problem", "--config", tmp],
                       cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)}, capture_output=True, text=True)
        Path(tmp).unlink(missing_ok=True)
        sp = ROOT / f"results/regional/optimization/regional_{tag}_summary.json"
        rec = {"status": "no_summary"}
        if sp.exists():
            s = json.loads(sp.read_text(encoding="utf-8"))
            pm = s.get("predicted_metrics", {})
            rec = {
                "status": s.get("status"),
                "predicted": {n: round(float(v), 4) for n, v in zip(pm.get("names", []), pm.get("values", []))} if s.get("status") == "optimal" else None,
                "m_opt": [round(x, 2) for x in s.get("dispatch_summary", {}).get("m_opt", [])] if s.get("status") == "optimal" else None,
            }
        out[tag] = rec
        print(tag, json.dumps(rec))
    (ROOT / "results/regional/regional_demo.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()
