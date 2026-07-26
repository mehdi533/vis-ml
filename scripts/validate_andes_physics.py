#!/usr/bin/env python3
"""Physical sanity checks on the ANDES-generated data (the surrogate's ground truth).

Validates, from the *actual* generated dataset (the trusted pipeline output rather
than hand-scripted TDS), that the simulator obeys the physics the surrogate must
learn, on the European 50 Hz base:
  (1) simulator health: high success rate, no non-finite security metrics;
  (2) a load *increase* drives frequency *down* (rocof_COI < 0, dev_COI < 0);
  (3) disturbance magnitude increases |RoCoF| (dominant driver);
  (4) higher aggregate inertia reduces |RoCoF| (negative coefficient controlling
      for disturbance) -- the core physics of virtual inertia.

A causal fixed-inertia sweep through the pipeline is documented in RESULTS.md
(M=1 -> |RoCoF| 0.328, M=7 -> 0.227, i.e. 1.45x on the modified IEEE 39-bus).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/conformal/data/simulation_results.csv")
    args = ap.parse_args()
    df = pd.read_csv(ROOT / args.csv if not Path(args.csv).is_absolute() else args.csv)
    n = len(df)
    dcol = "DELTA_PQ_tot" if "DELTA_PQ_tot" in df else "load_step_scale"

    up = df[df["load_step_scale"] > 1.0]
    r_m = float(np.corrcoef(df["M_agg"], df["rocof_COI"].abs())[0, 1])
    r_d = float(np.corrcoef(df[dcol].abs(), df["rocof_COI"].abs())[0, 1])
    X = np.column_stack([np.ones(n), df["M_agg"], df[dcol].abs()])
    beta, *_ = np.linalg.lstsq(X, df["rocof_COI"].abs().values, rcond=None)

    checks = {
        "success_rate_ok": bool(df["success"].mean() >= 0.95),
        "no_nonfinite_metrics": bool(np.isfinite(df[["rocof_COI", "dev_COI"]].to_numpy()).all()),
        "load_increase_drops_frequency": bool((up["rocof_COI"] < 0).mean() > 0.98
                                              and (up["dev_COI"] < 0).mean() > 0.98),
        "disturbance_increases_rocof": bool(r_d > 0.3),
        "inertia_reduces_rocof": bool(r_m < 0 and beta[1] < 0),
    }
    report = {
        "n": n,
        "success_rate": round(float(df["success"].mean()), 4),
        "corr_M_vs_absRocof": round(r_m, 3),
        "corr_dist_vs_absRocof": round(r_d, 3),
        "regression_M_coef": round(float(beta[1]), 5),
        "regression_dist_coef": round(float(beta[2]), 5),
        "checks": checks,
    }
    print(json.dumps(report, indent=2))
    (ROOT / "results/andes_physics_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    ok = all(checks.values())
    print("ALL PHYSICS CHECKS PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
