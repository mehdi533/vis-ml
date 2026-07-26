#!/usr/bin/env python3
"""Conformal robust-margin demonstration on real surrogate residuals.

Uses a trained surrogate's held-out test predictions (predicted vs true simulated
security metric) as the calibration data. For each frequency-security metric it:
  1. calibrates a split-conformal one-sided margin at level alpha, and
  2. measures the *safety coverage* -- the fraction of cases where the true
     (simulated) magnitude stays within predicted + margin -- BEFORE (margin=0,
     i.e. trust the raw surrogate) vs AFTER (with the conformal margin).

Averaged over many random calibration/validation splits for a stable estimate.
This is the mechanism behind closing the thesis Ch. 6.2 replay gap: the raw
surrogate is safe only ~half the time near the limit; the conformal margin lifts
coverage to >= 1 - alpha with a finite-sample guarantee.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.conformal.calibration import conformal_margin, empirical_coverage  # noqa: E402

SECURITY_METRICS = [
    "rocof_COI", "dev_COI",
    "Delta_P_IBR_1", "Delta_P_IBR_2", "Delta_P_IBR_3", "Delta_P_IBR_4",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", default="results/conformal/models/MTLSH__kendall__minmax__seed42/test_predictions.csv")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--splits", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.pred_csv)
    rng = np.random.default_rng(0)
    out = {"alpha": args.alpha, "n_samples": int(len(df)), "splits": args.splits, "metrics": {}}

    for metric in SECURITY_METRICS:
        tc, pc = f"{metric}__true", f"{metric}__pred"
        if tc not in df.columns or pc not in df.columns:
            continue
        true = df[tc].to_numpy(float)
        pred = df[pc].to_numpy(float)
        n = len(true)
        half = n // 2

        margins, cov_before, cov_after = [], [], []
        for _ in range(args.splits):
            perm = rng.permutation(n)
            cal, val = perm[:half], perm[half:]
            m = conformal_margin(pred[cal], true[cal], alpha=args.alpha, mode="abs")
            if not np.isfinite(m):
                continue
            margins.append(m)
            cov_before.append(empirical_coverage(pred[val], true[val], 0.0, mode="abs"))
            cov_after.append(empirical_coverage(pred[val], true[val], m, mode="abs"))

        out["metrics"][metric] = {
            "margin_mean": round(float(np.mean(margins)), 5),
            "coverage_before_mean": round(float(np.mean(cov_before)), 4),
            "coverage_after_mean": round(float(np.mean(cov_after)), 4),
            "target_coverage": round(1 - args.alpha, 4),
            "n_cal": half,
            "n_val": n - half,
        }

    print(json.dumps(out, indent=2))
    outfile = ROOT / "results/conformal/conformal_demo.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
