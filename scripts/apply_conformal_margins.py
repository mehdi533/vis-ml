#!/usr/bin/env python3
"""Turn calibrated conformal margins into a security-tightened optimization config.

Reads a surrogate's held-out test predictions (``<metric>__true`` / ``__pred``),
calibrates a per-metric split-conformal margin, tightens the optimization
config's dynamic-security envelope (``bounds.y_min`` / ``bounds.y_max``) by those
margins, and writes a ready-to-run config. This is the automated "apply" step of
the conformal loop: optimise -> calibrate -> **tighten** -> re-optimise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.conformal.apply import build_tightened_bounds  # noqa: E402
from research.conformal.calibration import ConformalMargins  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True, help="Surrogate test_predictions.csv")
    ap.add_argument("--config", required=True, help="Optimization config to tighten")
    ap.add_argument("--out", required=True, help="Where to write the tightened config")
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.pred_csv if not Path(args.pred_csv).is_absolute() else args.pred_csv)
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    y_names = list(cfg["outputs"]["y_names"])
    y_min = list(map(float, cfg["bounds"]["y_min"]))
    y_max = list(map(float, cfg["bounds"]["y_max"]))

    # Calibrate an abs-mode margin for every y_name that has residuals available.
    cols = {}
    for name in y_names:
        tc, pc = f"{name}__true", f"{name}__pred"
        if tc in df.columns and pc in df.columns:
            cols[name] = ("abs", df[pc].to_numpy(float), df[tc].to_numpy(float))
    if not cols:
        raise SystemExit("No matching <metric>__true/__pred columns for the config's y_names.")

    cm = ConformalMargins(alpha=args.alpha).fit(cols)
    tightened = build_tightened_bounds(cm, y_names, y_min, y_max)

    cfg["bounds"]["y_min"] = tightened["y_min"]
    cfg["bounds"]["y_max"] = tightened["y_max"]
    cfg.setdefault("formulation", {})["conformal_alpha"] = args.alpha
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    print(f"tightened {len(cols)} of {len(y_names)} bounds at alpha={args.alpha} -> {out.name}")
    for i, name in enumerate(y_names):
        m = cm.margins.get(name)
        if m is not None:
            print(f"  {name:16s} [{y_min[i]:+.3f},{y_max[i]:+.3f}] -> "
                  f"[{tightened['y_min'][i]:+.3f},{tightened['y_max'][i]:+.3f}]  (margin {m:.3f})")


if __name__ == "__main__":
    main()
