#!/usr/bin/env python3
"""Augment a dataset with regional (worst-bus) frequency-security targets.

The COI center-of-inertia metrics hide local severity: the worst individual bus
can see a far larger RoCoF than the COI value. This script adds signed
worst-bus targets (max |per-bus metric|, signed like the COI metric) so a
multi-head surrogate can predict the regional worst case, not just the average.

Usage:
    python scripts/build_regional_dataset.py --in <csv> --out <csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/conformal/data/simulation_results.csv")
    ap.add_argument("--out", dest="out", default="results/regional/data/simulation_results.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    br = [c for c in df.columns if c.startswith("bus_rocof_max_abs_") and c.split("_")[-1].isdigit()]
    bf = [c for c in df.columns if c.startswith("bus_freq_max_abs_dev_") and c.split("_")[-1].isdigit()]
    if not br or not bf:
        raise SystemExit("No per-bus RoCoF / frequency-deviation columns found in input.")

    df["rocof_worst_bus"] = df[br].abs().max(axis=1) * np.sign(df["rocof_COI"].replace(0, 1))
    df["dev_worst_bus"] = df[bf].abs().max(axis=1) * np.sign(df["dev_COI"].replace(0, 1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    r = (df["rocof_worst_bus"].abs() / df["rocof_COI"].abs().replace(0, np.nan)).dropna()
    print(f"wrote {out} ({df.shape[0]} rows); worst-bus/COI RoCoF ratio "
          f"mean={r.mean():.2f}x max={r.max():.2f}x over {len(br)} buses")


if __name__ == "__main__":
    main()
