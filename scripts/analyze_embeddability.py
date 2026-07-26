#!/usr/bin/env python3
"""Measure MILP embeddability of a trained MTLSH surrogate via interval bounds.

Runs the research/embeddability IBP on a real trained model and compares the
number of ReLU binaries needed over (a) the full scaled input domain vs (b) the
actual schedulable box where only M/D inputs vary -- demonstrating the payoff of
bound tightening. Emits JSON to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.models import create_model  # noqa: E402
from research.embeddability import (  # noqa: E402
    LinearLayer,
    propagate_interval_bounds,
    relu_stability,
)


def _linears(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _stability_over_box(model, feature_cols, x_lo, x_hi):
    shared_lin = _linears(model.shared)
    total = binaries = 0
    max_bigM = 0.0
    for head in model.heads:
        layers = shared_lin + _linears(head)
        L = [LinearLayer(l.weight.detach().numpy(), l.bias.detach().numpy()) for l in layers]
        pre = propagate_interval_bounds(L, x_lo, x_hi)
        for s in relu_stability(pre):
            total += s.n_total
            binaries += s.n_unstable
            max_bigM = max(max_bigM, s.max_abs_bigM)
    return {
        "hidden_relu_neurons": total,
        "binaries_needed": binaries,
        "stable_fixed": total - binaries,
        "binary_fraction": round(binaries / total, 4) if total else 0.0,
        "max_abs_bigM": round(max_bigM, 3),
    }


def main() -> None:
    model_dir = ROOT / "results/smoke/models_opt_ready/MTLSH__kendall__minmax__seed42"
    run_cfg = yaml.safe_load((model_dir / "run_config.yaml").read_text(encoding="utf-8"))
    feature_cols = list(run_cfg["resolved"]["feature_cols"])
    n = len(feature_cols)

    model, _ = create_model("MTLSH", in_dim=n, out_dim=6, shared_sizes=[32], head_sizes=[16])
    model.load_state_dict(torch.load(model_dir / "mtlsh_state_dict_best.pt", map_location="cpu"))
    model.eval()

    # Minmax scaler maps the training domain to [0, 1].
    full_lo, full_hi = np.zeros(n), np.ones(n)

    # Schedulable box: only M/D inputs vary; everything else fixed at its midpoint.
    sched_names = [c for c in feature_cols if c in ("M_agg", "D_agg") or c[:2] in ("M_", "D_")]
    sched_idx = [feature_cols.index(c) for c in sched_names]
    box_lo, box_hi = np.full(n, 0.5), np.full(n, 0.5)
    for i in sched_idx:
        box_lo[i], box_hi[i] = 0.0, 1.0

    result = {
        "model": "MTLSH (smoke opt-ready, shared=[32], heads=[16], 6 tasks)",
        "n_input_features": n,
        "n_schedulable_inputs": len(sched_idx),
        "schedulable_inputs": sched_names,
        "full_domain": _stability_over_box(model, feature_cols, full_lo, full_hi),
        "schedulable_box": _stability_over_box(model, feature_cols, box_lo, box_hi),
    }
    fd, sb = result["full_domain"], result["schedulable_box"]
    if fd["binaries_needed"]:
        result["binary_reduction_pct"] = round(
            100.0 * (fd["binaries_needed"] - sb["binaries_needed"]) / fd["binaries_needed"], 1
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
