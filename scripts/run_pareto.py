#!/usr/bin/env python3
"""Embeddability-vs-accuracy Pareto across architectures.

For each trained model in a sweep directory, pairs its aggregate accuracy
(RMSE) with its embedding cost (hidden ReLU neurons = max binaries over the full
input domain, and binaries over the schedulable box). Emits JSON for the
dashboard. Demonstrates the thesis's finding quantitatively: the best-accuracy
model is not necessarily the largest embedding.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.models import create_model  # noqa: E402
from research.embeddability import LinearLayer, propagate_interval_bounds, relu_stability  # noqa: E402

SWEEP_DIRS = [ROOT / "results/pareto/models", ROOT / "results/pareto/convex"]
CONVEX_TYPES = {"FICNN", "PICNN", "PICNN_MTLSH"}


def _linears(seq):
    return [m for m in seq if isinstance(m, torch.nn.Linear)]


def _path_layers(model):
    """One input->output Linear path for the model (representative for embedding)."""
    if hasattr(model, "net"):                       # MLP
        lins = _linears(model.net)
    elif hasattr(model, "shared") and hasattr(model, "heads"):
        lins = _linears(model.shared)
        if getattr(model, "group_blocks", None) and len(model.group_blocks):
            lins += _linears(model.group_blocks[0])
        lins += _linears(model.heads[0])
    else:
        lins = _linears(model.modules())
    return [LinearLayer(l.weight.detach().numpy(), l.bias.detach().numpy()) for l in lins]


def main() -> None:
    rows = []
    run_cfgs = []
    for sd in SWEEP_DIRS:
        run_cfgs += sorted(glob.glob(str(sd / "**/run_config.yaml"), recursive=True))
    for run_cfg in run_cfgs:
        d = Path(run_cfg).parent
        rc = yaml.safe_load(Path(run_cfg).read_text(encoding="utf-8"))
        fc = rc["resolved"]["feature_cols"]
        n = len(fc)
        mtype = str(rc.get("model", {}).get("type") or d.name.split("__")[0])
        metrics = json.loads((d / "metrics_summary.json").read_text(encoding="utf-8"))
        agg_rmse = float(metrics.get("agg_rmse_mean") or metrics.get("rmse_mean") or np.nan)

        # Convex families (ICNN) embed via convex constraints -> no ReLU binaries.
        if mtype in CONVEX_TYPES:
            rows.append({
                "model": mtype, "family": "convex", "agg_rmse": round(agg_rmse, 4),
                "hidden_relu_neurons": 0, "binaries_full_domain": 0, "binaries_schedulable_box": 0,
            })
            continue

        model, _ = create_model(
            mtype, in_dim=n, out_dim=6,
            hidden_sizes=[64, 32], shared_sizes=[32], head_sizes=[16],
            group_shared_sizes=[32, 16],
        )
        ckpt = glob.glob(str(d / "*state_dict_best.pt")) or glob.glob(str(d / "*state_dict.pt"))
        model.load_state_dict(torch.load(ckpt[0], map_location="cpu"))
        model.eval()
        L = _path_layers(model)

        full = relu_stability(propagate_interval_bounds(L, np.zeros(n), np.ones(n)))
        sched = [c for c in fc if c in ("M_agg", "D_agg") or c[:2] in ("M_", "D_")]
        si = [fc.index(c) for c in sched]
        lo, hi = np.full(n, 0.5), np.full(n, 0.5)
        for i in si:
            lo[i], hi[i] = 0.0, 1.0
        box = relu_stability(propagate_interval_bounds(L, lo, hi))

        rows.append({
            "model": mtype,
            "family": "relu",
            "agg_rmse": round(agg_rmse, 4),
            "n_parameters": int(metrics.get("n_parameters_total", 0) or 0),
            "hidden_relu_neurons": sum(s.n_total for s in full),
            "binaries_full_domain": sum(s.n_unstable for s in full),
            "binaries_schedulable_box": sum(s.n_unstable for s in box),
        })

    rows.sort(key=lambda r: r["agg_rmse"])
    out = {"study": "embeddability_vs_accuracy", "dataset": "IEEE 39, 250 sims", "models": rows}
    print(json.dumps(out, indent=2))
    (ROOT / "results/pareto/pareto.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
