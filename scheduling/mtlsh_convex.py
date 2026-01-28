from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import cvxpy as cp
import joblib
import numpy as np
import torch
import yaml

from models.models import MTLSharedHeads


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _relu_epigraph(z, y):
    return [y >= 0, y >= z]

def _load_csv_features(
    csv_path: Path,
    *,
    feature_cols: Sequence[str] | None = None,
    drop_cols: Sequence[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    drops = set(drop_cols or [])

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in drops]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in {csv_path}: {missing}")

    X = df[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    return X, list(feature_cols)


def compute_feature_bounds_from_training_data(cfg: dict):
    bounds_cfg = cfg.get("bounds", {})
    training_data = bounds_cfg.get("training_data")
    if not training_data:
        raise ValueError("bounds.training_data is required to compute bounds from data.")

    feature_cols = cfg.get("features", {}).get("x_features")
    drop_cols = bounds_cfg.get("drop_cols", [])

    X, feature_cols = _load_csv_features(
        Path(training_data),
        feature_cols=feature_cols if feature_cols else None,
        drop_cols=drop_cols,
    )

    use_scaler = bool(bounds_cfg.get("use_scaler_for_bounds", True))
    x_scaler_path = cfg.get("scalers", {}).get("x_scaler_path")
    if use_scaler and x_scaler_path:
        scaler = joblib.load(x_scaler_path)
        X = scaler.transform(X)

    x_min = np.nanmin(X, axis=0)
    x_max = np.nanmax(X, axis=0)
    return x_min, x_max, feature_cols


def build_mtlsh_convex_constraints(cfg: dict):
    model_cfg = cfg["model"]
    bounds_cfg = cfg["bounds"]

    state_path = Path(model_cfg["state_dict"])
    if state_path.is_dir():
        state_path = state_path / "vis_mlp_state_dict.pt"

    model = MTLSharedHeads(
        in_dim=int(model_cfg["in_dim"]),
        n_tasks=int(model_cfg["n_tasks"]),
        shared_sizes=model_cfg["shared_sizes"],
        head_sizes=model_cfg["head_sizes"],
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    shared_layers = [
        (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
        for m in _linear_layers(model.shared)
    ]
    head_layers = [
        [
            (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
            for m in _linear_layers(head)
        ]
        for head in model.heads
    ]

    in_dim = int(model_cfg["in_dim"])
    x = cp.Variable(in_dim)
    constraints = []

    h = x
    for w, b in shared_layers:
        z = w @ h + b
        y = cp.Variable(b.shape[0])
        constraints += _relu_epigraph(z, y)
        h = y

    outputs = []
    for layers in head_layers:
        h_head = h
        for idx, (w, b) in enumerate(layers):
            z = w @ h_head + b
            if idx < len(layers) - 1:
                y = cp.Variable(b.shape[0])
                constraints += _relu_epigraph(z, y)
                h_head = y
            else:
                y_out = cp.Variable(1)
                constraints.append(y_out == z)
                outputs.append(y_out)

    y = cp.hstack(outputs)
    if "y_min" in bounds_cfg:
        constraints.append(y >= np.asarray(bounds_cfg["y_min"], dtype=float))
    if "y_max" in bounds_cfg:
        constraints.append(y <= np.asarray(bounds_cfg["y_max"], dtype=float))

    return x, y, constraints


def main():
    parser = argparse.ArgumentParser(description="MTLSH convex ReLU relaxation builder.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    x, y, constraints = build_mtlsh_convex_constraints(cfg)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"constraints: {len(constraints)}")


if __name__ == "__main__":
    main()
