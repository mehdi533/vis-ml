from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import joblib
import numpy as np
import torch
import yaml

from models.models import MLP, MTLGroupedSharedHeads, MTLSharedHeads, SharedGroupSpec


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _relu_epigraph(z, y):
    return [y >= 0, y >= z]


def _apply_relu_stack(h, layers, constraints, *, prefix: str):
    for idx, (w, b) in enumerate(layers):
        z = w @ h + b
        y = cp.Variable(b.shape[0], name=f"{prefix}_{idx}")
        constraints += _relu_epigraph(z, y)
        h = y
    return h


def _extract_linear_layers(seq):
    return [
        (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
        for m in _linear_layers(seq)
    ]


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
    bounds_cfg = cfg.get("bounds", {})

    state_path = Path(model_cfg["state_dict"])
    if state_path.is_dir():
        state_path = state_path / "vis_mlp_state_dict.pt"

    model_type = str(model_cfg.get("type", "MTLSharedHeads"))

    if model_type == "MTLSharedHeads":
        model = MTLSharedHeads(
            in_dim=int(model_cfg["in_dim"]),
            n_tasks=int(model_cfg["n_tasks"]),
            shared_sizes=model_cfg["shared_sizes"],
            head_sizes=model_cfg["head_sizes"],
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    elif model_type == "MTLGroupedSharedHeads":
        raw_groups = model_cfg.get("group_shared_configs") or []
        group_specs = []
        for entry in raw_groups:
            if isinstance(entry, SharedGroupSpec):
                group_specs.append(entry)
            elif isinstance(entry, dict):
                group_specs.append(
                    SharedGroupSpec(
                        head_indices=entry.get("head_indices", []),
                        hidden_sizes=entry.get("hidden_sizes", []),
                    )
                )
            else:
                raise ValueError("group_shared_configs must be a list of dicts or SharedGroupSpec.")
        model = MTLGroupedSharedHeads(
            in_dim=int(model_cfg["in_dim"]),
            n_tasks=int(model_cfg["n_tasks"]),
            shared_sizes=model_cfg.get("shared_sizes"),
            head_sizes=model_cfg.get("head_sizes"),
            dropout=float(model_cfg.get("dropout", 0.0)),
            group_shared_configs=group_specs,
        )
    elif model_type == "MLP":
        model = MLP(
            in_dim=int(model_cfg["in_dim"]),
            out_dim=int(model_cfg["out_dim"]),
            hidden_sizes=model_cfg.get("hidden_sizes"),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    else:
        raise NotImplementedError(
            f"Model type '{model_type}' is not supported for convex constraints."
        )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    in_dim = int(model_cfg["in_dim"])
    x = cp.Variable(in_dim, name="features")
    constraints = []

    if model_type == "MLP":
        layers = _extract_linear_layers(model.net)
        h = x
        for idx, (w, b) in enumerate(layers):
            z = w @ h + b
            if idx < len(layers) - 1:
                y_h = cp.Variable(b.shape[0], name=f"mlp_{idx}")
                constraints += _relu_epigraph(z, y_h)
                h = y_h
            else:
                y = cp.Variable(b.shape[0], name="mlp_out")
                constraints.append(y == z)
    else:
        shared_layers = _extract_linear_layers(model.shared)
        h = _apply_relu_stack(x, shared_layers, constraints, prefix="shared")

        outputs = []
        if model_type == "MTLGroupedSharedHeads":
            group_blocks = list(model.group_blocks)
            group_indices = list(model.group_block_indices)
            group_layers = [_extract_linear_layers(block) for block in group_blocks]

            for head_idx, head in enumerate(model.heads):
                h_head = h
                for layers, indices in zip(group_layers, group_indices):
                    if head_idx in indices:
                        h_head = _apply_relu_stack(h_head, layers, constraints, prefix=f"group{head_idx}")

                head_layers = _extract_linear_layers(head)
                for idx, (w, b) in enumerate(head_layers):
                    z = w @ h_head + b
                    if idx < len(head_layers) - 1:
                        y_h = cp.Variable(b.shape[0], name=f"head{head_idx}_{idx}")
                        constraints += _relu_epigraph(z, y_h)
                        h_head = y_h
                    else:
                        y_out = cp.Variable(1, name=f"out{head_idx}")
                        constraints.append(y_out == z)
                        outputs.append(y_out)
        else:
            head_layers = [
                _extract_linear_layers(head)
                for head in model.heads
            ]
            for i, layers in enumerate(head_layers):
                h_head = h
                for idx, (w, b) in enumerate(layers):
                    z = w @ h_head + b
                    if idx < len(layers) - 1:
                        y_h = cp.Variable(b.shape[0], name=f"head{i}_{idx}")
                        constraints += _relu_epigraph(z, y_h)
                        h_head = y_h
                    else:
                        y_out = cp.Variable(1, name=f"out{i}")
                        constraints.append(y_out == z)
                        outputs.append(y_out)

        y = cp.hstack(outputs)

    if "y_min" in bounds_cfg and "y_max" in bounds_cfg:
        use_scaler = bool(bounds_cfg.get("use_scaler_for_bounds", True))
        y_scaler_path = cfg.get("scalers", {}).get("y_scaler_path")
        scaler = joblib.load(y_scaler_path) if (use_scaler and y_scaler_path) else None

        y_min = np.asarray(bounds_cfg["y_min"], dtype=float).reshape(1, -1)
        y_max = np.asarray(bounds_cfg["y_max"], dtype=float).reshape(1, -1)

        if scaler is not None:
            y_min = scaler.transform(y_min)
            y_max = scaler.transform(y_max)

        y_min = y_min.reshape(-1)
        y_max = y_max.reshape(-1)

        constraints.append(y >= y_min)
        constraints.append(y <= y_max)

    return x, y, constraints


def main():
    parser = argparse.ArgumentParser(description="Convex ReLU relaxation builder.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    x, y, constraints = build_mtlsh_convex_constraints(cfg)

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"constraints: {len(constraints)}")


if __name__ == "__main__":
    main()
