from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import joblib
import numpy as np
import torch

from models.models import MLP, MTLGroupedSharedHeads, MTLSharedHeads, SharedGroupSpec
from scheduling.convex_reformulation.utils import (
    _apply_relu_stack,
    _extract_linear_layers,
    compute_feature_bounds_from_training_data,
)


def _compute_x_bounds(cfg: dict):
    bounds_cfg = cfg.get("bounds", {})
    x_min_cfg = bounds_cfg.get("x_min", [])
    x_max_cfg = bounds_cfg.get("x_max", [])
    x_min = None
    x_max = None
    if x_min_cfg and x_max_cfg:
        x_bounds = np.vstack([np.asarray(x_min_cfg, dtype=float), np.asarray(x_max_cfg, dtype=float)])
        if bounds_cfg.get("use_scaler_for_bounds", True):
            x_scaler_path = cfg.get("scalers", {}).get("x_scaler_path")
            if x_scaler_path:
                x_scaler = joblib.load(x_scaler_path)
                x_bounds = x_scaler.transform(x_bounds)
        x_min = x_bounds[0]
        x_max = x_bounds[1]
    elif bounds_cfg.get("training_data"):
        x_min, x_max, feat_cols = compute_feature_bounds_from_training_data(cfg)
        x_features = cfg.get("features", {}).get("x_features")
        if x_features and feat_cols != x_features:
            name_to_pos = {name: i for i, name in enumerate(feat_cols)}
            reorder = [name_to_pos[name] for name in x_features]
            x_min = x_min[reorder]
            x_max = x_max[reorder]
    return x_min, x_max


def _compute_y_bounds(cfg: dict):
    bounds_cfg = cfg.get("bounds", {})
    y_min_raw = np.asarray(bounds_cfg.get("y_min", []), dtype=float).reshape(1, -1)
    y_max_raw = np.asarray(bounds_cfg.get("y_max", []), dtype=float).reshape(1, -1)
    y_scaler_path = cfg.get("scalers", {}).get("y_scaler_path")
    y_scaler = joblib.load(y_scaler_path) if y_scaler_path else None
    if y_scaler is not None and y_min_raw.size:
        y_min = y_scaler.transform(y_min_raw).reshape(-1)
        y_max = y_scaler.transform(y_max_raw).reshape(-1)
    else:
        y_min = y_min_raw.reshape(-1)
        y_max = y_max_raw.reshape(-1)
    return y_min, y_max


def build_epigraph_constraints(cfg: dict, *, apply_x_bounds: bool = True, apply_y_bounds: bool = True):
    model_cfg = cfg["model"]

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
                raise ValueError(f"Unsupported group config: {entry}")
        model = MTLGroupedSharedHeads(
            in_dim=int(model_cfg["in_dim"]),
            n_tasks=int(model_cfg["n_tasks"]),
            shared_sizes=model_cfg["shared_sizes"],
            head_sizes=model_cfg["head_sizes"],
            dropout=float(model_cfg.get("dropout", 0.0)),
            group_shared_configs=group_specs,
        )
    elif model_type == "MLP":
        model = MLP(
            in_dim=int(model_cfg["in_dim"]),
            out_dim=int(model_cfg.get("out_dim", model_cfg.get("n_tasks", 1))),
            hidden_sizes=model_cfg.get("hidden_sizes"),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x = cp.Variable(model_cfg["in_dim"], name="features")
    constraints = []

    if model_type == "MLP":
        layers = _extract_linear_layers(model.net)
        h = x
        h = _apply_relu_stack(h, layers[:-1], constraints, prefix="mlp")
        w_last, b_last = layers[-1]
        y = cp.Variable(w_last.shape[0], name="outputs")
        constraints.append(y == w_last @ h + b_last)
    else:
        shared_layers = _extract_linear_layers(model.shared)
        h = _apply_relu_stack(x, shared_layers, constraints, prefix="shared")
        outputs = []
        for i, head in enumerate(model.heads):
            head_layers = _extract_linear_layers(head)
            h_head = _apply_relu_stack(h, head_layers[:-1], constraints, prefix=f"head{i}")
            w_last, b_last = head_layers[-1]
            y_out = cp.Variable(1, name=f"out{i}")
            constraints.append(y_out == w_last @ h_head + b_last)
            outputs.append(y_out)
        y = cp.hstack(outputs)

    if apply_x_bounds:
        x_min, x_max = _compute_x_bounds(cfg)
        if x_min is not None and x_max is not None and x_min.size and x_max.size:
            constraints.append(x >= x_min)
            constraints.append(x <= x_max)

    if apply_y_bounds:
        y_min, y_max = _compute_y_bounds(cfg)
        if y_min.size and y_max.size:
            constraints.append(y >= y_min)
            constraints.append(y <= y_max)

    return x, y, constraints
