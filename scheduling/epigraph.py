from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import torch

from scheduling.utils import (
    _apply_relu_stack,
    _extract_linear_layers,
    build_torch_model,
    compute_x_bounds,
    compute_y_bounds,
)


def build_epigraph_constraints(cfg: dict, *, apply_x_bounds: bool = True, apply_y_bounds: bool = True):
    model_cfg = cfg["model"]

    state_path = Path(model_cfg["state_dict"])
    if state_path.is_dir():
        state_path = state_path / "vis_mlp_state_dict.pt"

    model_type = str(model_cfg.get("type", "MTLSharedHeads"))
    model = build_torch_model(model_cfg)

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
        x_min, x_max = compute_x_bounds(cfg)
        if x_min is not None and x_max is not None and x_min.size and x_max.size:
            constraints.append(x >= x_min)
            constraints.append(x <= x_max)

    if apply_y_bounds:
        y_min, y_max = compute_y_bounds(cfg)
        if y_min.size and y_max.size:
            constraints.append(y >= y_min)
            constraints.append(y <= y_max)

    return x, y, constraints
