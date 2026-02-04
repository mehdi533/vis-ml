from __future__ import annotations

import cvxpy as cp
import numpy as np

from scheduling.fixed_pattern import (
    build_fixed_pattern_constraints_mlp,
    build_fixed_pattern_constraints_mtlshared,
)
from scheduling.milp import (
    build_milp_constraints_mlp,
    build_milp_constraints_mtlshared,
)
from scheduling.utils import _apply_relu_stack, _extract_linear_layers


def _build_epigraph_constraints_mlp(model, x):
    constraints = []
    layers = _extract_linear_layers(model.net)
    h = x
    h = _apply_relu_stack(h, layers[:-1], constraints, prefix="mlp")
    w_last, b_last = layers[-1]
    y = cp.Variable(w_last.shape[0], name="outputs")
    constraints.append(y == w_last @ h + b_last)
    return y, constraints


def _build_epigraph_constraints_mtlshared(model, x):
    constraints = []
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
    return y, constraints


def build_constraints_from_model(
    model,
    x,
    *,
    mode: str,
    x_np: np.ndarray | None = None,
    x_min: np.ndarray | None = None,
    x_max: np.ndarray | None = None,
    milp_binary_last_relu_only: bool = False,
    milp_binary_last_shared_and_head: bool = False,
):
    """Build constraints for MLP or MTLSharedHeads without requiring caller to know the type."""
    mode = mode.lower()
    if mode not in {"epigraph", "fixed_pattern", "milp"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if hasattr(model, "group_blocks") and len(getattr(model, "group_blocks")) > 0:
        raise NotImplementedError("Grouped shared heads are not supported in constraint builder.")

    if hasattr(model, "net"):
        if mode == "epigraph":
            return _build_epigraph_constraints_mlp(model, x)
        if mode == "fixed_pattern":
            if x_np is None:
                raise ValueError("x_np is required for fixed_pattern constraints.")
            return build_fixed_pattern_constraints_mlp(model, x, x_np)
        if x_min is None or x_max is None:
            raise ValueError("x_min/x_max are required for MILP constraints.")
        return build_milp_constraints_mlp(
            model,
            x,
            x_min,
            x_max,
            binary_last_relu_only=milp_binary_last_relu_only,
        )

    if hasattr(model, "shared") and hasattr(model, "heads"):
        if mode == "epigraph":
            return _build_epigraph_constraints_mtlshared(model, x)
        if mode == "fixed_pattern":
            if x_np is None:
                raise ValueError("x_np is required for fixed_pattern constraints.")
            return build_fixed_pattern_constraints_mtlshared(model, x, x_np)
        if x_min is None or x_max is None:
            raise ValueError("x_min/x_max are required for MILP constraints.")
        return build_milp_constraints_mtlshared(
            model,
            x,
            x_min,
            x_max,
            binary_last_relu_only=milp_binary_last_relu_only,
            binary_last_shared_and_head=milp_binary_last_shared_and_head,
        )

    raise NotImplementedError("Unsupported model type for constraints.")
