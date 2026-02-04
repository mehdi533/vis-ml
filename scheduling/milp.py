from __future__ import annotations

import cvxpy as cp
import numpy as np

from scheduling.utils import _extract_linear_layers, _relu_epigraph


def _relu_big_m(z, y, a, z_min, z_max):
    return [
        y >= 0,
        y >= z,
        y <= z - cp.multiply(z_min, (1 - a)),
        y <= cp.multiply(z_max, a),
    ]


def _interval_bounds(W, b, h_min, h_max):
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)
    z_min = W_pos @ h_min + W_neg @ h_max + b
    z_max = W_pos @ h_max + W_neg @ h_min + b
    return z_min, z_max


def _relu_with_pruning(z, y, z_min, z_max, *, name: str):
    if np.all(z_min >= 0):
        return [y == z, z >= 0]
    if np.all(z_max <= 0):
        return [y == 0, z <= 0]
    a = cp.Variable(z.shape[0], boolean=True, name=f"{name}_bin")
    return _relu_big_m(z, y, a, z_min, z_max)


def build_milp_constraints_mtlshared(
    model,
    x,
    x_min,
    x_max,
    *,
    binary_last_relu_only: bool = False,
    binary_last_shared_and_head: bool = False,
):
    constraints = []
    h = x
    h_min = x_min.copy()
    h_max = x_max.copy()

    shared_layers = _extract_linear_layers(model.shared)
    last_shared_idx = len(shared_layers) - 1
    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        z_min, z_max = _interval_bounds(W, b, h_min, h_max)
        y = cp.Variable(b.shape[0], name=f"shared_{idx}")
        if binary_last_relu_only or binary_last_shared_and_head:
            use_binary = idx == last_shared_idx
        else:
            use_binary = True
        if use_binary:
            constraints += _relu_with_pruning(z, y, z_min, z_max, name=f"shared_{idx}")
        else:
            constraints += _relu_epigraph(z, y)
        h = y
        h_min = np.maximum(0, z_min)
        h_max = np.maximum(0, z_max)

    outputs = []
    for i, head in enumerate(model.heads):
        h_head = h
        hmin_head = h_min
        hmax_head = h_max
        head_layers = _extract_linear_layers(head)
        last_head_relu_idx = len(head_layers) - 2
        for idx, (W, b) in enumerate(head_layers):
            z = W @ h_head + b
            if idx < len(head_layers) - 1:
                z_min, z_max = _interval_bounds(W, b, hmin_head, hmax_head)
                y = cp.Variable(b.shape[0], name=f"head{i}_{idx}")
                if binary_last_shared_and_head:
                    use_binary = idx == last_head_relu_idx
                elif binary_last_relu_only:
                    use_binary = idx == last_head_relu_idx
                else:
                    use_binary = True
                if use_binary:
                    constraints += _relu_with_pruning(z, y, z_min, z_max, name=f"head{i}_{idx}")
                else:
                    constraints += _relu_epigraph(z, y)
                h_head = y
                hmin_head = np.maximum(0, z_min)
                hmax_head = np.maximum(0, z_max)
            else:
                y_out = cp.Variable(1, name=f"out{i}")
                constraints.append(y_out == z)
                outputs.append(y_out)

    y = cp.hstack(outputs)
    return y, constraints
