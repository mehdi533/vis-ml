from __future__ import annotations

import cvxpy as cp
import numpy as np

from scheduling.convex_reformulation.utils import (
    _extract_linear_layers,
    activation_masks_mtlshared,
)


def build_fixed_pattern_constraints_mtlshared(model, x, x_np: np.ndarray):
    constraints = []
    h = x

    shared_layers = _extract_linear_layers(model.shared)
    shared_masks, head_masks = activation_masks_mtlshared(model, x_np)

    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        y = cp.Variable(b.shape[0], name=f"shared_{idx}")
        mask = shared_masks[idx]
        if np.all(mask):
            constraints += [z >= 0, y == z]
        elif np.all(~mask):
            constraints += [z <= 0, y == 0]
        else:
            constraints += [
                z[mask] >= 0,
                y[mask] == z[mask],
                z[~mask] <= 0,
                y[~mask] == 0,
            ]
        h = y

    outputs = []
    head_relu_idx = 0
    for i, head in enumerate(model.heads):
        h_head = h
        head_layers = _extract_linear_layers(head)
        for idx, (W, b) in enumerate(head_layers):
            z = W @ h_head + b
            if idx < len(head_layers) - 1:
                y = cp.Variable(b.shape[0], name=f"head{i}_{idx}")
                mask = head_masks[head_relu_idx]
                head_relu_idx += 1
                if np.all(mask):
                    constraints += [z >= 0, y == z]
                elif np.all(~mask):
                    constraints += [z <= 0, y == 0]
                else:
                    constraints += [
                        z[mask] >= 0,
                        y[mask] == z[mask],
                        z[~mask] <= 0,
                        y[~mask] == 0,
                    ]
                h_head = y
            else:
                y_out = cp.Variable(1, name=f"out{i}")
                constraints.append(y_out == z)
                outputs.append(y_out)

    y = cp.hstack(outputs)
    return y, constraints
