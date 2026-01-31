from __future__ import annotations

from typing import List, Tuple

import cvxpy as cp
import numpy as np
import torch


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _extract_linear_layers(seq):
    return [
        (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
        for m in _linear_layers(seq)
    ]


def _activation_masks_mtlshared(model, x_np: np.ndarray):
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32)
        h = x
        shared_masks = []
        last_z = None
        for layer in model.shared:
            if isinstance(layer, torch.nn.Linear):
                last_z = layer(h)
                h = last_z
            elif isinstance(layer, torch.nn.ReLU):
                if last_z is None:
                    continue
                shared_masks.append((last_z > 0).cpu().numpy().reshape(-1))
                h = layer(last_z)
            elif isinstance(layer, torch.nn.Dropout):
                h = layer(h)

        head_masks = []
        for head in model.heads:
            h_head = h
            last_z = None
            for layer in head:
                if isinstance(layer, torch.nn.Linear):
                    last_z = layer(h_head)
                    h_head = last_z
                elif isinstance(layer, torch.nn.ReLU):
                    if last_z is None:
                        continue
                    head_masks.append((last_z > 0).cpu().numpy().reshape(-1))
                    h_head = layer(last_z)
                elif isinstance(layer, torch.nn.Dropout):
                    h_head = layer(h_head)

    return shared_masks, head_masks


def build_fixed_pattern_constraints_mtlshared(model, x, x_np: np.ndarray):
    constraints = []
    h = x

    shared_layers = _extract_linear_layers(model.shared)
    shared_masks, head_masks = _activation_masks_mtlshared(model, x_np)

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
