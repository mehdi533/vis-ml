from __future__ import annotations

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F

from models.convex_models import NonNegLinear


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _extract_linear_layers(seq):
    layers = []
    for m in _linear_layers(seq):
        w = m.weight
        if isinstance(m, NonNegLinear):
            w = F.softplus(w)
        layers.append((w.detach().cpu().numpy(), m.bias.detach().cpu().numpy()))
    return layers


def _resolve_active_output_idx(active_output_idx, n_outputs: int) -> list[int]:
    if active_output_idx is None:
        return list(range(int(n_outputs)))
    active = sorted(set(int(i) for i in active_output_idx))
    bad = [idx for idx in active if idx < 0 or idx >= int(n_outputs)]
    if bad:
        raise ValueError(
            f"constraints.nn_active_output_idx contains invalid output indices: {bad}. "
            f"Valid range is [0, {max(int(n_outputs) - 1, 0)}]."
        )
    return active


def _interval_bounds(W, b, h_min, h_max):
    W_pos = np.maximum(W, 0)
    W_neg = np.minimum(W, 0)
    z_min = W_pos @ h_min + W_neg @ h_max + b
    z_max = W_pos @ h_max + W_neg @ h_min + b
    return z_min, z_max


def _relu_with_pruning(z, y, z_min, z_max, *, name: str):
    constraints = []
    active_mask = z_min >= 0
    inactive_mask = z_max <= 0
    undecided_mask = ~(active_mask | inactive_mask)

    if np.any(active_mask):
        idx = np.flatnonzero(active_mask)
        constraints += [y[idx] == z[idx], z[idx] >= 0]
    if np.any(inactive_mask):
        idx = np.flatnonzero(inactive_mask)
        constraints += [y[idx] == 0, z[idx] <= 0]
    if np.any(undecided_mask):
        idx = np.flatnonzero(undecided_mask)
        a = cp.Variable(idx.shape[0], boolean=True, name=f"{name}_bin")
        constraints += [
            y[idx] >= 0,
            y[idx] >= z[idx],
            y[idx] <= z[idx] - cp.multiply(z_min[idx], (1 - a)),
            y[idx] <= cp.multiply(z_max[idx], a),
        ]
    return constraints


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


def _activation_masks_mlp(model, x_np: np.ndarray):
    with torch.no_grad():
        x = torch.tensor(x_np, dtype=torch.float32)
        h = x
        masks = []
        last_z = None
        for layer in model.net:
            if isinstance(layer, torch.nn.Linear):
                last_z = layer(h)
                h = last_z
            elif isinstance(layer, torch.nn.ReLU):
                if last_z is None:
                    continue
                masks.append((last_z > 0).cpu().numpy().reshape(-1))
                h = layer(last_z)
            elif isinstance(layer, torch.nn.Dropout):
                h = layer(h)
    return masks


def _build_fixed_pattern_constraints_mlp(model, x, x_np: np.ndarray):
    constraints = []
    masks = _activation_masks_mlp(model, x_np)
    h = x
    layers = _extract_linear_layers(model.net)
    relu_idx = 0
    for idx, (W, b) in enumerate(layers):
        z = W @ h + b
        if idx == len(layers) - 1:
            y = cp.Variable(W.shape[0], name="outputs")
            constraints.append(y == z)
            return y, constraints
        out = cp.Variable(len(b), name=f"mlp_{idx}")
        mask = masks[relu_idx]
        relu_idx += 1
        if np.all(mask):
            constraints += [z >= 0, out == z]
        elif np.all(~mask):
            constraints += [z <= 0, out == 0]
        else:
            constraints += [z[mask] >= 0, out[mask] == z[mask], z[~mask] <= 0, out[~mask] == 0]
        h = out
    raise RuntimeError("Invalid MLP architecture for fixed pattern constraints.")


def _build_fixed_pattern_constraints_mtlshared(
    model,
    x,
    x_np: np.ndarray,
    *,
    active_output_idx=None,
    collect_blocks: bool = False,
):
    constraints = []
    active_outputs = set(_resolve_active_output_idx(active_output_idx, len(model.heads)))
    block_map: dict[str, list[cp.Constraint]] = {"nn_shared": []}
    shared_masks, head_masks = _activation_masks_mtlshared(model, x_np)
    h = x
    shared_layers = _extract_linear_layers(model.shared)
    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        out = cp.Variable(len(b), name=f"shared_{idx}")
        mask = shared_masks[idx]
        if np.all(mask):
            relu_constraints = [z >= 0, out == z]
        elif np.all(~mask):
            relu_constraints = [z <= 0, out == 0]
        else:
            relu_constraints = [z[mask] >= 0, out[mask] == z[mask], z[~mask] <= 0, out[~mask] == 0]
        constraints += relu_constraints
        block_map["nn_shared"] += relu_constraints
        h = out

    outputs: dict[int, cp.Expression] = {}
    head_relu_idx = 0
    for i, head in enumerate(model.heads):
        hh = h
        head_layers = _extract_linear_layers(head)
        block_name = f"nn_head_{i}"
        if i in active_outputs:
            block_map[block_name] = []
        for idx, (W, b) in enumerate(head_layers):
            if idx < len(head_layers) - 1:
                mask = head_masks[head_relu_idx]
                head_relu_idx += 1
                if i not in active_outputs:
                    continue
            elif i not in active_outputs:
                continue
            z = W @ hh + b
            if idx == len(head_layers) - 1:
                y_out = cp.Variable(1, name=f"out{i}")
                out_constraint = (y_out == z)
                constraints.append(out_constraint)
                block_map[block_name].append(out_constraint)
                outputs[int(i)] = y_out
                continue
            out = cp.Variable(len(b), name=f"head{i}_{idx}")
            if np.all(mask):
                relu_constraints = [z >= 0, out == z]
            elif np.all(~mask):
                relu_constraints = [z <= 0, out == 0]
            else:
                relu_constraints = [z[mask] >= 0, out[mask] == z[mask], z[~mask] <= 0, out[~mask] == 0]
            constraints += relu_constraints
            block_map[block_name] += relu_constraints
            hh = out
    if collect_blocks:
        return outputs, constraints, block_map
    return outputs, constraints


def _build_milp_constraints_mlp(model, x, x_min, x_max):
    constraints = []
    h = x
    h_min = x_min.copy()
    h_max = x_max.copy()

    layers = _extract_linear_layers(model.net)
    for idx, (W, b) in enumerate(layers):
        z = W @ h + b
        if idx < len(layers) - 1:
            z_min, z_max = _interval_bounds(W, b, h_min, h_max)
            y = cp.Variable(b.shape[0], name=f"mlp_{idx}")
            constraints += _relu_with_pruning(z, y, z_min, z_max, name=f"mlp_{idx}")
            h = y
            h_min = np.maximum(0, z_min)
            h_max = np.maximum(0, z_max)
        else:
            y_out = cp.Variable(b.shape[0], name="mlp_out")
            constraints.append(y_out == z)
    return y_out, constraints


def _build_milp_constraints_mtlshared(
    model,
    x,
    x_min,
    x_max,
    *,
    active_output_idx=None,
    collect_blocks: bool = False,
):
    constraints = []
    active_outputs = set(_resolve_active_output_idx(active_output_idx, len(model.heads)))
    block_map: dict[str, list[cp.Constraint]] = {"nn_shared": []}
    h = x
    h_min = x_min.copy()
    h_max = x_max.copy()

    shared_layers = _extract_linear_layers(model.shared)
    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        z_min, z_max = _interval_bounds(W, b, h_min, h_max)
        y = cp.Variable(b.shape[0], name=f"shared_{idx}")
        relu_constraints = _relu_with_pruning(z, y, z_min, z_max, name=f"shared_{idx}")
        constraints += relu_constraints
        block_map["nn_shared"] += relu_constraints
        h = y
        h_min = np.maximum(0, z_min)
        h_max = np.maximum(0, z_max)

    outputs: dict[int, cp.Expression] = {}
    for i, head in enumerate(model.heads):
        if i not in active_outputs:
            continue
        h_head = h
        hmin_head = h_min
        hmax_head = h_max
        head_layers = _extract_linear_layers(head)
        block_name = f"nn_head_{i}"
        block_map[block_name] = []
        for idx, (W, b) in enumerate(head_layers):
            z = W @ h_head + b
            if idx < len(head_layers) - 1:
                z_min, z_max = _interval_bounds(W, b, hmin_head, hmax_head)
                y = cp.Variable(b.shape[0], name=f"head{i}_{idx}")
                relu_constraints = _relu_with_pruning(z, y, z_min, z_max, name=f"head{i}_{idx}")
                constraints += relu_constraints
                block_map[block_name] += relu_constraints
                h_head = y
                hmin_head = np.maximum(0, z_min)
                hmax_head = np.maximum(0, z_max)
            else:
                y_out = cp.Variable(1, name=f"out{i}")
                out_constraint = (y_out == z)
                constraints.append(out_constraint)
                block_map[block_name].append(out_constraint)
                outputs[int(i)] = y_out
    if collect_blocks:
        return outputs, constraints, block_map
    return outputs, constraints


def _build_constraints_from_relu_model(
    model,
    x,
    *,
    mode: str,
    x_np=None,
    x_min=None,
    x_max=None,
    active_output_idx=None,
    collect_blocks: bool = False,
):
    mode = mode.lower()
    if mode not in {"fixed_pattern", "milp"}:
        raise ValueError(f"Unsupported nn.mode: {mode}")

    if hasattr(model, "net"):
        if mode == "fixed_pattern":
            if x_np is None:
                raise ValueError("x_seed_sc is required for nn.mode='fixed_pattern'.")
            return _build_fixed_pattern_constraints_mlp(model, x, x_np)
        if x_min is None or x_max is None:
            raise ValueError("x_min_sc/x_max_sc are required for nn.mode='milp'.")
        return _build_milp_constraints_mlp(model, x, x_min, x_max)

    if hasattr(model, "shared") and hasattr(model, "heads"):
        if getattr(model, "group_blocks", None):
            raise ValueError("Grouped shared-head models are not supported in this NN MILP builder.")
        if mode == "fixed_pattern":
            if x_np is None:
                raise ValueError("x_seed_sc is required for nn.mode='fixed_pattern'.")
            return _build_fixed_pattern_constraints_mtlshared(
                model,
                x,
                x_np,
                active_output_idx=active_output_idx,
                collect_blocks=collect_blocks,
            )
        if x_min is None or x_max is None:
            raise ValueError("x_min_sc/x_max_sc are required for nn.mode='milp'.")
        return _build_milp_constraints_mtlshared(
            model,
            x,
            x_min,
            x_max,
            active_output_idx=active_output_idx,
            collect_blocks=collect_blocks,
        )

    raise NotImplementedError("Unsupported ReLU model type for generic builder.")


def compute_nn_output_interval(
    model,
    x_min_sc: np.ndarray,
    x_max_sc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the NN output interval [y_min, y_max] in scaled space via
    interval arithmetic (same bounds the MILP builder uses for Big-M).

    Only supports MTLSharedHeads models with ``shared`` + ``heads`` attributes.
    Returns arrays of shape ``(n_tasks,)`` with the per-head output bounds.
    """
    if not (hasattr(model, "shared") and hasattr(model, "heads")):
        raise NotImplementedError("compute_nn_output_interval only supports MTLSharedHeads")

    h_min, h_max = x_min_sc.copy(), x_max_sc.copy()
    for W, b in _extract_linear_layers(model.shared):
        z_min, z_max = _interval_bounds(W, b, h_min, h_max)
        h_min = np.maximum(0, z_min)
        h_max = np.maximum(0, z_max)

    y_lo = np.empty(len(model.heads))
    y_hi = np.empty(len(model.heads))
    for i, head in enumerate(model.heads):
        hmin, hmax = h_min.copy(), h_max.copy()
        head_layers = _extract_linear_layers(head)
        for idx, (W, b) in enumerate(head_layers):
            z_min, z_max = _interval_bounds(W, b, hmin, hmax)
            if idx < len(head_layers) - 1:
                hmin = np.maximum(0, z_min)
                hmax = np.maximum(0, z_max)
        y_lo[i] = float(z_min.item())
        y_hi[i] = float(z_max.item())
    return y_lo, y_hi


def build_nn_constraints(
    *,
    model,
    x,
    x_seed_sc: np.ndarray,
    x_min_sc: np.ndarray,
    x_max_sc: np.ndarray,
    mode: str,
    active_output_idx: list[int] | None = None,
    return_blocks: bool = False,
):
    """
    Unified NN builder used by final optimization.
    """
    try:
        return _build_constraints_from_relu_model(
            model,
            x,
            mode=mode,
            x_np=x_seed_sc,
            x_min=x_min_sc,
            x_max=x_max_sc,
            active_output_idx=active_output_idx,
            collect_blocks=return_blocks,
        )
    except NotImplementedError:
        pass

    model_name = model.__class__.__name__
    mode = mode.lower()

    if model_name == "FICNN" or hasattr(model, "first_wx"):
        raise ValueError(
            "FICNN optimization embeddings are unsupported in this scheduler."
        )

    if hasattr(model, "picnn"):
        raise ValueError(
            "PICNN optimization embeddings are unsupported in this scheduler."
        )

    if hasattr(model, "picnn_mtlsh"):
        raise ValueError(
            "PICNN_MTLSH optimization embeddings are unsupported in this scheduler."
        )

    raise ValueError(
        f"NN constraints are not implemented for model class '{model_name}'. "
        "Supported: ReLU MLP/MTLSH with nn.mode in {'milp', 'fixed_pattern'}."
    )


def predict_outputs_scaled(model, x_scaled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        if model.__class__.__name__ in {"PICNN", "PICNN_MTLSH"}:
            raise ValueError("Raw PICNN/PICNN_MTLSH models require explicit (u, v) inputs.")
        return model(x_t).cpu().numpy().reshape(-1)
