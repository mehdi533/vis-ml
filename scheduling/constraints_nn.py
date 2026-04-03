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


def _linear_weights_bias(layer):
    w = layer.weight
    if isinstance(layer, NonNegLinear):
        w = F.softplus(w)
    return w.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()


def _relu_epigraph(z, y):
    return [y >= 0, y >= z]


def _apply_relu_stack(h, layers, constraints, *, prefix: str):
    for idx, (w, b) in enumerate(layers):
        z = w @ h + b
        y = cp.Variable(b.shape[0], name=f"{prefix}_{idx}")
        constraints += _relu_epigraph(z, y)
        h = y
    return h


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


def _build_epigraph_constraints_mlp(model, x):
    constraints = []
    layers = _extract_linear_layers(model.net)
    h = _apply_relu_stack(x, layers[:-1], constraints, prefix="mlp")
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
    return cp.hstack(outputs), constraints


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


def _build_fixed_pattern_constraints_mtlshared(model, x, x_np: np.ndarray):
    constraints = []
    shared_masks, head_masks = _activation_masks_mtlshared(model, x_np)
    h = x
    shared_layers = _extract_linear_layers(model.shared)
    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        out = cp.Variable(len(b), name=f"shared_{idx}")
        mask = shared_masks[idx]
        if np.all(mask):
            constraints += [z >= 0, out == z]
        elif np.all(~mask):
            constraints += [z <= 0, out == 0]
        else:
            constraints += [z[mask] >= 0, out[mask] == z[mask], z[~mask] <= 0, out[~mask] == 0]
        h = out

    outputs = []
    head_relu_idx = 0
    for i, head in enumerate(model.heads):
        hh = h
        head_layers = _extract_linear_layers(head)
        for idx, (W, b) in enumerate(head_layers):
            z = W @ hh + b
            if idx == len(head_layers) - 1:
                y_out = cp.Variable(1, name=f"out{i}")
                constraints.append(y_out == z)
                outputs.append(y_out)
                continue
            out = cp.Variable(len(b), name=f"head{i}_{idx}")
            mask = head_masks[head_relu_idx]
            head_relu_idx += 1
            if np.all(mask):
                constraints += [z >= 0, out == z]
            elif np.all(~mask):
                constraints += [z <= 0, out == 0]
            else:
                constraints += [z[mask] >= 0, out[mask] == z[mask], z[~mask] <= 0, out[~mask] == 0]
            hh = out
    return cp.hstack(outputs), constraints


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


def _build_milp_constraints_mtlshared(model, x, x_min, x_max):
    constraints = []
    h = x
    h_min = x_min.copy()
    h_max = x_max.copy()

    shared_layers = _extract_linear_layers(model.shared)
    for idx, (W, b) in enumerate(shared_layers):
        z = W @ h + b
        z_min, z_max = _interval_bounds(W, b, h_min, h_max)
        y = cp.Variable(b.shape[0], name=f"shared_{idx}")
        constraints += _relu_with_pruning(z, y, z_min, z_max, name=f"shared_{idx}")
        h = y
        h_min = np.maximum(0, z_min)
        h_max = np.maximum(0, z_max)

    outputs = []
    for i, head in enumerate(model.heads):
        h_head = h
        hmin_head = h_min
        hmax_head = h_max
        head_layers = _extract_linear_layers(head)
        for idx, (W, b) in enumerate(head_layers):
            z = W @ h_head + b
            if idx < len(head_layers) - 1:
                z_min, z_max = _interval_bounds(W, b, hmin_head, hmax_head)
                y = cp.Variable(b.shape[0], name=f"head{i}_{idx}")
                constraints += _relu_with_pruning(z, y, z_min, z_max, name=f"head{i}_{idx}")
                h_head = y
                hmin_head = np.maximum(0, z_min)
                hmax_head = np.maximum(0, z_max)
            else:
                y_out = cp.Variable(1, name=f"out{i}")
                constraints.append(y_out == z)
                outputs.append(y_out)
    return cp.hstack(outputs), constraints


def _build_ficnn_epigraph(model, x):
    constraints = []

    if model.first_wx is None:
        w_out, b_out = _linear_weights_bias(model.out_wx)
        y = cp.Variable(model.out_dim, name="ficnn_out")
        constraints += [y == w_out @ x + b_out]
        return y, constraints

    w0, b0 = _linear_weights_bias(model.first_wx)
    z = cp.Variable(w0.shape[0], name="ficnn_z0")
    constraints += [z >= 0, z >= w0 @ x + b0]

    for idx, (wz_layer, wx_layer) in enumerate(zip(model.Wz_layers, model.Wx_layers), start=1):
        wz, bz = _linear_weights_bias(wz_layer)
        wx, bx = _linear_weights_bias(wx_layer)
        z_next = cp.Variable(wz.shape[0], name=f"ficnn_z{idx}")
        affine = wz @ z + bz + wx @ x + bx
        constraints += [z_next >= 0, z_next >= affine]
        z = z_next

    w_out_z, b_out_z = _linear_weights_bias(model.out_wz)
    w_out_x, b_out_x = _linear_weights_bias(model.out_wx)
    y = cp.Variable(model.out_dim, name="ficnn_out")
    constraints += [y == w_out_z @ z + b_out_z + w_out_x @ x + b_out_x]
    return y, constraints


def _split_tabular_picnn_inputs(model, x):
    u_idx = np.asarray(model.u_idx.detach().cpu().numpy(), dtype=int)
    v_idx = np.asarray(model.v_idx.detach().cpu().numpy(), dtype=int)
    u = x[u_idx]
    v = x[v_idx] if v_idx.size else None
    return u, v


def _affine_optional(layer, inp):
    if layer is None:
        return 0.0
    w, b = _linear_weights_bias(layer)
    return w @ inp + b


def _build_picnn_epigraph_inner(picnn, u, v):
    constraints = []

    if picnn.first_wu is None:
        w_u, b_u = _linear_weights_bias(picnn.out_wu)
        affine = w_u @ u + b_u
        if picnn.out_wv is not None:
            affine = affine + _affine_optional(picnn.out_wv, v)
        y = cp.Variable(picnn.out_dim, name="picnn_out")
        constraints += [y == affine]
        return y, constraints

    w_u0, b_u0 = _linear_weights_bias(picnn.first_wu)
    affine0 = w_u0 @ u + b_u0
    if picnn.first_wv is not None:
        affine0 = affine0 + _affine_optional(picnn.first_wv, v)
    z = cp.Variable(w_u0.shape[0], name="picnn_z0")
    constraints += [z >= 0, z >= affine0]

    for idx, (Wz_layer, Wu_layer, Wv_layer) in enumerate(
        zip(picnn.Wz_layers, picnn.Wu_layers, picnn.Wv_layers),
        start=1,
    ):
        wz, bz = _linear_weights_bias(Wz_layer)
        wu, bu = _linear_weights_bias(Wu_layer)
        affine = wz @ z + bz + wu @ u + bu
        if Wv_layer is not None:
            affine = affine + _affine_optional(Wv_layer, v)
        z_next = cp.Variable(wz.shape[0], name=f"picnn_z{idx}")
        constraints += [z_next >= 0, z_next >= affine]
        z = z_next

    wz_out, bz_out = _linear_weights_bias(picnn.out_wz)
    wu_out, bu_out = _linear_weights_bias(picnn.out_wu)
    y_affine = wz_out @ z + bz_out + wu_out @ u + bu_out
    if picnn.out_wv is not None:
        y_affine = y_affine + _affine_optional(picnn.out_wv, v)
    y = cp.Variable(picnn.out_dim, name="picnn_out")
    constraints += [y == y_affine]
    return y, constraints


def _build_picnn_epigraph(tabular_model, x):
    u, v = _split_tabular_picnn_inputs(tabular_model, x)
    return _build_picnn_epigraph_inner(tabular_model.picnn, u, v)


def _build_picnn_trunk_epigraph(trunk, u, v, constraints, *, prefix: str):
    if trunk.first_wu is None:
        return cp.Constant(np.zeros(0, dtype=float))

    wu0, bu0 = _linear_weights_bias(trunk.first_wu)
    affine0 = wu0 @ u + bu0
    if trunk.first_wv is not None:
        affine0 = affine0 + _affine_optional(trunk.first_wv, v)
    z = cp.Variable(wu0.shape[0], name=f"{prefix}_z0")
    constraints += [z >= 0, z >= affine0]

    for idx, (Wz_layer, Wu_layer, Wv_layer) in enumerate(
        zip(trunk.Wz_layers, trunk.Wu_layers, trunk.Wv_layers),
        start=1,
    ):
        wz, bz = _linear_weights_bias(Wz_layer)
        wu, bu = _linear_weights_bias(Wu_layer)
        affine = wz @ z + bz + wu @ u + bu
        if Wv_layer is not None:
            affine = affine + _affine_optional(Wv_layer, v)
        z_next = cp.Variable(wz.shape[0], name=f"{prefix}_z{idx}")
        constraints += [z_next >= 0, z_next >= affine]
        z = z_next
    return z


def _build_picnn_from_z_epigraph(block, z_in, u, v, constraints, *, prefix: str):
    if block.first_wz is None:
        return z_in

    wz0, bz0 = _linear_weights_bias(block.first_wz)
    wu0, bu0 = _linear_weights_bias(block.first_wu)
    affine0 = wz0 @ z_in + bz0 + wu0 @ u + bu0
    if block.first_wv is not None:
        affine0 = affine0 + _affine_optional(block.first_wv, v)
    z = cp.Variable(wz0.shape[0], name=f"{prefix}_z0")
    constraints += [z >= 0, z >= affine0]

    for idx, (Wz_layer, Wu_layer, Wv_layer) in enumerate(
        zip(block.Wz_layers, block.Wu_layers, block.Wv_layers),
        start=1,
    ):
        wz, bz = _linear_weights_bias(Wz_layer)
        wu, bu = _linear_weights_bias(Wu_layer)
        affine = wz @ z + bz + wu @ u + bu
        if Wv_layer is not None:
            affine = affine + _affine_optional(Wv_layer, v)
        z_next = cp.Variable(wz.shape[0], name=f"{prefix}_z{idx}")
        constraints += [z_next >= 0, z_next >= affine]
        z = z_next
    return z


def _build_picnn_head_epigraph(head, z_in, u, v, constraints, *, prefix: str):
    if head.first_wz is None:
        wz_out, bz_out = _linear_weights_bias(head.out_wz)
        wu_out, bu_out = _linear_weights_bias(head.out_wu)
        affine = wz_out @ z_in + bz_out + wu_out @ u + bu_out
        if head.out_wv is not None:
            affine = affine + _affine_optional(head.out_wv, v)
        y = cp.Variable(1, name=f"{prefix}_out")
        constraints += [y == affine]
        return y

    wz0, bz0 = _linear_weights_bias(head.first_wz)
    wu0, bu0 = _linear_weights_bias(head.first_wu)
    affine0 = wz0 @ z_in + bz0 + wu0 @ u + bu0
    if head.first_wv is not None:
        affine0 = affine0 + _affine_optional(head.first_wv, v)
    z = cp.Variable(wz0.shape[0], name=f"{prefix}_z0")
    constraints += [z >= 0, z >= affine0]

    for idx, (Wz_layer, Wu_layer, Wv_layer) in enumerate(
        zip(head.Wz_layers, head.Wu_layers, head.Wv_layers),
        start=1,
    ):
        wz, bz = _linear_weights_bias(Wz_layer)
        wu, bu = _linear_weights_bias(Wu_layer)
        affine = wz @ z + bz + wu @ u + bu
        if Wv_layer is not None:
            affine = affine + _affine_optional(Wv_layer, v)
        z_next = cp.Variable(wz.shape[0], name=f"{prefix}_z{idx}")
        constraints += [z_next >= 0, z_next >= affine]
        z = z_next

    w_mid, b_mid = _linear_weights_bias(head.mid_out_wz)
    w_out_z, b_out_z = _linear_weights_bias(head.out_wz)
    w_out_u, b_out_u = _linear_weights_bias(head.out_wu)

    affine_out = w_mid @ z + b_mid + w_out_z @ z_in + b_out_z + w_out_u @ u + b_out_u
    if head.out_wv is not None:
        affine_out = affine_out + _affine_optional(head.out_wv, v)
    y = cp.Variable(1, name=f"{prefix}_out")
    constraints += [y == affine_out]
    return y


def _build_picnn_mtlsh_epigraph(tabular_model, x):
    constraints = []
    u, v = _split_tabular_picnn_inputs(tabular_model, x)
    model = tabular_model.picnn_mtlsh

    z_shared = _build_picnn_trunk_epigraph(model.trunk, u, v, constraints, prefix="picnn_trunk")
    head_inputs = [z_shared for _ in range(model.n_tasks)]

    for block_idx, (block, indices) in enumerate(zip(model.group_blocks, model.group_block_indices)):
        for head_idx in indices:
            head_inputs[int(head_idx)] = _build_picnn_from_z_epigraph(
                block,
                head_inputs[int(head_idx)],
                u,
                v,
                constraints,
                prefix=f"picnn_group{block_idx}_h{int(head_idx)}",
            )

    outputs = []
    for i, head in enumerate(model.heads):
        y_i = _build_picnn_head_epigraph(
            head,
            head_inputs[i],
            u,
            v,
            constraints,
            prefix=f"picnn_head{i}",
        )
        outputs.append(y_i)
    return cp.hstack(outputs), constraints


def _build_constraints_from_relu_model(model, x, *, mode: str, x_np=None, x_min=None, x_max=None):
    mode = mode.lower()
    if mode not in {"epigraph", "fixed_pattern", "milp"}:
        raise ValueError(f"Unsupported nn.mode: {mode}")

    if hasattr(model, "net"):
        if mode == "epigraph":
            return _build_epigraph_constraints_mlp(model, x)
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
        if mode == "epigraph":
            return _build_epigraph_constraints_mtlshared(model, x)
        if mode == "fixed_pattern":
            if x_np is None:
                raise ValueError("x_seed_sc is required for nn.mode='fixed_pattern'.")
            return _build_fixed_pattern_constraints_mtlshared(model, x, x_np)
        if x_min is None or x_max is None:
            raise ValueError("x_min_sc/x_max_sc are required for nn.mode='milp'.")
        return _build_milp_constraints_mtlshared(model, x, x_min, x_max)

    raise NotImplementedError("Unsupported ReLU model type for generic builder.")


def build_nn_constraints(
    *,
    model,
    x,
    x_seed_sc: np.ndarray,
    x_min_sc: np.ndarray,
    x_max_sc: np.ndarray,
    mode: str,
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
        )
    except NotImplementedError:
        pass

    model_name = model.__class__.__name__
    mode = mode.lower()

    if model_name == "FICNN" or hasattr(model, "first_wx"):
        if mode != "epigraph":
            raise ValueError("FICNN currently supports nn.mode='epigraph' only.")
        return _build_ficnn_epigraph(model, x)

    if hasattr(model, "picnn"):
        if mode != "epigraph":
            raise ValueError("PICNN currently supports nn.mode='epigraph' only.")
        return _build_picnn_epigraph(model, x)

    if hasattr(model, "picnn_mtlsh"):
        if mode != "epigraph":
            raise ValueError("PICNN_MTLSH currently supports nn.mode='epigraph' only.")
        return _build_picnn_mtlsh_epigraph(model, x)

    raise ValueError(
        f"NN constraints are not implemented for model class '{model_name}'. "
        "Supported: ReLU MLP/MTLSH, FICNN, PICNN, PICNN_MTLSH."
    )


def predict_outputs_scaled(model, x_scaled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        if model.__class__.__name__ in {"PICNN", "PICNN_MTLSH"}:
            raise ValueError("Raw PICNN/PICNN_MTLSH models require explicit (u, v) inputs.")
        return model(x_t).cpu().numpy().reshape(-1)
