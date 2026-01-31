from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import andes
import cvxpy as cp
import joblib
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torch
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

from data_generation.extract_metrics import build_feature_row
from experiments.run_sim_extract_ed import _repeat_or_validate, add_measurement_devices
from scheduling.mtlsh_relu_convex import build_mtlsh_convex_constraints, compute_feature_bounds_from_training_data
from models.models import MLP, MTLGroupedSharedHeads, MTLSharedHeads, SharedGroupSpec


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_steps(args) -> np.ndarray:
    if args.steps:
        return np.asarray([float(x) for x in args.steps.split(",")], dtype=float)
    return np.linspace(args.step_min, args.step_max, args.step_num, dtype=float)


def _setup_system(cfg: Dict, base_scale: float, rng: np.random.Generator):
    ss = andes.load(cfg["case"], setup=False)
    ss.config.freq = float(50)
    add_measurement_devices(ss)

    regcv1_ids = ss.REGCV1.name.v
    M_vec = rng.uniform(cfg["ibr"]["M_range"][0], cfg["ibr"]["M_range"][1], size=len(regcv1_ids))
    D_vec = rng.uniform(cfg["ibr"]["D_range"][0], cfg["ibr"]["D_range"][1], size=len(regcv1_ids))

    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale

    ss.REGCV1.M.v, ss.REGCV1.D.v = M_vec, D_vec

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0

    ss.setup()

    return ss, M_vec, D_vec


def _build_features(
    ss,
    *,
    base_scale: float,
    step_scale: float,
    load_step_time: float,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
) -> Dict[str, float]:
    pq_p_before = np.asarray(ss.PQ.p0.v, dtype=float).copy()
    pq_q_before = np.asarray(ss.PQ.q0.v, dtype=float).copy()
    pq_names = list(ss.PQ.name.v) if ss.PQ.n else []

    M_agg = np.mean(np.concatenate([ss.GENROU.M.v, ss.REGCV1.M.v])).sum()
    D_agg = np.mean(np.concatenate([ss.GENROU.D.v, ss.REGCV1.D.v])).sum()   

    features = build_feature_row(
        base_load_scale=base_scale,
        load_step_scale=step_scale,
        load_step_time=float(load_step_time),
        pq_names=pq_names,
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
        pq_p_after=pq_p_before * step_scale,
        pq_q_after=pq_q_before * step_scale,
        M_vec=M_vec,
        D_vec=D_vec,
        M_agg=M_agg,
        D_agg=D_agg,
    )
    return features


def _solve_ed(
    *,
    Pd: float,
    Pg_min: np.ndarray,
    Pg_max: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    constraints: List[cp.Constraint],
    solver: str,
):
    ng = len(a)
    Pg = cp.Variable(ng)
    cost_expr = a + cp.multiply(b, Pg) + cp.multiply(c, cp.square(Pg))
    objective = cp.Minimize(cp.sum(cost_expr))
    constraints = list(constraints) + [
        cp.sum(Pg) == Pd,
        Pg >= Pg_min,
        Pg <= Pg_max,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)
    return prob, Pg


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _extract_linear_layers(seq):
    return [
        (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
        for m in _linear_layers(seq)
    ]


def _relu_epigraph(z, y):
    return [y >= 0, y >= z]


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


def _build_milp_constraints_mtlshared(
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
            a = cp.Variable(b.shape[0], boolean=True, name=f"shared_bin_{idx}")
            constraints += _relu_big_m(z, y, a, z_min, z_max)
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
                    a = cp.Variable(b.shape[0], boolean=True, name=f"head{i}_bin_{idx}")
                    constraints += _relu_big_m(z, y, a, z_min, z_max)
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


def _build_torch_model(model_cfg: Dict):
    model_type = str(model_cfg.get("type", "MTLSharedHeads"))
    if model_type == "MTLSharedHeads":
        model = MTLSharedHeads(
            in_dim=int(model_cfg["in_dim"]),
            n_tasks=int(model_cfg["n_tasks"]),
            shared_sizes=model_cfg.get("shared_sizes"),
            head_sizes=model_cfg.get("head_sizes"),
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
        raise NotImplementedError(f"Model type '{model_type}' not supported for torch prediction.")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan step_scale to find ED differences with NN convex constraints.")
    parser.add_argument("--config", default="experiments/generation.yaml", help="Path to sim YAML.")
    parser.add_argument("--cost-config", default="scheduling/mtlsh_convex.yaml", help="Path to cost YAML.")
    parser.add_argument("--base-scale", type=float, default=1.0, help="Base load scale.")
    parser.add_argument("--steps", type=str, default="", help="Comma-separated step_scale values.")
    parser.add_argument("--step-min", type=float, default=0.6, help="Min step_scale.")
    parser.add_argument("--step-max", type=float, default=1.0, help="Max step_scale.")
    parser.add_argument("--step-num", type=int, default=9, help="Number of step_scale samples.")
    parser.add_argument("--solver", type=str, default="GUROBI", help="CVXPY solver for convex ED.")
    parser.add_argument("--diff-tol", type=float, default=1e-3, help="Norm threshold for Pg diff.")
    parser.add_argument("--plot-dir", type=str, default="experiments", help="Directory to save plots.")
    parser.add_argument("--use-milp", action="store_true", help="Use exact ReLU MILP constraints.")
    parser.add_argument(
        "--milp-mode",
        type=str,
        default="full",
        choices=["full", "last"],
        help="MILP ReLU mode: full binaries or only last shared + last head.",
    )
    parser.add_argument("--milp-solver", type=str, default="GUROBI", help="CVXPY MILP solver.")
    args = parser.parse_args()

    cfg = _load_yaml(Path(args.config))
    cost_cfg = _load_yaml(Path(args.cost_config))
    if "ed_costs" not in cost_cfg:
        raise KeyError("Missing ed_costs in cost-config YAML.")

    andes.config_logger(stream_level=int(cfg.get("stream_level", 30)))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    steps = _parse_steps(args)

    ss, M_vec, D_vec = _setup_system(cfg, args.base_scale, rng)
    ng = ss.PV.n + ss.Slack.n

    Pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    Pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)

    a = _repeat_or_validate(cost_cfg["ed_costs"]["a"], ng, "ed_costs.a")
    b = _repeat_or_validate(cost_cfg["ed_costs"]["b"], ng, "ed_costs.b")
    c = _repeat_or_validate(cost_cfg["ed_costs"]["c"], ng, "ed_costs.c")

    x_features = cost_cfg.get("features", {}).get("x_features")
    x_scaler_path = cost_cfg.get("scalers", {}).get("x_scaler_path")
    x_scaler = joblib.load(x_scaler_path) if x_scaler_path else None

    ibr_idx = cost_cfg.get("ibr_idx", [0, 5, 7, 8])
    ibr_idx = np.asarray(ibr_idx, dtype=int)

    y_scaler_path = cost_cfg.get("scalers", {}).get("y_scaler_path")
    y_scaler = joblib.load(y_scaler_path) if y_scaler_path else None

    bounds_cfg = cost_cfg.get("bounds", {})
    x_min_cfg = bounds_cfg.get("x_min", [])
    x_max_cfg = bounds_cfg.get("x_max", [])
    x_min_scaled_all = None
    x_max_scaled_all = None
    if x_min_cfg and x_max_cfg:
        x_bounds = np.vstack([np.asarray(x_min_cfg, dtype=float), np.asarray(x_max_cfg, dtype=float)])
        if bounds_cfg.get("use_scaler_for_bounds", True) and x_scaler is not None:
            x_bounds = x_scaler.transform(x_bounds)
        x_min_scaled_all = x_bounds[0]
        x_max_scaled_all = x_bounds[1]
    elif bounds_cfg.get("training_data"):
        x_min_scaled_all, x_max_scaled_all, feat_cols = compute_feature_bounds_from_training_data(cost_cfg)
        if x_features and feat_cols != x_features:
            name_to_pos = {name: i for i, name in enumerate(feat_cols)}
            reorder = [name_to_pos[name] for name in x_features]
            x_min_scaled_all = x_min_scaled_all[reorder]
            x_max_scaled_all = x_max_scaled_all[reorder]

    # Load torch model once for prediction
    model_cfg = cost_cfg.get("model", {})
    state_path = Path(model_cfg.get("state_dict", ""))
    if state_path.is_dir():
        state_path = state_path / "vis_mlp_state_dict.pt"
    torch_model = _build_torch_model(model_cfg)
    state = torch.load(state_path, map_location="cpu")
    torch_model.load_state_dict(state)
    torch_model.eval()

    # Precompute scaled y bounds for MILP/convex constraints
    y_min_raw = np.asarray(cost_cfg.get("bounds", {}).get("y_min", []), dtype=float).reshape(1, -1)
    y_max_raw = np.asarray(cost_cfg.get("bounds", {}).get("y_max", []), dtype=float).reshape(1, -1)
    if y_scaler is not None and y_min_raw.size:
        y_min_scaled = y_scaler.transform(y_min_raw).reshape(-1)
        y_max_scaled = y_scaler.transform(y_max_raw).reshape(-1)
    else:
        y_min_scaled = y_min_raw.reshape(-1)
        y_max_scaled = y_max_raw.reshape(-1)

    print("Scanning step_scale values...")
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    steps_out = []
    diff_out = []
    cost_diff_out = []
    pg_baseline_out = []
    pg_nn_out = []
    pg_delta_out = []
    for step_scale in steps:

        M_agg = np.mean(np.concatenate([ss.GENROU.M.v, ss.REGCV1.M.v])).sum()
        # print(f"Rocof: {50*(sum(ss.PQ.p0.v)*step_scale-sum(ss.PQ.p0.v))/M_agg}")

        features = _build_features(
            ss,
            base_scale=args.base_scale,
            step_scale=float(step_scale),
            load_step_time=float(cfg["tds"]["load_step_time"]),
            M_vec=M_vec,
            D_vec=D_vec,
        )
        if not x_features:
            x_features = list(features.keys())

        feat_vec = np.array([features[name] for name in x_features], dtype=float).reshape(1, -1)
        feat_scaled = np.array(x_scaler.transform(feat_vec) if x_scaler is not None else feat_vec, dtype=float)
        x_val = feat_scaled.reshape(-1)

        import time
        torch.set_num_threads(1)

        t = time.time()
        state = torch.load(state_path, map_location="cpu")
        print("torch.load:", time.time()-t)

        t = time.time()
        torch_model.load_state_dict(state)
        torch_model.eval()
        print("load_state_dict:", time.time()-t)

        t = time.time()
        with torch.no_grad():
            feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
            pred_scaled = torch_model(feat_tensor).cpu().numpy().reshape(-1)
        print("forward:", time.time()-t)

        Pd = float(np.sum(ss.PQ.p0.v)) * float(step_scale)

        # Baseline ED
        prob_ed, Pg_ed = _solve_ed(
            Pd=Pd,
            Pg_min=Pg_min,
            Pg_max=Pg_max,
            a=a,
            b=b,
            c=c,
            constraints=[],
            solver=args.solver,
        )
        if prob_ed.status not in ("optimal", "optimal_inaccurate"):
            print(f"step_scale={step_scale:.4f} baseline status={prob_ed.status}")
            continue

        Pg_baseline = Pg_ed.value.copy()

        m_min, m_max = bounds_cfg["M_bounds"]
        d_min, d_max = bounds_cfg["D_bounds"]
        # Find M and D indices:
        m_names, d_names = [f"M_{i+1}" for i in range(ss.REGCV1.n)], [f"D_{i+1}" for i in range(ss.REGCV1.n)]
        name_to_idx = {name: i for i, name in enumerate(x_features)}
        m_idx, d_idx = [name_to_idx[n] for n in m_names], [name_to_idx[n] for n in d_names]

        # Scale M and D bounds
        center = x_scaler.center_[m_idx]
        scale = x_scaler.scale_[m_idx]
        m_min_scaled = (m_min - center) / scale
        m_max_scaled = (m_max - center) / scale


        center = x_scaler.center_[d_idx]
        scale = x_scaler.scale_[d_idx]
        d_min_scaled = (d_min - center) / scale
        d_max_scaled = (d_max - center) / scale
        

        # NN constraints (convex epigraph or MILP exact)
        if args.use_milp:
            if model_cfg.get("type", "MTLSharedHeads") != "MTLSharedHeads":
                raise NotImplementedError("MILP builder only supports MTLSharedHeads for now.")
            x_nn = cp.Variable(len(x_features), name="features")
            if x_min_scaled_all is not None and x_max_scaled_all is not None:
                x_min = x_min_scaled_all.copy()
                x_max = x_max_scaled_all.copy()
            else:
                x_min = x_val.copy()
                x_max = x_val.copy()
            x_min[m_idx] = m_min_scaled
            x_max[m_idx] = m_max_scaled
            x_min[d_idx] = d_min_scaled
            x_max[d_idx] = d_max_scaled
            use_partial = args.milp_mode == "last"
            y_nn, nn_constraints = _build_milp_constraints_mtlshared(
                torch_model,
                x_nn,
                x_min,
                x_max,
                binary_last_shared_and_head=use_partial,
            )
            nn_constraints = list(nn_constraints)
            nn_constraints += [x_nn >= x_min, x_nn <= x_max]
            if y_min_scaled.size and y_max_scaled.size:
                nn_constraints += [y_nn >= y_min_scaled, y_nn <= y_max_scaled]
        else:
            x_nn, y_nn, nn_constraints = build_mtlsh_convex_constraints(cost_cfg)
            nn_constraints = list(nn_constraints)

        nn_constraints.append(x_nn[m_idx] >= m_min_scaled)
        nn_constraints.append(x_nn[m_idx] <= m_max_scaled)
        nn_constraints.append(x_nn[d_idx] >= d_min_scaled)
        nn_constraints.append(x_nn[d_idx] <= d_max_scaled)

        # Add equality constraints on x for non-M/D features
        exclude = sorted(set(m_idx) | set(d_idx))
        keep_idx = [i for i in range(x_nn.shape[0]) if i not in set(exclude)]

        x_val = feat_scaled.reshape(-1)
        nn_constraints.append(x_nn[keep_idx] == x_val[keep_idx])

        Pg_nn = cp.Variable(ng)

        cost_expr_nn = a + cp.multiply(b, Pg_nn) + cp.multiply(c, cp.square(Pg_nn))
        # objective_nn = cp.Minimize(cp.sum(cost_expr_nn))

        Pg = np.array(ss.PV.p0.v.tolist() + ss.Slack.p0.v.tolist())

        # raw quantity we want in y (unscaled)
        # Difference in power output for IBRs
        p_raw = Pg_nn[ibr_idx] - Pg[ibr_idx]

        # scale it with y_scaler params for outputs 4..7
        if y_scaler is not None:
            center = y_scaler.center_[4:8]
            scale = y_scaler.scale_[4:8]
            p_scaled = (p_raw - center) / scale
        else:
            p_scaled = p_raw

        constraints_nn = list(nn_constraints) + [
            cp.sum(Pg_nn) == Pd,
            Pg_nn >= Pg_min,
            Pg_nn <= Pg_max,
            # The difference in power output is equal to y_nn outputs 4..7
            # y_nn[4:8] == p_scaled,
        ]

        slack = cp.Variable(4, nonneg=True)
        constraints_nn += [
            y_nn[4:8] - p_scaled <= slack,
            p_scaled - y_nn[4:8] <= slack,
        ]
        objective_nn = cp.Minimize(cp.sum(cost_expr_nn) + 1e3 * cp.sum(slack))


        # eps = 1e-2
        # constraints_nn += [
        #     y_nn[4:8] <= p_scaled + eps,
        #     y_nn[4:8] >= p_scaled - eps,
        # ]


        prob_nn = cp.Problem(objective_nn, constraints_nn)
        if args.use_milp:
            prob_nn.solve(solver=args.milp_solver, verbose=True)
        else:
            prob_nn.solve(solver=args.solver, verbose=False)

        if prob_nn.status not in ("optimal", "optimal_inaccurate"):
            print(f"step_scale={step_scale:.4f} nn status={prob_nn.status}")
            continue

        diff = np.linalg.norm(Pg_nn.value - Pg_baseline)
        steps_out.append(float(step_scale))
        diff_out.append(float(diff))
        cost_diff_out.append(float(prob_nn.value - prob_ed.value))
        pg_baseline_out.append(Pg_baseline.copy())
        pg_nn_out.append(Pg_nn.value.copy())
        pg_delta_out.append((Pg - Pg_nn.value).copy())
        if True: #prob_nn.value - prob_ed.value > 0.01 or diff > args.diff_tol:
            print(f"step_scale={step_scale:.4f} diff={diff:.6f}")
            print(f"value={prob_nn.value:.4f} ")
            print("Pg:", Pg)
            print("Pg_nn:", Pg_nn.value)
            print("Pg_ed:", Pg_ed.value)
            print("Diff:", Pg - Pg_nn.value)

            center = x_scaler.center_[m_idx]
            scale = x_scaler.scale_[m_idx]
            m = x_nn.value[m_idx] * scale + center

            center = x_scaler.center_[d_idx]
            scale = x_scaler.scale_[d_idx]
            d = x_nn.value[d_idx] * scale + center

            print("M:", m, x_nn.value[m_idx])
            print("D:", d, x_nn.value[d_idx])

            if y_scaler is not None:
                print("y_nn (unscaled):", y_scaler.inverse_transform(y_nn.value.reshape(1, -1)).reshape(-1))
            else:
                print("y_nn:", y_nn.value)
            print("y_nn:", y_nn.value)

            feat_scaled_flat = feat_scaled.reshape(-1)
            feat_scaled_flat[exclude] = np.array(x_nn.value, dtype=float)[exclude]
            feat_scaled = feat_scaled_flat.reshape(1, -1)
            print(feat_scaled)
            
            with torch.no_grad():
                feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
                pred_scaled = torch_model(feat_tensor).cpu().numpy().reshape(-1)
            if y_scaler is not None:
                pred_unscaled = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).reshape(-1)
            else:
                pred_unscaled = pred_scaled

            print("y_pred (scaled):", pred_scaled)
            print("y_pred (unscaled):", pred_unscaled)

    if steps_out:
        steps_arr = np.asarray(steps_out, dtype=float)
        diff_arr = np.asarray(diff_out, dtype=float)
        cost_diff_arr = np.asarray(cost_diff_out, dtype=float)
        pg_baseline_arr = np.asarray(pg_baseline_out, dtype=float)
        pg_nn_arr = np.asarray(pg_nn_out, dtype=float)
        pg_delta_arr = np.asarray(pg_delta_out, dtype=float)

        plt.figure(figsize=(6, 4))
        plt.plot(steps_arr, diff_arr, marker="o")
        plt.xlabel("step_scale")
        plt.ylabel("||Pg_baseline - Pg_nn||")
        plt.title("Dispatch difference vs step_scale")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "scan_pg_diff_norm.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(steps_arr, cost_diff_arr, marker="o")
        plt.xlabel("step_scale")
        plt.ylabel("ED cost (NN) - ED cost (baseline)")
        plt.title("Cost difference vs step_scale")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "scan_cost_diff.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 4))
        for i in range(pg_delta_arr.shape[1]):
            plt.plot(steps_arr, pg_delta_arr[:, i], label=f"Pg{i+1}")
        plt.xlabel("step_scale")
        plt.ylabel("Pg_baseline - Pg_nn")
        plt.title("Per-generator delta across step_scale")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / "scan_pg_delta_per_gen.png", dpi=150)
        plt.close()

        # Barplot per step_scale (stacked per generator)
        for idx, step in enumerate(steps_arr):
            plt.figure(figsize=(7, 4))
            colors = ["red" if i in set(ibr_idx.tolist()) else "tab:blue" for i in range(pg_delta_arr.shape[1])]
            plt.bar(np.arange(pg_delta_arr.shape[1]) + 1, pg_delta_arr[idx], color=colors)
            plt.xlabel("Generator index")
            plt.ylabel("Pg_baseline - Pg_nn")
            plt.title(f"Pg delta per generator (step_scale={step:.4f})")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_dir / f"scan_pg_delta_bar_step_{idx:02d}.png", dpi=150)
            plt.close()

        np.savez(
            plot_dir / "scan_step_scale_results.npz",
            step_scale=steps_arr,
            diff_norm=diff_arr,
            cost_diff=cost_diff_arr,
            pg_baseline=pg_baseline_arr,
            pg_nn=pg_nn_arr,
            pg_delta=pg_delta_arr,
        )

if __name__ == "__main__":
    main()
