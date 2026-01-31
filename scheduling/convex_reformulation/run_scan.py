from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence

import cvxpy as cp
import joblib
import numpy as np
import torch
import os
import andes

from scheduling.convex_reformulation.epigraph import build_epigraph_constraints
from scheduling.convex_reformulation.milp import build_milp_constraints_mtlshared
from scheduling.convex_reformulation.fixed_pattern import build_fixed_pattern_constraints_mtlshared
from scheduling.convex_reformulation.diagnostics import (
    relu_activation_pattern_mtlshared,
    plot_diff_norm,
    plot_pg_delta_per_gen,
    plot_pg_delta_bars,
    save_scan_results,
    plot_m_d_ibrs,
    plot_pred_vs_opt,
)
from scheduling.convex_reformulation.utils import (
    load_yaml,
    parse_steps,
    repeat_or_validate,
    compute_feature_bounds_from_training_data,
    build_torch_model,
    setup_system,
    build_features,
    solve_ed,
    scale_values_with_scaler,
    unscale_values_with_scaler,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")




def main() -> None:
    parser = argparse.ArgumentParser(description="Scan step_scale to find ED differences with NN convex constraints.")
    parser.add_argument("--config", default="experiments/generation.yaml", help="Path to sim YAML.")
    parser.add_argument("--cost-config", default="scheduling/mtlsh_convex.yaml", help="Path to cost YAML.")
    parser.add_argument("--base-scale", type=float, default=1.0, help="Base load scale.")
    parser.add_argument("--step-scale", type=float, default=0.9, help="Load step scale.")
    parser.add_argument("--solver", type=str, default="GUROBI", help="CVXPY solver for convex ED.")
    parser.add_argument("--diff-tol", type=float, default=1e-3, help="Norm threshold for Pg diff.")
    parser.add_argument("--plot-dir", type=str, default="experiments", help="Directory to save plots.")
    parser.add_argument("--use-milp", action="store_true", help="Use exact ReLU MILP constraints.")
    parser.add_argument("--milp-mode", type=str, default="full", choices=["full", "last"], help="MILP ReLU mode.")
    parser.add_argument("--milp-solver", type=str, default="GUROBI", help="CVXPY MILP solver.")
    parser.add_argument("--fixed-pattern", action="store_true", help="Use fixed activation pattern QP.")
    parser.add_argument("--relax-y-bounds", action="store_true", help="Relax y bounds to include torch prediction.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    cost_cfg = load_yaml(Path(args.cost_config))
    if "ed_costs" not in cost_cfg:
        raise KeyError("Missing ed_costs in cost-config YAML.")

    andes.config_logger(stream_level=int(cfg.get("stream_level", 30)))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    step_scale = float(args.step_scale)

    ss, M_vec, D_vec = setup_system(cfg, args.base_scale, rng)
    ng = ss.PV.n + ss.Slack.n

    Pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    Pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)

    a = repeat_or_validate(cost_cfg["ed_costs"]["a"], ng, "ed_costs.a")
    b = repeat_or_validate(cost_cfg["ed_costs"]["b"], ng, "ed_costs.b")
    c = repeat_or_validate(cost_cfg["ed_costs"]["c"], ng, "ed_costs.c")

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

    model_cfg = cost_cfg.get("model", {})
    state_path = Path(model_cfg.get("state_dict", ""))
    if state_path.is_dir():
        state_path = state_path / "vis_mlp_state_dict.pt"
    torch_model = build_torch_model(model_cfg)
    state = torch.load(state_path, map_location="cpu")
    torch_model.load_state_dict(state)
    torch_model.eval()

    y_min_raw = np.asarray(bounds_cfg.get("y_min", []), dtype=float).reshape(1, -1)
    y_max_raw = np.asarray(bounds_cfg.get("y_max", []), dtype=float).reshape(1, -1)
    if y_scaler is not None and y_min_raw.size:
        y_min_scaled = y_scaler.transform(y_min_raw).reshape(-1)
        y_max_scaled = y_scaler.transform(y_max_raw).reshape(-1)
    else:
        y_min_scaled = y_min_raw.reshape(-1)
        y_max_scaled = y_max_raw.reshape(-1)

    print("Solving single step_scale problem...")
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    diff_out = []
    cost_diff_out = []
    pg_baseline_out = []
    pg_nn_out = []
    pg_delta_out = []
    features = build_features(
        ss,
        base_scale=args.base_scale,
        step_scale=step_scale,
        load_step_time=float(cfg["tds"]["load_step_time"]),
        M_vec=M_vec,
        D_vec=D_vec,
    )

    if not x_features:
        x_features = list(features.keys())

    feat_vec = np.array([features[name] for name in x_features], dtype=float).reshape(1, -1)
    feat_scaled = x_scaler.transform(feat_vec) if x_scaler is not None else feat_vec
    x_val = feat_scaled.reshape(-1)

    with torch.no_grad():
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
        pred_scaled = torch_model(feat_tensor).cpu().numpy().reshape(-1)

    if y_min_scaled.size and y_max_scaled.size and args.relax_y_bounds:
        pred_scaled_arr = np.asarray(pred_scaled, dtype=float).reshape(-1)
        y_min_scaled = np.minimum(y_min_scaled.reshape(-1), pred_scaled_arr)
        y_max_scaled = np.maximum(y_max_scaled.reshape(-1), pred_scaled_arr)

    Pd = float(np.sum(ss.PQ.p0.v)) * float(step_scale)

    prob_ed, Pg_ed = solve_ed(
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
        raise RuntimeError(f"baseline status={prob_ed.status}")

    Pg_baseline = Pg_ed.value.copy()

    m_min, m_max = bounds_cfg["M_bounds"]
    d_min, d_max = bounds_cfg["D_bounds"]
    m_names = [f"M_{i+1}" for i in range(ss.REGCV1.n)]
    d_names = [f"D_{i+1}" for i in range(ss.REGCV1.n)]
    name_to_idx = {name: i for i, name in enumerate(x_features)}
    m_idx, d_idx = [name_to_idx[n] for n in m_names], [name_to_idx[n] for n in d_names]

    m_min_scaled = scale_values_with_scaler(x_scaler, m_min, m_idx)
    m_max_scaled = scale_values_with_scaler(x_scaler, m_max, m_idx)
    d_min_scaled = scale_values_with_scaler(x_scaler, d_min, d_idx)
    d_max_scaled = scale_values_with_scaler(x_scaler, d_max, d_idx)

    if model_cfg.get("type", "MTLSharedHeads") != "MTLSharedHeads":
        raise NotImplementedError("Fixed pattern only supports MTLSharedHeads for now.")
    
    x_nn = cp.Variable(len(x_features), name="features")

    if args.fixed_pattern:
        y_nn, constraints_nn = build_fixed_pattern_constraints_mtlshared(torch_model, x_nn, feat_scaled)
        constraints_nn = list(constraints_nn)
    elif args.use_milp:
        use_partial = args.milp_mode == "last"
        if x_min_scaled_all is not None and x_max_scaled_all is not None:
            x_min = x_min_scaled_all.copy()
            x_max = x_max_scaled_all.copy()
        else:
            x_min = x_val.copy()
            x_max = x_val.copy()
        x_min = np.minimum(x_min, x_val)
        x_max = np.maximum(x_max, x_val)
        x_min[m_idx] = m_min_scaled
        x_max[m_idx] = m_max_scaled
        x_min[d_idx] = d_min_scaled
        x_max[d_idx] = d_max_scaled
        y_nn, constraints_nn = build_milp_constraints_mtlshared(
            torch_model,
            x_nn,
            x_min,
            x_max,
            binary_last_shared_and_head=use_partial,
        )
        constraints_nn = list(constraints_nn)
    else:
        x_nn, y_nn, constraints_nn = build_epigraph_constraints(cost_cfg, apply_x_bounds=False, apply_y_bounds=True)
        constraints_nn = list(constraints_nn)

    # Apply bounds on x and y
    # constraints_nn += [x_nn >= x_min, x_nn <= x_max]
    if y_min_scaled.size and y_max_scaled.size:
        constraints_nn += [y_nn >= y_min_scaled, y_nn <= y_max_scaled]

    constraints_nn.append(x_nn[m_idx] >= m_min_scaled)
    constraints_nn.append(x_nn[m_idx] <= m_max_scaled)
    constraints_nn.append(x_nn[d_idx] >= d_min_scaled)
    constraints_nn.append(x_nn[d_idx] <= d_max_scaled)

    exclude = sorted(set(m_idx) | set(d_idx))
    keep_idx = [i for i in range(x_nn.shape[0]) if i not in set(exclude)]

    x_val = feat_scaled.reshape(-1)
    constraints_nn.append(x_nn[keep_idx] == x_val[keep_idx])

    Pg_nn = cp.Variable(ng)
    cost_expr_nn = a + cp.multiply(b, Pg_nn) + cp.multiply(c, cp.square(Pg_nn))
    objective_nn = cp.Minimize(cp.sum(cost_expr_nn))

    Pg = np.array(ss.PV.p0.v.tolist() + ss.Slack.p0.v.tolist())

    p_raw =  Pg_nn[ibr_idx] - Pg[ibr_idx]
    if y_scaler is not None:
        idx = np.arange(4, 8)
        if hasattr(y_scaler, "center_"):
            center = y_scaler.center_[idx]
            scale = y_scaler.scale_[idx]
            p_scaled = (p_raw - center) / scale
        elif hasattr(y_scaler, "min_"):
            scale = y_scaler.scale_[idx]
            min_ = y_scaler.min_[idx]
            p_scaled = cp.multiply(p_raw, scale) + min_
        else:
            raise AttributeError("Unsupported y_scaler; expected center_/scale_ or min_/scale_.")
    else:
        p_scaled = p_raw

    constraints_combined = list(constraints_nn) + [
        cp.sum(Pg_nn) == Pd,
        Pg_nn >= Pg_min,
        Pg_nn <= Pg_max,
        y_nn[4:8] == p_scaled, # Enforced via constraints that the output increase of the IBRs is equal to the input Pg_nn 
    ]

    if y_min_scaled.size and y_max_scaled.size:
        pred_scaled_arr = np.asarray(pred_scaled, dtype=float).reshape(-1)
        y_lo = y_min_scaled.reshape(-1)
        y_hi = y_max_scaled.reshape(-1)
        if pred_scaled_arr.shape[0] == y_lo.shape[0]:
            if np.any(pred_scaled_arr < y_lo) or np.any(pred_scaled_arr > y_hi):
                print("WARN: torch pred_scaled violates y bounds (scaled).")
                print("pred_scaled:", pred_scaled_arr)
                print("y_min_scaled:", y_lo)
                print("y_max_scaled:", y_hi)

    if x_min_scaled_all is not None and x_max_scaled_all is not None:
        x_lo = x_min_scaled_all.reshape(-1)
        x_hi = x_max_scaled_all.reshape(-1)
        if np.any(x_val < x_lo) or np.any(x_val > x_hi):
            print("WARN: fixed x_val violates x bounds (scaled).")
            print("x_val:", x_val)
            print("x_min_scaled:", x_lo)
            print("x_max_scaled:", x_hi)

    def _solve_feas(check_constraints, *, label: str):
        prob = cp.Problem(cp.Minimize(0), check_constraints)
        prob.solve(
            solver=args.milp_solver if args.use_milp else args.solver,
            verbose=False,
            reoptimize=True,
        )
        print(f"{label} status={prob.status}")
        return prob.status

    status_nn = _solve_feas(constraints_nn, label="nn_only")
    status_ed = _solve_feas(
        [cp.sum(Pg_nn) == Pd, Pg_nn >= Pg_min, Pg_nn <= Pg_max],
        label="ed_only",
    )
    status_combined = _solve_feas(constraints_combined, label="combined")
    if status_combined not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"combined status={status_combined}")

    prob_nn = cp.Problem(objective_nn, constraints_combined)
    if args.use_milp:
        prob_nn.solve(solver=args.milp_solver, verbose=True, reoptimize=True, MIPGap=0.02)
    else:
        prob_nn.solve(solver=args.solver, verbose=False, reoptimize=True)

    if prob_nn.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"nn status={prob_nn.status}")

    diff = np.linalg.norm(Pg_nn.value - Pg_baseline)
    diff_out.append(float(diff))
    cost_diff_out.append(float(prob_nn.value - prob_ed.value))
    pg_baseline_out.append(Pg_baseline.copy())
    pg_nn_out.append(Pg_nn.value.copy())
    pg_delta_out.append((Pg - Pg_nn.value).copy())

    diff_arr = np.asarray(diff_out, dtype=float)
    cost_diff_arr = np.asarray(cost_diff_out, dtype=float)
    pg_baseline_arr = np.asarray(pg_baseline_out, dtype=float)
    pg_nn_arr = np.asarray(pg_nn_out, dtype=float)
    pg_delta_arr = np.asarray(pg_delta_out, dtype=float)

    # ================================================================

    m = unscale_values_with_scaler(x_scaler, x_nn.value[m_idx], m_idx)
    d = unscale_values_with_scaler(x_scaler, x_nn.value[d_idx], d_idx)

    print("M:", m, x_nn.value[m_idx])
    print("D:", d, x_nn.value[d_idx])

    y_nn_scaled = np.asarray(y_nn.value, dtype=float).reshape(-1)
    if y_scaler is not None:
        y_nn_unscaled = y_scaler.inverse_transform(y_nn_scaled.reshape(1, -1)).reshape(-1)
        print("y_nn (unscaled):", y_nn_unscaled)
    else:
        y_nn_unscaled = y_nn_scaled
        print("y_nn:", y_nn_scaled)
    print("y_nn:", y_nn_scaled)

    feat_scaled_flat = feat_scaled.reshape(-1)
    feat_scaled_flat[exclude] = np.array(x_nn.value, dtype=float)[exclude]
    feat_scaled = feat_scaled_flat.reshape(1, -1)
    
    with torch.no_grad():
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
        pred_scaled = torch_model(feat_tensor).cpu().numpy().reshape(-1)
    if y_scaler is not None:
        pred_unscaled = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).reshape(-1)
    else:
        pred_unscaled = pred_scaled

    print("y_pred (scaled):", pred_scaled)
    print("y_pred (unscaled):", pred_unscaled)

    # ================================================================
    plot_m_d_ibrs(m, d, ibr_idx, plot_dir)
    plot_pred_vs_opt(y_nn_scaled, y_nn_unscaled, pred_scaled, pred_unscaled, plot_dir)

    plot_diff_norm(np.asarray([step_scale]), diff_arr, plot_dir)
    plot_pg_delta_per_gen(pg_delta_arr, plot_dir)
    plot_pg_delta_bars(pg_delta_arr, plot_dir, ibr_idx)
    save_scan_results(
        np.asarray([step_scale]),
        diff_arr,
        cost_diff_arr,
        pg_baseline_arr,
        pg_nn_arr,
        pg_delta_arr,
        plot_dir / "scan_step_scale_results.npz",
    )


if __name__ == "__main__":
    main()
