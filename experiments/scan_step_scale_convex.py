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

from data_generation.extract_metrics import build_feature_row
from experiments.run_sim_extract_ed import _repeat_or_validate, add_measurement_devices
from scheduling.mtlsh_convex import build_mtlsh_convex_constraints


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


def _scale_bounds(bounds: Sequence[float], scaler) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float).reshape(1, -1)
    if scaler is None:
        return arr.reshape(-1)
    return scaler.transform(arr).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan step_scale to find ED differences with NN convex constraints.")
    parser.add_argument("--config", default="experiments/generation.yaml", help="Path to sim YAML.")
    parser.add_argument("--cost-config", default="scheduling/mtlsh_convex.yaml", help="Path to cost YAML.")
    parser.add_argument("--base-scale", type=float, default=1.0, help="Base load scale.")
    parser.add_argument("--steps", type=str, default="", help="Comma-separated step_scale values.")
    parser.add_argument("--step-min", type=float, default=0.6, help="Min step_scale.")
    parser.add_argument("--step-max", type=float, default=1.0, help="Max step_scale.")
    parser.add_argument("--step-num", type=int, default=9, help="Number of step_scale samples.")
    parser.add_argument("--solver", type=str, default="OSQP", help="CVXPY solver for convex ED.")
    parser.add_argument("--diff-tol", type=float, default=1e-3, help="Norm threshold for Pg diff.")
    parser.add_argument("--plot-dir", type=str, default="experiments", help="Directory to save plots.")
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
        feat_scaled = x_scaler.transform(feat_vec) if x_scaler is not None else feat_vec

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

        # NN convex constraints
        x_nn, y_nn, nn_constraints = build_mtlsh_convex_constraints(cost_cfg)
        nn_constraints = list(nn_constraints)
        nn_constraints.append(x_nn == feat_scaled.reshape(-1))

        Pg_nn = cp.Variable(ng)
        cost_expr_nn = a + cp.multiply(b, Pg_nn) + cp.multiply(c, cp.square(Pg_nn))
        objective_nn = cp.Minimize(cp.sum(cost_expr_nn))

        Pg = np.array(ss.PV.p0.v.tolist() + ss.Slack.p0.v.tolist())

        # raw quantity we want in y (unscaled)
        p_raw = Pg_nn[ibr_idx] - Pg[ibr_idx]

        # scale it with y_scaler params for outputs 4..7
        center = y_scaler.center_[4:8]
        scale = y_scaler.scale_[4:8]

        p_scaled = (p_raw - center) / scale

        constraints_nn = list(nn_constraints) + [
            cp.sum(Pg_nn) == Pd,
            Pg_nn >= Pg_min,
            Pg_nn <= Pg_max,
            y_nn[4:8] == p_scaled,
        ]

        prob_nn = cp.Problem(objective_nn, constraints_nn)
        prob_nn.solve(solver=args.solver)

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
        if prob_nn.value - prob_ed.value > 0.01 or diff > args.diff_tol:
            print(f"step_scale={step_scale:.4f} diff={diff:.6f}")
            print("Pg:", Pg)
            print("Pg_nn:", Pg_nn.value)
            print("Pg_ed:", Pg_ed.value)
            print("Diff:", Pg - Pg_nn.value)
            if y_scaler is not None:
                print("y_nn (unscaled):", y_scaler.inverse_transform(y_nn.value.reshape(1, -1)).reshape(-1))
            else:
                print("y_nn:", y_nn.value)
            print("y_nn:", y_nn.value)

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
