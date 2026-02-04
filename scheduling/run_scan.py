from __future__ import annotations

import argparse
import csv
import logging
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import torch
import andes

from scheduling.epigraph import build_epigraph_constraints
from scheduling.milp import build_milp_constraints_mtlshared
from scheduling.fixed_pattern import build_fixed_pattern_constraints_mtlshared
from scheduling.diagnostics import (
    plot_diff_norm,
    plot_pg_delta_per_gen,
    plot_pg_delta_bars,
    save_scan_results,
    plot_m_d_ibrs,
    plot_pred_vs_opt,
)
from scheduling.utils import (
    load_yaml,
    repeat_or_validate,
    build_torch_model,
    setup_system,
    build_features,
    solve_ed,
    scale_values_with_scaler,
    unscale_values_with_scaler,
    compute_x_bounds,
    compute_y_bounds,
    load_scaler,
    scale_cvxpy_values_with_scaler,
)
from scheduling.line_flow_constraints import (
    build_pandapower_net,
    compute_ptdf,
    extract_fmax_from_pandapower,
    build_injection_matrices,
    compute_net_injections,
    compute_line_flows,
    build_line_flow_constraints,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanArgs:
    config: Path
    cost_config: Path
    base_scale: float
    step_scale: float
    solver: str
    diff_tol: float
    plot_dir: Path
    use_milp: bool
    milp_mode: str
    milp_solver: str
    fixed_pattern: bool
    relax_y_bounds: bool


@dataclass(frozen=True)
class FeatureData:
    features: Dict[str, float]
    x_features: List[str]
    feat_scaled: np.ndarray
    x_val: np.ndarray


@dataclass(frozen=True)
class NNConstraintData:
    x_nn: cp.Variable
    y_nn: cp.Expression
    constraints: List
    x_val: np.ndarray
    m_idx: List[int]
    d_idx: List[int]
    keep_idx: List[int]


def _parse_args() -> ScanArgs:
    parser = argparse.ArgumentParser(description="Scan step_scale to find ED differences with NN convex constraints.")
    parser.add_argument("--config", default="experiments/generation.yaml", help="Path to sim YAML.")
    parser.add_argument("--cost-config", default="scheduling/mtlsh_convex.yaml", help="Path to cost YAML.")
    parser.add_argument("--base-scale", type=float, default=1.0, help="Base load scale.")
    parser.add_argument("--step-scale", type=float, default=0.8, help="Load step scale.")
    parser.add_argument("--solver", type=str, default="GUROBI", help="CVXPY solver for convex ED.")
    parser.add_argument("--diff-tol", type=float, default=1e-3, help="Norm threshold for Pg diff.")
    parser.add_argument("--plot-dir", type=str, default="experiments", help="Directory to save plots.")
    parser.add_argument("--use-milp", action="store_true", help="Use exact ReLU MILP constraints.")
    parser.add_argument("--milp-mode", type=str, default="full", choices=["full", "last"], help="MILP ReLU mode.")
    parser.add_argument("--milp-solver", type=str, default="GUROBI", help="CVXPY MILP solver.")
    parser.add_argument("--fixed-pattern", action="store_true", help="Use fixed activation pattern QP.")
    parser.add_argument("--relax-y-bounds", action="store_true", help="Relax y bounds to include torch prediction.")
    args = parser.parse_args()

    return ScanArgs(
        config=Path(args.config),
        cost_config=Path(args.cost_config),
        base_scale=float(args.base_scale),
        step_scale=float(args.step_scale),
        solver=str(args.solver),
        diff_tol=float(args.diff_tol),
        plot_dir=Path(args.plot_dir),
        use_milp=bool(args.use_milp),
        milp_mode=str(args.milp_mode),
        milp_solver=str(args.milp_solver),
        fixed_pattern=bool(args.fixed_pattern),
        relax_y_bounds=bool(args.relax_y_bounds),
    )


def _configure_logging(stream_level: int) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=stream_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root.setLevel(stream_level)


def _load_configs(args: ScanArgs) -> Tuple[Dict, Dict]:
    cfg = load_yaml(args.config)
    cost_cfg = load_yaml(args.cost_config)
    if "ed_costs" not in cost_cfg:
        raise KeyError("Missing ed_costs in cost-config YAML.")
    return cfg, cost_cfg


def _load_model_and_scalers(cost_cfg: Dict):
    model_cfg = cost_cfg.get("model", {})
    state_path = Path(model_cfg.get("state_dict", ""))
    if not state_path:
        raise ValueError("model.state_dict must be set in cost config.")
    if state_path.is_dir():
        state_path = state_path / "vis_mlp_state_dict.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Model state_dict not found at {state_path}")

    model = build_torch_model(model_cfg)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scalers_cfg = cost_cfg.get("scalers", {})
    x_scaler = load_scaler(scalers_cfg.get("x_scaler_path"), name="x_scaler")
    y_scaler = load_scaler(scalers_cfg.get("y_scaler_path"), name="y_scaler")

    return model, x_scaler, y_scaler


def _prepare_system(cfg: Dict, base_scale: float, rng: np.random.Generator):
    ss, M_vec, D_vec = setup_system(cfg, base_scale, rng)
    ng = ss.PV.n + ss.Slack.n
    Pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    Pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)
    return ss, M_vec, D_vec, ng, Pg_min, Pg_max


def _build_feature_data(
    ss,
    *,
    cfg: Dict,
    base_scale: float,
    step_scale: float,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    x_features: Sequence[str] | None,
    x_scaler,
) -> FeatureData:
    features = build_features(
        ss,
        base_scale=base_scale,
        step_scale=step_scale,
        load_step_time=float(cfg["tds"]["load_step_time"]),
        M_vec=M_vec,
        D_vec=D_vec,
    )

    if not x_features:
        x_features = list(features.keys())

    missing = [name for name in x_features if name not in features]
    if missing:
        raise KeyError(f"Missing feature(s) in computed features: {missing}")

    feat_vec = np.array([features[name] for name in x_features], dtype=float).reshape(1, -1)
    feat_scaled = x_scaler.transform(feat_vec) if x_scaler is not None else feat_vec
    x_val = feat_scaled.reshape(-1)

    return FeatureData(features=features, x_features=list(x_features), feat_scaled=feat_scaled, x_val=x_val)


def _predict_scaled(torch_model, feat_scaled: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
        out = torch_model(feat_tensor).cpu()
        try:
            return out.numpy().reshape(-1)
        except RuntimeError:
            return np.asarray(out.tolist(), dtype=float).reshape(-1)


def _relax_y_bounds(y_min: np.ndarray, y_max: np.ndarray, pred_scaled: np.ndarray):
    if y_min.size and y_max.size and pred_scaled.size:
        pred_scaled_arr = np.asarray(pred_scaled, dtype=float).reshape(-1)
        y_min = np.minimum(y_min.reshape(-1), pred_scaled_arr)
        y_max = np.maximum(y_max.reshape(-1), pred_scaled_arr)
    return y_min, y_max


def _compute_md_indices(ss, x_features: Sequence[str]):
    m_names = [f"M_{i+1}" for i in range(ss.REGCV1.n)]
    d_names = [f"D_{i+1}" for i in range(ss.REGCV1.n)]
    name_to_idx = {name: i for i, name in enumerate(x_features)}
    try:
        m_idx = [name_to_idx[n] for n in m_names]
        d_idx = [name_to_idx[n] for n in d_names]
    except KeyError as exc:
        raise KeyError(f"Missing required feature {exc} for M/D bounds.") from exc
    return m_idx, d_idx


def _build_nn_constraints(
    *,
    cost_cfg: Dict,
    torch_model,
    feat_scaled: np.ndarray,
    x_features: Sequence[str],
    x_val: np.ndarray,
    x_min_scaled_all: np.ndarray | None,
    x_max_scaled_all: np.ndarray | None,
    y_min_scaled: np.ndarray,
    y_max_scaled: np.ndarray,
    m_idx: Sequence[int],
    d_idx: Sequence[int],
    m_min_scaled: np.ndarray,
    m_max_scaled: np.ndarray,
    d_min_scaled: np.ndarray,
    d_max_scaled: np.ndarray,
    use_milp: bool,
    milp_mode: str,
    fixed_pattern: bool,
) -> NNConstraintData:
    if cost_cfg.get("model", {}).get("type", "MTLSharedHeads") != "MTLSharedHeads":
        raise NotImplementedError("Fixed pattern only supports MTLSharedHeads for now.")

    x_nn = cp.Variable(len(x_features), name="features")

    if fixed_pattern:
        y_nn, constraints_nn = build_fixed_pattern_constraints_mtlshared(torch_model, x_nn, feat_scaled)
        constraints_nn = list(constraints_nn)
    elif use_milp:
        use_partial = milp_mode == "last"
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
        x_nn, y_nn, constraints_nn = build_epigraph_constraints(cost_cfg, apply_x_bounds=False, apply_y_bounds=False)
        constraints_nn = list(constraints_nn)

    if y_min_scaled.size and y_max_scaled.size:
        constraints_nn += [y_nn >= y_min_scaled, y_nn <= y_max_scaled]

    constraints_nn.append(x_nn[m_idx] >= m_min_scaled)
    constraints_nn.append(x_nn[m_idx] <= m_max_scaled)
    constraints_nn.append(x_nn[d_idx] >= d_min_scaled)
    constraints_nn.append(x_nn[d_idx] <= d_max_scaled)

    exclude = sorted(set(m_idx) | set(d_idx))
    keep_idx = [i for i in range(x_nn.shape[0]) if i not in set(exclude)]

    constraints_nn.append(x_nn[keep_idx] == x_val[keep_idx])

    return NNConstraintData(
        x_nn=x_nn,
        y_nn=y_nn,
        constraints=constraints_nn,
        x_val=x_val,
        m_idx=list(m_idx),
        d_idx=list(d_idx),
        keep_idx=keep_idx,
    )


def _solver_log_path(plot_dir: Path) -> Path:
    return plot_dir / "solver_verbose.txt"


def _log_solver_header(file_obj, label: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    file_obj.write(f"\n=== {label} ({timestamp}) ===\n")
    file_obj.flush()


def _solve_feasibility(
    constraints,
    *,
    solver: str,
    label: str,
    log_path: Path | None = None,
    verbose: bool = True,
    solver_opts: dict | None = None,
) -> str:
    prob = cp.Problem(cp.Minimize(0), constraints)
    solve_kwargs = {"solver": solver, "verbose": verbose, "reoptimize": True}
    if solver_opts:
        solve_kwargs.update(solver_opts)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as f:
            _log_solver_header(f, label)
            with redirect_stdout(f), redirect_stderr(f):
                prob.solve(**solve_kwargs)
    else:
        prob.solve(**solve_kwargs)
    LOGGER.info("Feasibility check %s status=%s", label, prob.status)
    return prob.status


def _save_parameters_csv(
    plot_dir: Path,
    *,
    Pg_nn: np.ndarray,
    Pg_baseline: np.ndarray,
    m_vals: np.ndarray,
    d_vals: np.ndarray,
    ibr_idx: Sequence[int],
) -> Path:
    path = plot_dir / "scan_parameters.csv"
    ibr_set = set(int(i) for i in ibr_idx)
    fieldnames = [
        "record_type",
        "index",
        "index_1based",
        "is_ibr",
        "Pg_nn",
        "Pg_baseline",
        "Pg_delta",
        "M",
        "D",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, pg_val in enumerate(Pg_nn):
            writer.writerow(
                {
                    "record_type": "generator",
                    "index": idx,
                    "index_1based": idx + 1,
                    "is_ibr": int(idx in ibr_set),
                    "Pg_nn": float(pg_val),
                    "Pg_baseline": float(Pg_baseline[idx]),
                    "Pg_delta": float(Pg_baseline[idx] - pg_val),
                    "M": "",
                    "D": "",
                }
            )
        for idx, (m_val, d_val) in enumerate(zip(m_vals, d_vals)):
            writer.writerow(
                {
                    "record_type": "ibr",
                    "index": idx,
                    "index_1based": idx + 1,
                    "is_ibr": 1,
                    "Pg_nn": "",
                    "Pg_baseline": "",
                    "Pg_delta": "",
                    "M": float(m_val),
                    "D": float(d_val),
                }
            )
    return path


def _warn_on_bound_violations(
    *,
    pred_scaled: np.ndarray,
    y_min_scaled: np.ndarray,
    y_max_scaled: np.ndarray,
    x_val: np.ndarray,
    x_min_scaled_all: np.ndarray | None,
    x_max_scaled_all: np.ndarray | None,
):
    if y_min_scaled.size and y_max_scaled.size:
        pred_scaled_arr = np.asarray(pred_scaled, dtype=float).reshape(-1)
        y_lo = y_min_scaled.reshape(-1)
        y_hi = y_max_scaled.reshape(-1)
        if pred_scaled_arr.shape[0] == y_lo.shape[0]:
            if np.any(pred_scaled_arr < y_lo) or np.any(pred_scaled_arr > y_hi):
                LOGGER.warning("torch pred_scaled violates y bounds (scaled)")
                LOGGER.warning("pred_scaled=%s", pred_scaled_arr)
                LOGGER.warning("y_min_scaled=%s", y_lo)
                LOGGER.warning("y_max_scaled=%s", y_hi)

    if x_min_scaled_all is not None and x_max_scaled_all is not None:
        x_lo = x_min_scaled_all.reshape(-1)
        x_hi = x_max_scaled_all.reshape(-1)
        if np.any(x_val < x_lo) or np.any(x_val > x_hi):
            LOGGER.warning("fixed x_val violates x bounds (scaled)")
            LOGGER.warning("x_val=%s", x_val)
            LOGGER.warning("x_min_scaled=%s", x_lo)
            LOGGER.warning("x_max_scaled=%s", x_hi)


def run_scan(args: ScanArgs) -> None:
    cfg, cost_cfg = _load_configs(args)

    stream_level = int(cfg.get("stream_level", 30))
    _configure_logging(stream_level)
    andes.config_logger(stream_level=stream_level)

    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    ss, M_vec, D_vec, ng, Pg_min, Pg_max = _prepare_system(cfg, args.base_scale, rng)

    a = repeat_or_validate(cost_cfg["ed_costs"]["a"], ng, "ed_costs.a")
    b = repeat_or_validate(cost_cfg["ed_costs"]["b"], ng, "ed_costs.b")
    c = repeat_or_validate(cost_cfg["ed_costs"]["c"], ng, "ed_costs.c")

    x_features_cfg = cost_cfg.get("features", {}).get("x_features")
    ibr_idx = np.asarray(cost_cfg.get("ibr_idx", [0, 5, 7, 8]), dtype=int)

    torch_model, x_scaler, y_scaler = _load_model_and_scalers(cost_cfg)

    x_min_scaled_all, x_max_scaled_all = compute_x_bounds(
        cost_cfg, x_scaler=x_scaler, x_features=x_features_cfg
    )
    y_min_scaled, y_max_scaled = compute_y_bounds(cost_cfg, y_scaler=y_scaler)

    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_path = _solver_log_path(plot_dir)
    LOGGER.info("Solving single step_scale problem (step_scale=%.4f)", args.step_scale)

    feature_data = _build_feature_data(
        ss,
        cfg=cfg,
        base_scale=args.base_scale,
        step_scale=args.step_scale,
        M_vec=M_vec,
        D_vec=D_vec,
        x_features=x_features_cfg,
        x_scaler=x_scaler,
    )

    if np.any(ibr_idx < 0) or np.any(ibr_idx >= ng):
        raise ValueError(f"ibr_idx must be within [0, {ng - 1}] (got {ibr_idx})")

    if x_min_scaled_all is not None and x_min_scaled_all.size != len(feature_data.x_features):
        raise ValueError("x_min bounds size does not match number of features.")
    if x_max_scaled_all is not None and x_max_scaled_all.size != len(feature_data.x_features):
        raise ValueError("x_max bounds size does not match number of features.")

    pred_scaled = _predict_scaled(torch_model, feature_data.feat_scaled)
    if args.relax_y_bounds:
        y_min_scaled, y_max_scaled = _relax_y_bounds(y_min_scaled, y_max_scaled, pred_scaled)

    Pd = float(np.sum(ss.PQ.p0.v)) * float(args.step_scale)

    line_flow_cfg = cost_cfg.get("line_flow", {})
    line_flow_builder = None
    line_flows_ed = None
    line_flows_nn = None
    if line_flow_cfg.get("enable", False):
        source = str(line_flow_cfg.get("source", "pandapower")).lower()
        if source != "pandapower":
            raise ValueError(f"Unsupported line_flow.source='{source}' (expected 'pandapower').")

        pp_net = build_pandapower_net(ss)
        ptdf, bus_ids, _line_ids = compute_ptdf(pp_net)

        fmax = line_flow_cfg.get("line_limits")
        if fmax is None or (hasattr(fmax, "__len__") and len(fmax) == 0):
            fmax = extract_fmax_from_pandapower(pp_net)
        fmax = np.asarray(fmax, dtype=float)
        if not np.any(fmax > 0):
            raise ValueError(
                "No valid fmax found. Set line_flow.line_limits or define limits in pandapower net."
            )

        print("FMAX:", fmax)

        Cg, Cd = build_injection_matrices(ss, bus_ids=bus_ids)
        Pd_vec = np.asarray(ss.PQ.p0.v, dtype=float) * float(args.step_scale)

        def _build_line_flow_constraints(Pg_var: cp.Expression):
            injections = compute_net_injections(Cg, Pg_var, Cd, Pd_vec)
            flows = compute_line_flows(ptdf, injections)
            cons = build_line_flow_constraints(flows, fmax=fmax)
            return cons, flows

        line_flow_builder = _build_line_flow_constraints

    with log_path.open("a", encoding="utf-8") as f:
        _log_solver_header(f, "baseline_ed")
        with redirect_stdout(f), redirect_stderr(f):
            line_constraints_ed = []
            Pg_ed_var = None
            if line_flow_builder is not None:
                Pg_ed_var = cp.Variable(ng)
                line_constraints_ed, line_flows_ed = line_flow_builder(Pg_ed_var)
            prob_ed, Pg_ed = solve_ed(
                Pd=Pd,
                Pg_min=Pg_min,
                Pg_max=Pg_max,
                a=a,
                b=b,
                c=c,
                constraints=line_constraints_ed,
                solver=args.solver,
                Pg_var=Pg_ed_var,
                verbose=True,
            )
    if prob_ed.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"baseline status={prob_ed.status}")

    Pg_baseline = Pg_ed.value.copy()

    bounds_cfg = cost_cfg.get("bounds", {})
    if "M_bounds" not in bounds_cfg or "D_bounds" not in bounds_cfg:
        raise KeyError("bounds.M_bounds and bounds.D_bounds must be set in cost config.")
    m_min, m_max = bounds_cfg["M_bounds"]
    d_min, d_max = bounds_cfg["D_bounds"]

    m_idx, d_idx = _compute_md_indices(ss, feature_data.x_features)

    if x_scaler is None:
        raise ValueError("x_scaler_path is required to scale M/D bounds.")

    m_min_scaled = scale_values_with_scaler(x_scaler, m_min, m_idx)
    m_max_scaled = scale_values_with_scaler(x_scaler, m_max, m_idx)
    d_min_scaled = scale_values_with_scaler(x_scaler, d_min, d_idx)
    d_max_scaled = scale_values_with_scaler(x_scaler, d_max, d_idx)

    constraint_data = _build_nn_constraints(
        cost_cfg=cost_cfg,
        torch_model=torch_model,
        feat_scaled=feature_data.feat_scaled,
        x_features=feature_data.x_features,
        x_val=feature_data.x_val,
        x_min_scaled_all=x_min_scaled_all,
        x_max_scaled_all=x_max_scaled_all,
        y_min_scaled=y_min_scaled,
        y_max_scaled=y_max_scaled,
        m_idx=m_idx,
        d_idx=d_idx,
        m_min_scaled=m_min_scaled,
        m_max_scaled=m_max_scaled,
        d_min_scaled=d_min_scaled,
        d_max_scaled=d_max_scaled,
        use_milp=args.use_milp,
        milp_mode=args.milp_mode,
        fixed_pattern=args.fixed_pattern,
    )

    Pg_nn = cp.Variable(ng)
    cost_expr_nn = a + cp.multiply(b, Pg_nn) + cp.multiply(c, cp.square(Pg_nn))
    objective_nn = cp.Minimize(cp.sum(cost_expr_nn))

    line_constraints_nn = []
    if line_flow_builder is not None:
        line_constraints_nn, line_flows_nn = line_flow_builder(Pg_nn)

    Pg = np.array(ss.PV.p0.v.tolist() + ss.Slack.p0.v.tolist())
    p_raw = Pg_nn[ibr_idx] - Pg[ibr_idx]
    if y_scaler is not None:
        idx = np.arange(4, 8)
        p_scaled = scale_cvxpy_values_with_scaler(y_scaler, p_raw, idx)
    else:
        p_scaled = p_raw

    constraints_combined = list(constraint_data.constraints) +  line_constraints_nn + [
        cp.sum(Pg_nn) == Pd,
        Pg_nn >= Pg_min,
        Pg_nn <= Pg_max,
        constraint_data.y_nn[4:8] == p_scaled,
    ]

    _warn_on_bound_violations(
        pred_scaled=pred_scaled,
        y_min_scaled=y_min_scaled,
        y_max_scaled=y_max_scaled,
        x_val=feature_data.x_val,
        x_min_scaled_all=x_min_scaled_all,
        x_max_scaled_all=x_max_scaled_all,
    )

    _solve_feasibility(
        constraint_data.constraints,
        solver=args.milp_solver if args.use_milp else args.solver,
        label="nn_only",
        log_path=log_path,
        verbose=True,
    )
    _solve_feasibility(
        [
            cp.sum(Pg_nn) == Pd,
            Pg_nn >= Pg_min,
            Pg_nn <= Pg_max,
        ]
        + line_constraints_nn,
        solver=args.milp_solver if args.use_milp else args.solver,
        label="ed_only",
        log_path=log_path,
        verbose=True,
    )
    status_combined = _solve_feasibility(
        constraints_combined,
        solver=args.milp_solver if args.use_milp else args.solver,
        label="combined",
        log_path=log_path,
        verbose=True,
    )

    if status_combined not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"combined status={status_combined}")

    prob_nn = cp.Problem(objective_nn, constraints_combined)
    if args.use_milp:
        solver = args.milp_solver
        solver_opts = {"reoptimize": True, "MIPGap": 0.02, "verbose": True}
    else:
        solver = args.solver
        solver_opts = {"reoptimize": True, "verbose": True}
    with log_path.open("a", encoding="utf-8") as f:
        _log_solver_header(f, "nn_solution")
        with redirect_stdout(f), redirect_stderr(f):
            prob_nn.solve(solver=solver, **solver_opts)

    if prob_nn.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"nn status={prob_nn.status}")

    diff = np.linalg.norm(Pg_nn.value - Pg_baseline)
    if diff > args.diff_tol:
        LOGGER.info("Dispatch diff %.6f exceeds diff_tol %.6f", diff, args.diff_tol)

    diff_arr = np.asarray([float(diff)], dtype=float)
    cost_diff_arr = np.asarray([float(prob_nn.value - prob_ed.value)], dtype=float)
    pg_baseline_arr = np.asarray([Pg_baseline.copy()], dtype=float)
    pg_nn_arr = np.asarray([Pg_nn.value.copy()], dtype=float)
    pg_delta_arr = np.asarray([(Pg - Pg_nn.value).copy()], dtype=float)

    m = unscale_values_with_scaler(x_scaler, constraint_data.x_nn.value[m_idx], m_idx)
    d = unscale_values_with_scaler(x_scaler, constraint_data.x_nn.value[d_idx], d_idx)

    LOGGER.info("M (unscaled)=%s", m)
    LOGGER.info("D (unscaled)=%s", d)

    y_nn_scaled = np.asarray(constraint_data.y_nn.value, dtype=float).reshape(-1)
    if y_scaler is not None:
        y_nn_unscaled = y_scaler.inverse_transform(y_nn_scaled.reshape(1, -1)).reshape(-1)
    else:
        y_nn_unscaled = y_nn_scaled
    LOGGER.info("y_nn (scaled)=%s", y_nn_scaled)
    LOGGER.info("y_nn (unscaled)=%s", y_nn_unscaled)

    feat_scaled_flat = feature_data.feat_scaled.reshape(-1)
    x_nn_val = np.array(constraint_data.x_nn.value, dtype=float)
    feat_scaled_flat[constraint_data.m_idx] = x_nn_val[constraint_data.m_idx]
    feat_scaled_flat[constraint_data.d_idx] = x_nn_val[constraint_data.d_idx]
    feat_scaled = feat_scaled_flat.reshape(1, -1)

    pred_scaled = _predict_scaled(torch_model, feat_scaled)
    if y_scaler is not None:
        pred_unscaled = y_scaler.inverse_transform(pred_scaled.reshape(1, -1)).reshape(-1)
    else:
        pred_unscaled = pred_scaled

    LOGGER.info("y_pred (scaled)=%s", pred_scaled)
    LOGGER.info("y_pred (unscaled)=%s", pred_unscaled)

    if line_flows_nn is not None:
        LOGGER.info("line flows (nn)=%s", line_flows_nn.value)
    if line_flows_ed is not None:
        LOGGER.info("line flows (baseline)=%s", line_flows_ed.value)
    else:
        LOGGER.info("No line flow constraints were applied.")

    plot_m_d_ibrs(m, d, ibr_idx, plot_dir)
    plot_pred_vs_opt(y_nn_scaled, y_nn_unscaled, pred_scaled, pred_unscaled, plot_dir)

    plot_diff_norm(np.asarray([args.step_scale]), diff_arr, plot_dir)
    plot_pg_delta_per_gen(pg_delta_arr, plot_dir)
    plot_pg_delta_bars(pg_delta_arr, plot_dir, ibr_idx)
    save_scan_results(
        np.asarray([args.step_scale]),
        diff_arr,
        cost_diff_arr,
        pg_baseline_arr,
        pg_nn_arr,
        pg_delta_arr,
        plot_dir / "scan_step_scale_results.npz",
    )

    params_path = _save_parameters_csv(
        plot_dir,
        Pg_nn=Pg_nn.value,
        Pg_baseline=Pg_baseline,
        m_vals=m,
        d_vals=d,
        ibr_idx=ibr_idx,
    )
    LOGGER.info("Saved parameters CSV to %s", params_path)


def main() -> None:
    run_scan(_parse_args())


if __name__ == "__main__":
    main()
