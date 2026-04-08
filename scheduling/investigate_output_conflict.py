from __future__ import annotations

import argparse

import andes
import cvxpy as cp
import numpy as np

from scheduling.constraints import build_ed_constraints
from scheduling.problem import (
    _build_dispatch_objective_expr,
    _build_constraint_blocks,
    _build_input_block,
    _build_network_blocks,
    _build_nn_blocks,
    _build_output_block,
    _build_reserve_objective_expr,
    _build_x_seed,
    _enforce_strict_feature_policy,
    _load_feature_contract_from_model_dir,
    _load_missing_feature_reference_values,
    _resolve_feature_contingency,
    _scale_subset,
)
from scheduling.utils import (
    build_model_from_cfg,
    build_prefault_p_vector,
    load_optimization_config,
    load_scalers,
    load_table_3_1_dispatch_cost_arrays,
    parse_constraint_switches,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fresh block-level optimizer repro for scheduling.")
    p.add_argument(
        "--config",
        default="results/thesis_optimization_results/configs/base_optimization_mtlsh.yaml",
        help="Optimization config path.",
    )
    p.add_argument("--use-input", action="store_true", default=False)
    p.add_argument("--use-output", action="store_true", default=False)
    p.add_argument("--use-nn", action="store_true", default=False)
    p.add_argument("--use-line", action="store_true", default=False)
    p.add_argument("--use-n1", action="store_true", default=False)
    p.add_argument("--use-n1-redispatch", action="store_true", default=False)
    p.add_argument("--use-ed", action="store_true", default=False)
    p.add_argument(
        "--all-core",
        action="store_true",
        help="Shortcut for input + output + nn + ed.",
    )
    p.add_argument(
        "--all-full",
        action="store_true",
        help="Shortcut for input + output + nn + line + n1 + ed.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_core:
        args.use_input = True
        args.use_output = True
        args.use_nn = True
        args.use_ed = True
    if args.all_full:
        args.use_input = True
        args.use_output = True
        args.use_nn = True
        args.use_line = True
        args.use_n1 = True
        args.use_n1_redispatch = True
        args.use_ed = True

    cfg = load_optimization_config(args.config)

    ss = andes.load(cfg["system"]["case"], setup=False)
    base_scale = float(cfg["scenario"]["base_scale"])
    step_scale = float(cfg["scenario"]["step_scale"])
    load_step_time = float(cfg["scenario"]["load_step_time"])

    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale

    n_ibr = int(getattr(ss.REGCV1, "n", 0))
    m_seed = np.full(n_ibr, float(cfg["seed"]["M"]))
    d_seed = np.full(n_ibr, float(cfg["seed"]["D"]))
    ss.REGCV1.M.v = m_seed.tolist()
    ss.REGCV1.D.v = d_seed.tolist()

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0
    ss.setup()
    ss.PFlow.run()

    model = build_model_from_cfg(cfg["model"])
    x_scaler, y_scaler = load_scalers(cfg)

    feature_cfg = dict(cfg.get("features", {}) or {})
    if bool(feature_cfg.get("strict_from_system", False)):
        _enforce_strict_feature_policy(feature_cfg)

    x_features = _load_feature_contract_from_model_dir(cfg["model"])

    contingency = _resolve_feature_contingency(
        ss,
        feature_cfg=feature_cfg,
        scenario_cfg=cfg.get("scenario", {}),
    )
    fixed_feature_values = _load_missing_feature_reference_values(cfg=cfg, x_features=x_features)
    x_seed = _build_x_seed(
        ss,
        x_features=x_features,
        base_scale=base_scale,
        step_scale=step_scale,
        load_step_time=load_step_time,
        m_seed=m_seed,
        d_seed=d_seed,
        contingency=contingency,
        feature_names_path=str(
            feature_cfg.get("feature_names_path", "configs/data_generation_feature_names.yaml")
        ),
        load_step_target_pq_names=None,
        load_step_target_owners=None,
        fixed_feature_values=fixed_feature_values,
        missing_fill_value=float(cfg.get("features", {}).get("missing_fill_value", 0.0)),
        allow_missing_features=bool(cfg.get("features", {}).get("allow_missing_features", False)),
        allow_constant_fill_for_unresolved_missing=bool(
            cfg.get("features", {}).get("allow_constant_fill_for_unresolved_missing", False)
        ),
        fixed_source_override_non_sched=bool(
            cfg.get("features", {}).get("fixed_source_override_non_sched", False)
        ),
    )
    x_seed_sc = x_seed * x_scaler.scale_ + x_scaler.min_

    pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)
    pd = build_prefault_p_vector(ss)

    name_to_idx = {name: i for i, name in enumerate(x_features)}
    m_idx = [name_to_idx[f"M_{i + 1}"] for i in range(n_ibr)]
    d_idx = [name_to_idx[f"D_{i + 1}"] for i in range(n_ibr)]

    x_min_sc = x_seed_sc.copy()
    x_max_sc = x_seed_sc.copy()
    x_min_sc[m_idx] = _scale_subset(
        np.full(n_ibr, float(cfg["bounds"]["M_bounds"][0])), x_scaler, m_idx
    )
    x_max_sc[m_idx] = _scale_subset(
        np.full(n_ibr, float(cfg["bounds"]["M_bounds"][1])), x_scaler, m_idx
    )
    x_min_sc[d_idx] = _scale_subset(
        np.full(n_ibr, float(cfg["bounds"]["D_bounds"][0])), x_scaler, d_idx
    )
    x_max_sc[d_idx] = _scale_subset(
        np.full(n_ibr, float(cfg["bounds"]["D_bounds"][1])), x_scaler, d_idx
    )

    y_min_raw = np.asarray(cfg["bounds"]["y_min"], dtype=float)
    y_max_raw = np.asarray(cfg["bounds"]["y_max"], dtype=float)
    y_min_sc = y_min_raw * y_scaler.scale_ + y_scaler.min_
    y_max_sc = y_max_raw * y_scaler.scale_ + y_scaler.min_

    pg = cp.Variable(pg_min.size, name="pg")
    x = cp.Variable(len(x_features), name="x")
    y = cp.Variable(len(y_min_raw), name="y")

    shared_data = {
        "x": x,
        "y": y,
        "pg": pg,
        "x_seed_sc": x_seed_sc,
        "m_idx": m_idx,
        "d_idx": d_idx,
        "m_min_sc": x_min_sc[m_idx],
        "m_max_sc": x_max_sc[m_idx],
        "d_min_sc": x_min_sc[d_idx],
        "d_max_sc": x_max_sc[d_idx],
        "pg_feat_idx": [],
        "x_scaler": x_scaler,
        "configured_pg_link_order": None,
        "x_features": x_features,
        "n_genrou": int(getattr(ss.GENROU, "n", 0)),
        "n_regcv1": int(getattr(ss.REGCV1, "n", 0)),
        "genrou_m_fixed_arr": np.asarray([], dtype=float),
        "genrou_d_fixed_arr": np.asarray([], dtype=float),
        "pg_max": pg_max,
        "dispatch_total_constant": None,
        "model": model,
        "x_min_sc": x_min_sc,
        "x_max_sc": x_max_sc,
        "nn_mode": "milp",
        "y_min_sc": y_min_sc,
        "y_max_sc": y_max_sc,
        "pg_min": pg_min,
        "y_scaler": y_scaler,
        "ss": ss,
        "pd": pd,
        "solver_name": str(cfg.get("solver", {}).get("name", "GUROBI")),
    }

    constraints: list[cp.Constraint] = []
    if args.use_input:
        constraints += _build_input_block(cfg, shared_data)
    if args.use_nn:
        switches = parse_constraint_switches(cfg)
        nn_blocks, _ = _build_nn_blocks(cfg, switches, shared_data)
        constraints += nn_blocks["nn"]
    if args.use_output:
        constraints += _build_output_block(cfg, shared_data)
    if args.use_line or args.use_n1 or args.use_n1_redispatch:
        switches = parse_constraint_switches(cfg)
        network_blocks, _, _, _ = _build_network_blocks(
            cfg,
            switches,
            shared_data,
            logger=_DummyLogger(),
        )
        if args.use_line:
            constraints += network_blocks["line"]
        if args.use_n1:
            constraints += network_blocks["n1"]
        if args.use_n1_redispatch:
            constraints += network_blocks["n1_redispatch"]
    if args.use_ed:
        constraints += build_ed_constraints(
            pg=pg,
            pg_min=pg_min,
            pg_max=pg_max,
            pd=pd,
            step_scale=step_scale,
        )

    a, b, c, b_r = load_table_3_1_dispatch_cost_arrays(
        ss,
        cost_table_path=cfg["ed_costs"]["cost_table_path"],
    )
    ibr_idx = np.asarray(cfg["ibr"]["indices"], dtype=int).reshape(-1)
    y_ibr_idx = np.asarray(cfg.get("constraints", {}).get("y_ibr_idx", [2, 3, 4, 5]), dtype=int).reshape(-1)
    objective_dispatch = _build_dispatch_objective_expr(a=a, b=b, c=c, pg=pg)
    objective_reserve, _, _ = _build_reserve_objective_expr(
        b_r=b_r,
        pg=pg,
        pg_max=pg_max,
        y=y,
        y_scaler=y_scaler,
        ibr_idx=ibr_idx,
        y_ibr_idx=y_ibr_idx,
        enable_postcont_down_term=bool(args.use_nn),
    )
    objective = cp.Minimize(objective_dispatch + objective_reserve)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver="GUROBI", verbose=False, reoptimize=True, TimeLimit=60, MIPGap=1e-3)

    print("status", prob.status)
    print("objective", prob.value)
    print(
        "enabled_blocks",
        {
            "input": args.use_input,
            "output": args.use_output,
            "nn": args.use_nn,
            "line": args.use_line,
            "n1": args.use_n1,
            "n1_redispatch": args.use_n1_redispatch,
            "ed": args.use_ed,
        },
    )
    if x.value is not None:
        x_val = np.asarray(x.value, dtype=float).reshape(-1)
        m_raw = (x_val[m_idx] - x_scaler.min_[m_idx]) / x_scaler.scale_[m_idx]
        d_raw = (x_val[d_idx] - x_scaler.min_[d_idx]) / x_scaler.scale_[d_idx]
        print("M_raw", m_raw.tolist())
        print("D_raw", d_raw.tolist())
    if y.value is not None:
        y_sc = np.asarray(y.value, dtype=float).reshape(-1)
        y_raw = (y_sc - y_scaler.min_) / y_scaler.scale_
        print("y_raw", y_raw.tolist())
        print("y_min_raw", y_min_raw.tolist())
        print("y_max_raw", y_max_raw.tolist())
        print("viol_low", (y_raw < y_min_raw).tolist())
        print("viol_high", (y_raw > y_max_raw).tolist())


class _DummyLogger:
    def info(self, *args, **kwargs) -> None:
        return None


if __name__ == "__main__":
    main()
