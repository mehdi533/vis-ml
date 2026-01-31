from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import joblib
import numpy as np
import torch
import yaml

from models.models import MLP, MTLGroupedSharedHeads, MTLSharedHeads, SharedGroupSpec


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_steps(args) -> np.ndarray:
    if args.steps:
        return np.asarray([float(x) for x in args.steps.split(",")], dtype=float)
    return np.linspace(args.step_min, args.step_max, args.step_num, dtype=float)


def repeat_or_validate(values: Sequence[float], n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.full(n, float(arr[0]))
    if arr.size != n:
        raise ValueError(f"{name} must have length 1 or {n} (got {arr.size}).")
    return arr


def add_measurement_devices(ss):
    """Add BusROCOF + PMU at every bus (matches analyze_sim notebook)."""
    for bus in ss.Bus.as_df().idx.values:
        ss.add(
            model="BusROCOF",
            idx=f"BusROCOF_{bus}",
            name=f"BusROCOF {bus}",
            param_dict=dict(bus=bus, Tr=0.02, Tw=0.1, Tf=0.02),
        )

    existing = list(ss.PMU.as_df().bus.values) if ss.PMU.n > 0 else []
    for bus in ss.Bus.as_df().idx.values:
        if bus not in existing:
            ss.add(model="PMU", param_dict=dict(bus=bus))


def build_feature_row(
    *,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Sequence[str],
    pq_p_before: np.ndarray,
    pq_q_before: np.ndarray,
    pq_p_after: np.ndarray,
    pq_q_after: np.ndarray,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    M_agg: float,
    D_agg: float,
) -> Dict[str, float]:
    features: Dict[str, float] = {
        "base_load_scale": float(base_load_scale),
        "load_step_scale": float(load_step_scale),
        "load_step_time": float(load_step_time),
        "DELTA_PQ_tot": 0.0,
        "M_agg": float(M_agg),
        "D_agg": float(D_agg),
        "base_load_p_total": float(np.sum(pq_p_before)) if pq_p_before.any() else 0.0,
        "base_load_q_total": float(np.sum(pq_q_before)) if pq_q_before.any() else 0.0,
    }

    for i, (m_val, d_val) in enumerate(zip(M_vec, D_vec), start=1):
        features[f"M_{i}"] = float(m_val)
        features[f"D_{i}"] = float(d_val)

    delta_p_total = 0.0
    delta_q_total = 0.0
    for name, p_before, p_after, q_before, q_after in zip(
        pq_names, pq_p_before, pq_p_after, pq_q_before, pq_q_after
    ):
        dp = float(p_after - p_before)
        dq = float(q_after - q_before)
        features[f"DELTA_P_{name}"] = dp
        features[f"DELTA_Q_{name}"] = dq
        delta_p_total += dp
        delta_q_total += dq

    features["DELTA_PQ_tot"] = float(delta_p_total + delta_q_total)
    return features


def load_csv_features(
    csv_path: Path,
    *,
    feature_cols: Sequence[str] | None = None,
    drop_cols: Sequence[str] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    drops = set(drop_cols or [])

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in drops]
    else:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in {csv_path}: {missing}")

    X = df[list(feature_cols)].to_numpy(dtype=np.float32, copy=False)
    return X, list(feature_cols)


def compute_feature_bounds_from_training_data(cfg: dict):
    bounds_cfg = cfg.get("bounds", {})
    training_data = bounds_cfg.get("training_data")
    if not training_data:
        raise ValueError("bounds.training_data is required to compute bounds from data.")

    feature_cols = cfg.get("features", {}).get("x_features")
    drop_cols = bounds_cfg.get("drop_cols", [])

    X, feature_cols = load_csv_features(
        Path(training_data),
        feature_cols=feature_cols if feature_cols else None,
        drop_cols=drop_cols,
    )

    use_scaler = bool(bounds_cfg.get("use_scaler_for_bounds", True))
    x_scaler_path = cfg.get("scalers", {}).get("x_scaler_path")
    if use_scaler and x_scaler_path:
        scaler = joblib.load(x_scaler_path)
        X = scaler.transform(X)

    x_min = np.nanmin(X, axis=0)
    x_max = np.nanmax(X, axis=0)
    return x_min, x_max, feature_cols


def scale_values_with_scaler(scaler, values: np.ndarray, idx: Sequence[int] | None = None) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if idx is not None:
        vals = vals.copy()
        idx = np.asarray(idx, dtype=int)
        if hasattr(scaler, "center_"):
            vals = (vals - scaler.center_[idx]) / scaler.scale_[idx]
        elif hasattr(scaler, "min_"):
            vals = vals * scaler.scale_[idx] + scaler.min_[idx]
        else:
            raise AttributeError("Unsupported scaler; expected center_/scale_ or min_/scale_.")
        return vals
    if hasattr(scaler, "center_"):
        return (vals - scaler.center_) / scaler.scale_
    if hasattr(scaler, "min_"):
        return vals * scaler.scale_ + scaler.min_
    raise AttributeError("Unsupported scaler; expected center_/scale_ or min_/scale_.")


def unscale_values_with_scaler(scaler, values: np.ndarray, idx: Sequence[int] | None = None) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if idx is not None:
        vals = vals.copy()
        idx = np.asarray(idx, dtype=int)
        if hasattr(scaler, "center_"):
            vals = vals * scaler.scale_[idx] + scaler.center_[idx]
        elif hasattr(scaler, "min_"):
            vals = (vals - scaler.min_[idx]) / scaler.scale_[idx]
        else:
            raise AttributeError("Unsupported scaler; expected center_/scale_ or min_/scale_.")
        return vals
    if hasattr(scaler, "center_"):
        return vals * scaler.scale_ + scaler.center_
    if hasattr(scaler, "min_"):
        return (vals - scaler.min_) / scaler.scale_
    raise AttributeError("Unsupported scaler; expected center_/scale_ or min_/scale_.")


def build_torch_model(model_cfg: Dict):
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
                raise ValueError(f"Unsupported group config: {entry}")
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
            out_dim=int(model_cfg.get("out_dim", model_cfg.get("n_tasks", 1))),
            hidden_sizes=model_cfg.get("hidden_sizes"),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    else:
        raise NotImplementedError(f"Unsupported model type: {model_type}")

    return model


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _relu_epigraph(z, y):
    return [y >= 0, y >= z]


def _apply_relu_stack(h, layers, constraints, *, prefix: str):
    for idx, (w, b) in enumerate(layers):
        z = w @ h + b
        y = cp.Variable(b.shape[0], name=f"{prefix}_{idx}")
        constraints += _relu_epigraph(z, y)
        h = y
    return h


def _extract_linear_layers(seq):
    return [
        (m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
        for m in _linear_layers(seq)
    ]


def setup_system(cfg: Dict, base_scale: float, rng: np.random.Generator):
    import andes

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


def build_features(
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

    pq_p_after = pq_p_before * step_scale
    pq_q_after = pq_q_before * step_scale

    M_agg = np.mean(np.concatenate([ss.GENROU.M.v, ss.REGCV1.M.v])).sum()
    D_agg = np.mean(np.concatenate([ss.GENROU.D.v, ss.REGCV1.D.v])).sum()

    row = build_feature_row(
        base_load_scale=base_scale,
        load_step_scale=step_scale,
        load_step_time=load_step_time,
        pq_names=list(ss.PQ.name.v) if ss.PQ.n else [],
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
        pq_p_after=pq_p_after,
        pq_q_after=pq_q_after,
        M_vec=M_vec,
        D_vec=D_vec,
        M_agg=M_agg,
        D_agg=D_agg,
    )
    return row


def solve_ed(Pd, Pg_min, Pg_max, a, b, c, constraints, solver: str):
    Pg = cp.Variable(len(Pg_min))
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
