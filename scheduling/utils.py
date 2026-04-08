from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_GEN_ROOT = ROOT / "data_generation"
if DATA_GEN_ROOT.exists() and str(DATA_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_ROOT))

from extract_metrics import extract_line_metrics, extract_operating_point_snapshot, extract_x_cont, extract_x_op, extract_x_sched
from models.models import create_model


@dataclass(frozen=True)
class ConstraintSwitches:
    input: bool
    output: bool
    nn: bool
    line: bool
    n1: bool
    ed: bool


@dataclass(frozen=True)
class SolverOptions:
    name: str
    verbose: bool
    reoptimize: bool
    feasibility_checks: bool


@dataclass(frozen=True)
class PlotOptions:
    enabled: bool
    dir: Path
    m_d: bool
    pred_vs_opt: bool
    dispatch: bool
    network: bool
    network_layout: str


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_table_3_1_dispatch_cost_arrays(
    ss,
    *,
    cost_table_path: str | Path = "configs/table_3_1_dispatch_costs.yaml",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = _resolve_repo_path(cost_table_path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    rows = list(payload.get("generators") or [])
    if not rows:
        raise RuntimeError(f"Invalid Table 3.1 cost file at {path}: missing 'generators' list.")

    costs_by_bus: dict[int, dict[str, float]] = {}
    for row in rows:
        bus = int(row["bus"])
        if bus in costs_by_bus:
            raise RuntimeError(f"Duplicate generator bus entry in Table 3.1 cost file: {path} (bus {bus})")
        costs_by_bus[bus] = {
            "a": float(row["a"]),
            "b": float(row["b"]),
            "c": float(row["c"]),
            "b_r": float(row.get("b_r", 0.0)),
        }

    gen_buses = [int(v) for v in list(ss.PV.bus.v) + list(ss.Slack.bus.v)]
    a = np.zeros(len(gen_buses), dtype=float)
    b = np.zeros(len(gen_buses), dtype=float)
    c = np.zeros(len(gen_buses), dtype=float)
    b_r = np.zeros(len(gen_buses), dtype=float)
    missing: list[int] = []

    for idx, bus in enumerate(gen_buses):
        coeffs = costs_by_bus.get(bus)
        if coeffs is None:
            missing.append(bus)
            continue
        a[idx] = float(coeffs["a"])
        b[idx] = float(coeffs["b"])
        c[idx] = float(coeffs["c"])
        b_r[idx] = float(coeffs["b_r"])

    if missing:
        raise RuntimeError(
            "Missing Table 3.1 ED coefficients for generator buses: "
            + ", ".join(str(bus) for bus in missing)
        )

    return a, b, c, b_r


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("optimization")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def resolve_model_dir(model_cfg: Mapping[str, Any]) -> Path:
    raw = str(model_cfg.get("model_dir", model_cfg.get("state_dict", ""))).strip()
    if not raw:
        raise ValueError("Missing model.model_dir in config.")
    path = _resolve_repo_path(raw)
    if path.suffix:
        path = path.parent
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found at {path}")
    return path


def resolve_state_dict_path(model_cfg: Mapping[str, Any]) -> Path:
    model_dir = resolve_model_dir(model_cfg)
    candidates = [
        "vis_mlp_state_dict_best.pt",
        "mtlsh_state_dict_best.pt",
        "mlp_state_dict_best.pt",
        "state_dict_best.pt",
        "mtlsh_state_dict.pt",
        "state_dict.pt",
    ]
    for fname in candidates:
        candidate = model_dir / fname
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Model state dict not found in model_dir. "
        f"Tried: {', '.join(str(model_dir / name) for name in candidates)}"
    )


def _normalize_model_type(model_type: str) -> str:
    aliases = {
        "MTLSharedHeads": "MTLSH",
        "MTLSH": "MTLSH",
        "MLP": "MLP",
        "FICNN": "FICNN",
        "PICNN": "PICNN",
        "PICNN_MTLSH": "PICNN_MTLSH",
    }
    if model_type not in aliases:
        raise ValueError(f"Unsupported model.type='{model_type}'.")
    return aliases[model_type]


def build_model_from_cfg(model_cfg: Mapping[str, Any]):
    model_type = _normalize_model_type(str(model_cfg.get("type", "")).strip())
    out_dim = int(model_cfg.get("out_dim", model_cfg.get("n_tasks", 0)))
    if out_dim <= 0:
        raise ValueError("model.out_dim or model.n_tasks must be > 0.")

    model, _ = create_model(
        model_type=model_type,
        in_dim=int(model_cfg["in_dim"]),
        out_dim=out_dim,
        device="cpu",
        hidden_sizes=model_cfg.get("hidden_sizes"),
        shared_sizes=model_cfg.get("shared_sizes"),
        head_sizes=model_cfg.get("head_sizes"),
        dropout=float(model_cfg.get("dropout", 0.0)),
        u_feature_idx=model_cfg.get("u_feature_idx"),
        v_feature_idx=model_cfg.get("v_feature_idx"),
        activation=str(model_cfg.get("activation", "relu")),
    )
    state_dict = torch.load(resolve_state_dict_path(model_cfg), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_scalers(cfg: Mapping[str, Any]):
    model_dir = resolve_model_dir(cfg["model"])
    scalers_cfg = dict(cfg.get("scalers", {}) or {})

    x_path_raw = scalers_cfg.get("x_scaler_path")
    y_path_raw = scalers_cfg.get("y_scaler_path")
    x_path = _resolve_repo_path(x_path_raw) if x_path_raw else (model_dir / "x_scaler.pkl")
    y_path = _resolve_repo_path(y_path_raw) if y_path_raw else (model_dir / "y_scaler.pkl")
    return joblib.load(x_path), joblib.load(y_path)


def parse_constraint_switches(cfg: Mapping[str, Any]) -> ConstraintSwitches:
    c = dict(cfg.get("constraints", {}) or {})
    return ConstraintSwitches(
        input=bool(c.get("use_input", True)),
        output=bool(c.get("use_output", True)),
        nn=bool(c.get("use_nn", True)),
        line=bool(c.get("use_line", True)),
        n1=bool(c.get("use_n1", False)),
        ed=bool(c.get("use_ed", True)),
    )


def parse_solver_options(cfg: Mapping[str, Any]) -> SolverOptions:
    s = dict(cfg.get("solver", {}) or {})
    return SolverOptions(
        name=str(s.get("name", "GUROBI")),
        verbose=bool(s.get("verbose", False)),
        reoptimize=bool(s.get("reoptimize", False)),
        feasibility_checks=bool(s.get("feasibility_checks", False)),
    )


def parse_plot_options(cfg: Mapping[str, Any]) -> PlotOptions:
    p = dict(cfg.get("plots", {}) or {})
    return PlotOptions(
        enabled=bool(p.get("enabled", False)),
        dir=Path(p.get("dir", "optimization/plots")),
        m_d=bool(p.get("m_d", True)),
        pred_vs_opt=bool(p.get("pred_vs_opt", False)),
        dispatch=bool(p.get("dispatch", True)),
        network=bool(p.get("network", True)),
        network_layout=str(p.get("network_layout", "kamada")),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_optimization_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(_resolve_repo_path(path))


def add_measurement_devices(ss):
    existing_rocof = set()
    if getattr(ss, "BusROCOF", None) is not None and getattr(ss.BusROCOF, "n", 0) > 0:
        try:
            existing_rocof = {str(v) for v in list(ss.BusROCOF.idx.v)}
        except Exception:
            existing_rocof = set()

    for bus in ss.Bus.as_df().idx.values:
        idx = f"BusROCOF_{bus}"
        if idx in existing_rocof:
            continue
        ss.add(
            model="BusROCOF",
            idx=idx,
            name=f"BusROCOF {bus}",
            param_dict=dict(bus=bus, Tr=0.02, Tw=0.1, Tf=0.02),
        )

    existing = list(ss.PMU.as_df().bus.values) if ss.PMU.n > 0 else []
    for bus in ss.Bus.as_df().idx.values:
        if bus not in existing:
            ss.add(model="PMU", param_dict=dict(bus=bus))


def _sanitize_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return safe or "owner"


def resolve_load_step_target_names(
    ss,
    *,
    target_pq_names: Sequence[str] | None = None,
    target_owner_labels: Sequence[str] | None = None,
) -> set[str]:
    pq_model = getattr(ss, "PQ", None)
    if pq_model is None or int(getattr(pq_model, "n", 0)) <= 0:
        return set()

    actual_names = [str(value) for value in list(pq_model.name.v)]
    selected = set(actual_names)

    if target_pq_names:
        selected &= {str(value) for value in list(target_pq_names)}

    if target_owner_labels:
        owner_filter = {_sanitize_label(value) for value in list(target_owner_labels)}
        actual_owners = [_sanitize_label(value) for value in list(pq_model.owner.v)]
        selected &= {
            name
            for name, owner in zip(actual_names, actual_owners)
            if owner in owner_filter
        }
    return selected


def build_post_step_p_vector(
    ss,
    *,
    step_scale: float,
    target_pq_names: Sequence[str] | None = None,
    target_owner_labels: Sequence[str] | None = None,
) -> np.ndarray:
    pq_model = getattr(ss, "PQ", None)
    if pq_model is None or int(getattr(pq_model, "n", 0)) <= 0:
        return np.zeros(0, dtype=float)

    actual_names = [str(value) for value in list(pq_model.name.v)]
    p_before = np.asarray(pq_model.p0.v, dtype=float).copy()
    p_after = p_before.copy()

    explicit_targeting = bool(target_pq_names) or bool(target_owner_labels)
    targets = resolve_load_step_target_names(
        ss,
        target_pq_names=target_pq_names,
        target_owner_labels=target_owner_labels,
    )
    if not targets and not explicit_targeting:
        targets = set(actual_names)

    for idx, name in enumerate(actual_names):
        if name in targets:
            p_after[idx] = float(p_after[idx]) * float(step_scale)
    return p_after


def build_prefault_p_vector(ss) -> np.ndarray:
    pq_model = getattr(ss, "PQ", None)
    if pq_model is None or int(getattr(pq_model, "n", 0)) <= 0:
        return np.zeros(0, dtype=float)
    return np.asarray(pq_model.p0.v, dtype=float).copy()


def derive_sched_dispatch_vectors(ss) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (GENROU, REGCV1) dispatch vectors from the static dispatch stack [PV..., Slack...].
    This matches the data-generation split logic used for reserve features.
    """
    pv_p0 = getattr(getattr(getattr(ss, "PV", None), "p0", None), "v", None)
    slack_p0 = getattr(getattr(getattr(ss, "Slack", None), "p0", None), "v", None)
    dispatch = np.asarray(
        (list(pv_p0) if pv_p0 is not None else [])
        + (list(slack_p0) if slack_p0 is not None else []),
        dtype=float,
    ).reshape(-1)
    n_dispatch = int(dispatch.size)
    n_genrou = int(getattr(getattr(ss, "GENROU", None), "n", 0))
    n_regcv1 = int(getattr(getattr(ss, "REGCV1", None), "n", 0))

    ibr_positions: list[int] = []
    gen_values = getattr(getattr(getattr(ss, "REGCV1", None), "gen", None), "v", None)
    if gen_values is not None:
        for value in list(gen_values):
            try:
                pos = int(value) - 1
            except Exception:
                continue
            if 0 <= pos < n_dispatch and pos not in ibr_positions:
                ibr_positions.append(pos)
    if not ibr_positions:
        ibr_positions = list(range(min(n_regcv1, n_dispatch)))

    regcv1_pg = np.zeros(n_regcv1, dtype=float)
    for i, pos in enumerate(ibr_positions[:n_regcv1]):
        if 0 <= int(pos) < n_dispatch:
            regcv1_pg[i] = float(dispatch[int(pos)])

    ibr_set = set(int(v) for v in ibr_positions)
    genrou_positions = [i for i in range(n_dispatch) if i not in ibr_set]
    genrou_pg = np.zeros(n_genrou, dtype=float)
    for i, pos in enumerate(genrou_positions[:n_genrou]):
        if 0 <= int(pos) < n_dispatch:
            genrou_pg[i] = float(dispatch[int(pos)])

    return genrou_pg, regcv1_pg


def build_features(
    ss,
    *,
    base_scale: float,
    step_scale: float,
    load_step_time: float,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    contingency: Mapping[str, Any] | None = None,
    feature_names_path: str | None = None,
    load_step_target_pq_names: Sequence[str] | None = None,
    load_step_target_owners: Sequence[str] | None = None,
) -> dict[str, float]:
    """
    Build optimization-side input features using the same extraction blocks as data_generation.
    """
    if feature_names_path is None:
        feature_names_path = "configs/data_generation_feature_names.yaml"

    pq_names = list(ss.PQ.name.v) if getattr(ss, "PQ", None) is not None and ss.PQ.n > 0 else []
    pq_owners = (
        [_sanitize_label(owner) for owner in list(ss.PQ.owner.v)]
        if getattr(ss, "PQ", None) is not None and ss.PQ.n > 0
        else []
    )
    pq_p_before = np.asarray(ss.PQ.p0.v, dtype=float).copy() if pq_names else np.zeros(0, dtype=float)
    pq_q_before = np.asarray(ss.PQ.q0.v, dtype=float).copy() if pq_names else np.zeros(0, dtype=float)
    pq_p_after = (
        build_post_step_p_vector(
            ss,
            step_scale=float(step_scale),
            target_pq_names=load_step_target_pq_names,
            target_owner_labels=load_step_target_owners,
        )
        if pq_names
        else np.zeros(0, dtype=float)
    )
    line_uids = list(range(int(getattr(getattr(ss, "Line", None), "n", 0))))

    operating_snapshot = extract_operating_point_snapshot(ss)
    line_metrics_snapshot = extract_line_metrics(
        ss=ss,
        contingency=dict(contingency) if contingency is not None else None,
        line_uids=line_uids,
        feature_names_path=feature_names_path,
    )

    x_op = extract_x_op(
        ss=ss,
        base_load_scale=float(base_scale),
        pq_names=pq_names,
        pq_owners=pq_owners,
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
        operating_point_snapshot=operating_snapshot,
        feature_names_path=feature_names_path,
    )

    x_cont = extract_x_cont(
        ss=ss,
        contingency=dict(contingency) if contingency is not None else None,
        load_step_scale=float(step_scale),
        load_step_time=float(load_step_time),
        pq_names=pq_names,
        pq_owners=pq_owners,
        pq_p_before=pq_p_before,
        pq_p_after=pq_p_after,
        line_uids=line_uids,
        line_metrics_snapshot=line_metrics_snapshot,
        feature_names_path=feature_names_path,
    )

    genrou_pg, regcv1_pg = derive_sched_dispatch_vectors(ss)
    x_sched = extract_x_sched(
        ss=ss,
        M_vec=M_vec,
        D_vec=D_vec,
        genrou_pg=genrou_pg,
        regcv1_pg=regcv1_pg,
        feature_names_path=feature_names_path,
    )

    row: dict[str, float] = {}
    row.update(x_op)
    for section in ("load_mismatch", "line_identity", "line_flow", "line_bus", "line_severity"):
        row.update(dict(x_cont.get(section, {}) or {}))
    row.update(x_sched)
    return row
