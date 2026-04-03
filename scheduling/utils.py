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


def build_features(
    ss,
    *,
    base_scale: float,
    step_scale: float,
    load_step_time: float,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    feature_names_path: str | None = None,
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
    pq_p_after = pq_p_before * float(step_scale)
    line_uids = list(range(int(getattr(getattr(ss, "Line", None), "n", 0))))

    operating_snapshot = extract_operating_point_snapshot(ss)
    line_metrics_snapshot = extract_line_metrics(
        ss=ss,
        contingency=None,
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
        contingency=None,
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

    x_sched = extract_x_sched(
        ss=ss,
        M_vec=M_vec,
        D_vec=D_vec,
        feature_names_path=feature_names_path,
    )

    row: dict[str, float] = {}
    row.update(x_op)
    for section in ("load_mismatch", "line_identity", "line_flow", "line_bus", "line_severity"):
        row.update(dict(x_cont.get(section, {}) or {}))
    row.update(x_sched)
    return row
