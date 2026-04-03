import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TMP_ROOT = ROOT / "tmp"
if TMP_ROOT.exists() and str(TMP_ROOT) not in sys.path:
    sys.path.insert(0, str(TMP_ROOT))

from models.workflow_utils import load_yaml, resolve_data_config


@dataclass(frozen=True)
class AffineScaler:
    scale: np.ndarray
    shift: np.ndarray
    supported: bool
    note: str


def _load_model(run_dir: Path, cfg: dict, feature_cols, target_cols):
    import torch

    from models.models import create_model
    from models.workflow_utils import build_model_kwargs, load_feature_name_registry

    feature_name_registry = load_feature_name_registry(cfg["data"].get("feature_names_path"))
    model_type = cfg.get("resolved", {}).get("model_type") or cfg.get("sweep", {}).get("models", ["MLP"])[0]
    model, device = create_model(
        model_type,
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        **build_model_kwargs(
            cfg.get("model", {}),
            feature_cols,
            train_cfg=cfg.get("training", {}),
            feature_name_registry=feature_name_registry,
        ),
    )
    manifest_path = run_dir / "artifact_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        state_path = run_dir / manifest["primary_best_state_dict"]
    else:
        state_path = run_dir / "vis_mlp_state_dict_best.pt"
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _affine_scaler(scaler, n_features: int) -> AffineScaler:
    scaler_name = type(scaler).__name__
    if scaler_name == "MinMaxScaler":
        return AffineScaler(
            scale=np.asarray(scaler.scale_, dtype=float),
            shift=np.asarray(scaler.min_, dtype=float),
            supported=True,
            note="Affine scaling supported directly in the MILP input link.",
        )
    if scaler_name == "StandardScaler":
        scale = 1.0 / np.asarray(scaler.scale_, dtype=float)
        shift = -np.asarray(scaler.mean_, dtype=float) / np.asarray(scaler.scale_, dtype=float)
        return AffineScaler(
            scale=scale,
            shift=shift,
            supported=True,
            note="Affine scaling supported after rewriting x_sc = x / sigma - mu / sigma.",
        )
    if scaler_name == "RobustScaler":
        scale_attr = np.asarray(scaler.scale_, dtype=float)
        scale = 1.0 / scale_attr
        center = np.asarray(getattr(scaler, "center_", np.zeros_like(scale_attr)), dtype=float)
        shift = -center / scale_attr
        return AffineScaler(
            scale=scale,
            shift=shift,
            supported=True,
            note="Affine scaling supported after rewriting x_sc = x / iqr - center / iqr.",
        )
    if scaler_name == "IdentityScaler":
        return AffineScaler(
            scale=np.ones(n_features, dtype=float),
            shift=np.zeros(n_features, dtype=float),
            supported=True,
            note="Identity scaling.",
        )
    if scaler_name in {"Log1pScaler", "Log1pRobustScaler"}:
        return AffineScaler(
            scale=np.full(n_features, np.nan, dtype=float),
            shift=np.full(n_features, np.nan, dtype=float),
            supported=False,
            note="Nonlinear preprocessing; not directly representable by the current affine input-link formulation.",
        )
    return AffineScaler(
        scale=np.full(n_features, np.nan, dtype=float),
        shift=np.full(n_features, np.nan, dtype=float),
        supported=False,
        note=f"Unsupported scaler type for affine embedding analysis: {type(scaler).__name__}.",
    )


def _linear_layers(module):
    return [m for m in module if isinstance(m, torch.nn.Linear)]


def _extract_linear_layers(seq):
    layers = []
    for layer in _linear_layers(seq):
        w = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()
        layers.append((w, b))
    return layers


def _interval_bounds(W, b, h_min, h_max):
    W_pos = np.maximum(W, 0.0)
    W_neg = np.minimum(W, 0.0)
    z_min = W_pos @ h_min + W_neg @ h_max + b
    z_max = W_pos @ h_max + W_neg @ h_min + b
    return z_min, z_max


def _relu_stats(z_min: np.ndarray, z_max: np.ndarray) -> dict[str, float]:
    active = z_min >= 0.0
    inactive = z_max <= 0.0
    undecided = ~(active | inactive)
    abs_m = np.maximum(-z_min[undecided], z_max[undecided]) if np.any(undecided) else np.zeros(0)
    spans = (z_max - z_min)[undecided] if np.any(undecided) else np.zeros(0)
    return {
        "n_total": int(z_min.size),
        "n_active": int(np.sum(active)),
        "n_inactive": int(np.sum(inactive)),
        "n_undecided": int(np.sum(undecided)),
        "bigm_abs_max": float(np.max(abs_m)) if abs_m.size else 0.0,
        "bigm_abs_mean": float(np.mean(abs_m)) if abs_m.size else 0.0,
        "bigm_abs_p95": float(np.quantile(abs_m, 0.95)) if abs_m.size else 0.0,
        "bigm_span_max": float(np.max(spans)) if spans.size else 0.0,
        "bigm_span_mean": float(np.mean(spans)) if spans.size else 0.0,
        "bigm_span_sum": float(np.sum(spans)) if spans.size else 0.0,
    }


def _mtlsh_bigm_stats(model, x_min_sc: np.ndarray, x_max_sc: np.ndarray) -> dict[str, float]:
    layers_stats = []
    h_min = x_min_sc.copy()
    h_max = x_max_sc.copy()

    for W, b in _extract_linear_layers(model.shared):
        z_min, z_max = _interval_bounds(W, b, h_min, h_max)
        stats = _relu_stats(z_min, z_max)
        layers_stats.append(stats)
        h_min = np.maximum(0.0, z_min)
        h_max = np.maximum(0.0, z_max)

    head_stats = []
    for head in model.heads:
        hmin_head = h_min.copy()
        hmax_head = h_max.copy()
        head_layers = _extract_linear_layers(head)
        for W, b in head_layers[:-1]:
            z_min, z_max = _interval_bounds(W, b, hmin_head, hmax_head)
            stats = _relu_stats(z_min, z_max)
            layers_stats.append(stats)
            head_stats.append(stats)
            hmin_head = np.maximum(0.0, z_min)
            hmax_head = np.maximum(0.0, z_max)

    n_total = sum(s["n_total"] for s in layers_stats)
    n_active = sum(s["n_active"] for s in layers_stats)
    n_inactive = sum(s["n_inactive"] for s in layers_stats)
    n_undecided = sum(s["n_undecided"] for s in layers_stats)
    bigm_abs_max = max((s["bigm_abs_max"] for s in layers_stats), default=0.0)
    bigm_span_max = max((s["bigm_span_max"] for s in layers_stats), default=0.0)

    undecided_abs = []
    undecided_spans = []
    for stats in layers_stats:
        if stats["n_undecided"] > 0:
            undecided_abs.append((stats["bigm_abs_mean"], stats["n_undecided"]))
            undecided_spans.append((stats["bigm_span_mean"], stats["n_undecided"]))

    def _weighted_mean(items):
        if not items:
            return 0.0
        total_weight = sum(weight for _, weight in items)
        return sum(value * weight for value, weight in items) / total_weight

    return {
        "relu_total": int(n_total),
        "relu_pruned_active": int(n_active),
        "relu_pruned_inactive": int(n_inactive),
        "relu_undecided": int(n_undecided),
        "relu_pruned_fraction": float((n_active + n_inactive) / n_total) if n_total else 0.0,
        "relu_undecided_fraction": float(n_undecided / n_total) if n_total else 0.0,
        "bigm_abs_max": float(bigm_abs_max),
        "bigm_abs_mean": float(_weighted_mean(undecided_abs)),
        "bigm_span_max": float(bigm_span_max),
        "bigm_span_mean": float(_weighted_mean(undecided_spans)),
        "bigm_span_sum": float(sum(s["bigm_span_sum"] for s in layers_stats)),
    }


def _build_optimizer_box(
    cfg: dict,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import andes

    try:
        from optimization.utils import add_measurement_devices, build_features
    except ModuleNotFoundError:
        from tmp.optimization.utils import add_measurement_devices, build_features

    system_cfg = cfg["system"]
    scenario_cfg = cfg["scenario"]
    seed_cfg = cfg.get("seed", {})
    bounds_cfg = cfg["bounds"]

    ss = andes.load(system_cfg["case"], setup=False)
    ss.config.freq = float(system_cfg.get("frequency_hz", 50.0))
    add_measurement_devices(ss)

    base_scale = float(scenario_cfg["base_scale"])
    step_scale = float(scenario_cfg["step_scale"])
    load_step_time = float(scenario_cfg["load_step_time"])

    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale

    n_ibr = int(ss.REGCV1.n)
    m_seed = np.full(n_ibr, float(seed_cfg.get("M", 4.0)), dtype=float)
    d_seed = np.full(n_ibr, float(seed_cfg.get("D", 2.0)), dtype=float)
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

    feat = build_features(
        ss,
        base_scale=base_scale,
        step_scale=step_scale,
        load_step_time=load_step_time,
        M_vec=m_seed,
        D_vec=d_seed,
    )

    genrou = getattr(ss, "GENROU", None)
    regcv1 = getattr(ss, "REGCV1", None)
    genrou_pg = (
        np.asarray(getattr(genrou, "Pg", np.zeros(0)), dtype=float).reshape(-1)
        if genrou is not None
        else np.zeros(0, dtype=float)
    )
    if genrou is not None and genrou_pg.size == 0 and hasattr(genrou, "p0"):
        genrou_pg = np.asarray(genrou.p0.v, dtype=float).reshape(-1)
    regcv1_pg = (
        np.asarray(regcv1.pref.v, dtype=float).reshape(-1)
        if regcv1 is not None and hasattr(regcv1, "pref")
        else np.zeros(0, dtype=float)
    )
    for i, val in enumerate(genrou_pg, start=1):
        feat[f"P_GENROU_{i}"] = float(val)
    for i, val in enumerate(regcv1_pg, start=1):
        feat[f"P_REGCV1_{i}"] = float(val)

    missing = [name for name in feature_cols if name not in feat]
    if missing:
        preview = ", ".join(missing[:12])
        if len(missing) > 12:
            preview += ", ..."
        raise ValueError(
            "The scaler big-M diagnostic was asked to analyze a surrogate feature contract that the "
            "current optimization-side feature builder does not yet construct. "
            f"Missing {len(missing)} features. Examples: {preview}. "
            "This usually means the model was trained on the broader x_op + x_cont + x_sched schema "
            "while the optimization builder still only exposes a smaller subset. Extend the optimization "
            "feature builder first, then rerun the big-M comparison."
        )

    x_seed = np.asarray([feat[name] for name in feature_cols], dtype=float)
    x_min = x_seed.copy()
    x_max = x_seed.copy()

    name_to_idx = {name: i for i, name in enumerate(feature_cols)}
    m_idx = [name_to_idx[f"M_{i + 1}"] for i in range(n_ibr) if f"M_{i + 1}" in name_to_idx]
    d_idx = [name_to_idx[f"D_{i + 1}"] for i in range(n_ibr) if f"D_{i + 1}" in name_to_idx]
    pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)
    pg_feat_idx = [
        i for i, name in enumerate(feature_cols) if name.startswith("P_GENROU_") or name.startswith("P_REGCV1_")
    ]

    if m_idx:
        x_min[m_idx] = float(bounds_cfg["M_bounds"][0])
        x_max[m_idx] = float(bounds_cfg["M_bounds"][1])
    if d_idx:
        x_min[d_idx] = float(bounds_cfg["D_bounds"][0])
        x_max[d_idx] = float(bounds_cfg["D_bounds"][1])
    for k, idx in enumerate(pg_feat_idx[: min(len(pg_feat_idx), pg_min.size)]):
        x_min[idx] = float(pg_min[k])
        x_max[idx] = float(pg_max[k])

    control_mask = x_max > x_min + 1e-12
    return x_seed, x_min, x_max, control_mask.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scaler impact on big-M tightness for the optimizer contract.")
    parser.add_argument("--sweep-dir", required=True, help="Sweep output directory.")
    parser.add_argument("--optimization-config", required=True, help="Optimization config defining the raw input box.")
    parser.add_argument("--output-csv", required=True, help="Destination CSV.")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    run_df = pd.read_csv(sweep_dir / "sweep_run_summary.csv")
    opt_cfg = load_yaml(args.optimization_config)

    rows = []
    for _, run in run_df.iterrows():
        run_dir = Path(str(run["run_dir"]))
        run_cfg = load_yaml(run_dir / "run_config.yaml")
        data_cfg = resolve_data_config(run_cfg["data"])
        feature_cols = list(run_cfg.get("resolved", {}).get("feature_cols") or data_cfg.get("feature_cols") or [])
        target_cols = list(run_cfg.get("resolved", {}).get("target_cols") or data_cfg.get("target_cols") or [])
        x_scaler = joblib.load(run_dir / "x_scaler.pkl")
        affine = _affine_scaler(x_scaler, len(feature_cols))
        x_seed, x_min_raw, x_max_raw, control_mask = _build_optimizer_box(opt_cfg, feature_cols)

        row = {
            "run_dir": str(run_dir),
            "model": str(run["model"]),
            "scaler": str(run["scaler"]),
            "agg_rmse_mean": float(run["agg_rmse_mean"]),
            "agg_mae_mean": float(run["agg_mae_mean"]),
            "best_val_loss": float(run["best_val_loss"]),
            "n_parameters_trainable": float(run["n_parameters_trainable"]),
            "relu_units_estimate": float(run["relu_units_estimate"]),
            "n_features": int(len(feature_cols)),
            "n_targets": int(len(target_cols)),
            "affine_embedding_supported": int(affine.supported),
            "embedding_note": affine.note,
        }

        if not affine.supported:
            rows.append(row)
            continue

        x_seed_sc = x_seed * affine.scale + affine.shift
        x_min_sc = x_min_raw * affine.scale + affine.shift
        x_max_sc = x_max_raw * affine.scale + affine.shift
        lo = np.minimum(x_min_sc, x_max_sc)
        hi = np.maximum(x_min_sc, x_max_sc)
        box_width = hi - lo
        control_widths = box_width[control_mask]

        model = _load_model(run_dir, run_cfg, feature_cols, target_cols)
        if str(run["model"]) != "MTLSH":
            raise ValueError("scaler_bigm_summary.py currently supports MTLSH runs only.")
        stats = _mtlsh_bigm_stats(model, lo, hi)
        row.update(stats)
        row.update(
            {
                "control_dim": int(np.sum(control_mask)),
                "scaled_control_box_l1": float(np.sum(control_widths)) if control_widths.size else 0.0,
                "scaled_control_box_l2": float(np.linalg.norm(control_widths)) if control_widths.size else 0.0,
                "scaled_control_box_linf": float(np.max(control_widths)) if control_widths.size else 0.0,
            }
        )
        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["affine_embedding_supported", "agg_rmse_mean"], ascending=[False, True])
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved scaler big-M summary to {output_csv}")


if __name__ == "__main__":
    main()
