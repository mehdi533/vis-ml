import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.models import KANLinear, create_model
from models.utils import build_model_kwargs, load_feature_name_registry, load_yaml, resolve_data_config


DEFAULT_FEATURES = [
    "base_load_scale",
    "load_step_scale",
    "M_agg",
    "D_agg",
    "P_REGCV1_SHARE",
    "M_3",
    "D_3",
    "M_4",
]


def _resolve_state_dict_path(model_dir: Path) -> Path:
    manifest_path = model_dir / "artifact_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        for key in ("primary_best_state_dict", "primary_final_state_dict"):
            name = manifest.get(key)
            if name and (model_dir / name).exists():
                return model_dir / name
    for name in (
        "mtlgsh_kan_shared_state_dict_best.pt",
        "mtlgsh_kan_state_dict_best.pt",
        "mtlgsh_kan_shared_state_dict.pt",
        "mtlgsh_kan_state_dict.pt",
    ):
        path = model_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No KAN-compatible state dict found in {model_dir}")


def _load_model_from_run(model_dir: Path):
    run_cfg = load_yaml(model_dir / "run_config.yaml")
    resolved = run_cfg.get("resolved", {})
    model_type = resolved.get("model_type")
    feature_cols = list(resolved.get("feature_cols", []))
    target_cols = list(resolved.get("target_cols", []))
    if not model_type or not feature_cols or not target_cols:
        raise ValueError(f"run_config.yaml in {model_dir} is missing resolved model metadata.")

    data_cfg = resolve_data_config(run_cfg["data"])
    feature_name_registry = load_feature_name_registry(data_cfg.get("feature_names_path"))
    model = create_model(
        model_type,
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        **build_model_kwargs(
            run_cfg.get("model", {}),
            feature_cols,
            train_cfg=run_cfg.get("training", {}),
            feature_name_registry=feature_name_registry,
        ),
    )[0]
    state = torch.load(_resolve_state_dict_path(model_dir), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    x_scaler = joblib.load(model_dir / "x_scaler.pkl")
    return model, feature_cols, x_scaler


def _first_kan_layer(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            return name, module
    raise ValueError("No KANLinear layer found in model.")


def _inverse_feature_grid(x_scaler, feature_idx: int, x_grid_norm: np.ndarray, n_features: int) -> np.ndarray:
    probe = np.zeros((len(x_grid_norm), n_features), dtype=float)
    probe[:, feature_idx] = x_grid_norm
    probe_raw = x_scaler.inverse_transform(probe)
    return probe_raw[:, feature_idx]


def _plot_domain_for_feature(x_scaler, n_points: int) -> np.ndarray:
    if hasattr(x_scaler, "feature_range"):
        lo, hi = x_scaler.feature_range
        return np.linspace(float(lo), float(hi), int(n_points))
    return np.linspace(-1.0, 1.0, int(n_points))


def _piecewise_eval(knots: np.ndarray, coeffs: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    return np.interp(x_grid, knots, coeffs)


def _feature_category(name: str) -> str:
    if name in {"base_load_scale", "load_step_scale"}:
        return "disturbance"
    if name in {"M_agg", "D_agg", "P_REGCV1_SHARE"}:
        return "aggregate schedule"
    return "local schedule"


def _shape_metrics(x_raw: np.ndarray, y: np.ndarray) -> dict:
    dy_dx = np.gradient(y, x_raw)
    d2y_dx2 = np.gradient(dy_dx, x_raw)
    total_variation = float(np.sum(np.abs(np.diff(y))))
    end_to_end = float(np.abs(y[-1] - y[0]))
    monotonicity = end_to_end / (total_variation + 1e-12)
    slope_sign = np.sign(dy_dx)
    sign_changes = int(np.sum(np.abs(np.diff(slope_sign)) > 0))
    early_idx = x_raw <= np.quantile(x_raw, 0.35)
    late_idx = x_raw >= np.quantile(x_raw, 0.65)
    return {
        "response_span": float(np.max(y) - np.min(y)),
        "mean_abs_slope": float(np.mean(np.abs(dy_dx))),
        "max_abs_slope": float(np.max(np.abs(dy_dx))),
        "mean_abs_curvature": float(np.mean(np.abs(d2y_dx2))),
        "monotonicity_score": float(monotonicity),
        "slope_sign_changes": sign_changes,
        "early_mean": float(np.mean(y[early_idx])),
        "late_mean": float(np.mean(y[late_idx])),
        "late_minus_early": float(np.mean(y[late_idx]) - np.mean(y[early_idx])),
    }


def _axis_label(feature_name: str) -> str:
    if feature_name == "base_load_scale":
        return "base load scale"
    if feature_name == "load_step_scale":
        return "load-step scale"
    if feature_name.startswith("M_"):
        return feature_name
    if feature_name.startswith("D_"):
        return feature_name
    return feature_name


def _plot_panels(samples_df: pd.DataFrame, value_col: str, ylabel: str, out_fig: Path, color: str) -> None:
    features = list(dict.fromkeys(samples_df["feature_name"]))
    n_panels = len(features)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, feature_name in zip(axes, features):
        sub = samples_df[samples_df["feature_name"] == feature_name]
        ax.plot(sub["x_raw"], sub[value_col], color=color, linewidth=2.0)
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(feature_name, fontsize=10)
        ax.set_xlabel(_axis_label(feature_name))
        ax.set_ylabel(ylabel)
    for ax in axes[n_panels:]:
        ax.axis("off")

    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a thesis-oriented KAN spline-shape study.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--x-points", type=int, default=256)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument(
        "--out-summary-csv",
        default="results/thesis_model_results/tables/kan_spline_study_summary.csv",
    )
    parser.add_argument(
        "--out-samples-csv",
        default="results/thesis_model_results/tables/kan_spline_study_samples.csv",
    )
    parser.add_argument(
        "--out-curve-fig",
        default="LaTeX/figures/kan_spline_study_curves.png",
    )
    parser.add_argument(
        "--out-slope-fig",
        default="LaTeX/figures/kan_spline_study_slopes.png",
    )
    args = parser.parse_args()

    model, feature_cols, x_scaler = _load_model_from_run(Path(args.model_dir))
    layer_name, layer = _first_kan_layer(model)
    knots_t, coeffs_t = layer.edge_params()
    knots = np.asarray(knots_t)
    coeffs = np.asarray(coeffs_t)
    x_grid_norm = _plot_domain_for_feature(x_scaler, args.x_points)

    summary_rows = []
    sample_rows = []
    for feature_name in args.features:
        if feature_name not in feature_cols:
            continue
        feature_idx = feature_cols.index(feature_name)
        amplitudes = coeffs[:, feature_idx, :].ptp(axis=1)
        hidden_idx = int(np.argmax(amplitudes))
        y = _piecewise_eval(knots, coeffs[hidden_idx, feature_idx, :], x_grid_norm)
        x_raw = _inverse_feature_grid(x_scaler, feature_idx, x_grid_norm, len(feature_cols))
        dy_dx = np.gradient(y, x_raw)
        metrics = _shape_metrics(x_raw, y)
        summary_rows.append(
            {
                "feature_name": feature_name,
                "feature_category": _feature_category(feature_name),
                "layer_name": layer_name,
                "hidden_unit_idx": hidden_idx,
                "x_raw_min": float(np.min(x_raw)),
                "x_raw_max": float(np.max(x_raw)),
                **metrics,
            }
        )
        for x_raw_i, x_norm_i, y_i, slope_i in zip(x_raw, x_grid_norm, y, dy_dx):
            sample_rows.append(
                {
                    "feature_name": feature_name,
                    "x_raw": float(x_raw_i),
                    "x_norm": float(x_norm_i),
                    "y": float(y_i),
                    "dy_dx": float(slope_i),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    samples_df = pd.DataFrame(sample_rows)
    summary_df.to_csv(args.out_summary_csv, index=False)
    samples_df.to_csv(args.out_samples_csv, index=False)
    _plot_panels(samples_df, "y", "Spline value", Path(args.out_curve_fig), color="#E45756")
    _plot_panels(samples_df, "dy_dx", "Spline slope", Path(args.out_slope_fig), color="#4C78A8")

    print(f"Saved summary to {args.out_summary_csv}")
    print(f"Saved samples to {args.out_samples_csv}")
    print(f"Saved curve figure to {args.out_curve_fig}")
    print(f"Saved slope figure to {args.out_slope_fig}")


if __name__ == "__main__":
    main()
