import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import joblib

from models.models import KANLinear, create_model
from models.workflow_utils import build_model_kwargs, load_feature_name_registry, load_yaml, resolve_data_config


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


DEFAULT_FEATURES = ["M_1", "D_1", "M_2", "D_2", "M_3", "D_3", "M_4", "D_4"]


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

    state_path = _resolve_state_dict_path(model_dir)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    scaler_path = model_dir / "x_scaler.pkl"
    x_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, feature_cols, x_scaler


def _piecewise_eval(knots: np.ndarray, coeffs: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    return np.interp(x_grid, knots, coeffs)


def _first_kan_layer(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            return name, module
    raise ValueError("No KANLinear layer found in model.")


def _inverse_feature_grid(x_scaler, feature_idx: int, x_grid_norm: np.ndarray, n_features: int) -> np.ndarray:
    if x_scaler is None or not hasattr(x_scaler, "inverse_transform"):
        return x_grid_norm
    probe = np.zeros((len(x_grid_norm), n_features), dtype=float)
    probe[:, feature_idx] = x_grid_norm
    probe_raw = x_scaler.inverse_transform(probe)
    return probe_raw[:, feature_idx]


def _plot_domain_for_feature(x_scaler, feature_idx: int, default_knots: np.ndarray) -> np.ndarray:
    if x_scaler is None:
        return default_knots
    if hasattr(x_scaler, "feature_range"):
        lo, hi = x_scaler.feature_range
        return np.linspace(float(lo), float(hi), len(default_knots))
    if hasattr(x_scaler, "center_") and hasattr(x_scaler, "scale_"):
        return np.linspace(-3.0, 3.0, len(default_knots))
    return default_knots


def _format_feature_axis_label(feature_name: str) -> str:
    if feature_name.startswith("M_"):
        return f"{feature_name} (raw value)"
    if feature_name.startswith("D_"):
        return f"{feature_name} (raw value)"
    return f"{feature_name} (original scale)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export learned KAN spline functions for thesis figures.")
    parser.add_argument("--model-dir", required=True, help="Run directory containing KAN checkpoint + run_config.")
    parser.add_argument("--max-features", type=int, default=8, help="Number of top input features to plot.")
    parser.add_argument(
        "--features",
        nargs="*",
        default=DEFAULT_FEATURES,
        help="Explicit feature names to export. Defaults to the scheduled inertia/damping channels M_1,D_1,...,M_4,D_4.",
    )
    parser.add_argument("--x-points", type=int, default=256, help="Samples used to draw each spline.")
    parser.add_argument(
        "--out-csv",
        default="results/thesis_model_results/tables/kan_learned_function_summary.csv",
        help="Summary CSV of the selected spline functions.",
    )
    parser.add_argument(
        "--out-samples-csv",
        default="results/thesis_model_results/tables/kan_learned_function_samples.csv",
        help="Dense sampled spline values used for plotting.",
    )
    parser.add_argument(
        "--out-fig",
        default="LaTeX/figures/kan_shared_learned_functions.png",
        help="Destination figure path.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model, feature_cols, x_scaler = _load_model_from_run(model_dir)
    layer_name, layer = _first_kan_layer(model)
    knots_t, coeffs_t = layer.edge_params()
    knots = np.asarray(knots_t)
    coeffs = np.asarray(coeffs_t)  # [out_dim, in_dim, grid_size]

    feature_rows = []
    for feature_idx, feature_name in enumerate(feature_cols):
        amplitudes = coeffs[:, feature_idx, :].ptp(axis=1)
        best_hidden_idx = int(np.argmax(amplitudes))
        amplitude = float(amplitudes[best_hidden_idx])
        feature_rows.append(
            {
                "layer_name": layer_name,
                "feature_idx": int(feature_idx),
                "feature_name": str(feature_name),
                "hidden_unit_idx": best_hidden_idx,
                "amplitude": amplitude,
            }
        )

    summary_all = pd.DataFrame(feature_rows)
    requested_features = [name for name in args.features if name in feature_cols]
    if requested_features:
        summary_df = (
            summary_all[summary_all["feature_name"].isin(requested_features)]
            .copy()
            .set_index("feature_name")
            .loc[requested_features]
            .reset_index()
        )
    else:
        summary_df = summary_all.sort_values("amplitude", ascending=False).head(int(args.max_features))
    x_grid_knots = np.linspace(float(knots.min()), float(knots.max()), int(args.x_points))
    sample_rows = []
    for row in summary_df.itertuples(index=False):
        x_grid = _plot_domain_for_feature(x_scaler, int(row.feature_idx), x_grid_knots)
        y_grid = _piecewise_eval(knots, coeffs[row.hidden_unit_idx, row.feature_idx, :], x_grid)
        x_grid_raw = _inverse_feature_grid(x_scaler, int(row.feature_idx), x_grid, len(feature_cols))
        summary_df.loc[summary_df["feature_name"] == row.feature_name, "x_raw_min"] = float(np.min(x_grid_raw))
        summary_df.loc[summary_df["feature_name"] == row.feature_name, "x_raw_max"] = float(np.max(x_grid_raw))
        for x_val, x_raw_val, y_val in zip(x_grid, x_grid_raw, y_grid):
            sample_rows.append(
                {
                    "feature_name": row.feature_name,
                    "hidden_unit_idx": int(row.hidden_unit_idx),
                    "x_norm": float(x_val),
                    "x_raw": float(x_raw_val),
                    "y": float(y_val),
                }
            )
    summary_df.to_csv(args.out_csv, index=False)

    samples_df = pd.DataFrame(sample_rows)
    samples_df.to_csv(args.out_samples_csv, index=False)

    n_panels = len(summary_df)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2.8 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes, summary_df.itertuples(index=False)):
        sub = samples_df[samples_df["feature_name"] == row.feature_name]
        ax.plot(sub["x_raw"], sub["y"], color="#E45756", linewidth=2.0)
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
        ax.set_title(f"{row.feature_name}\nunit {row.hidden_unit_idx}, amp={row.amplitude:.3f}", fontsize=10)
        ax.set_xlabel(_format_feature_axis_label(row.feature_name))
        ax.set_ylabel("Spline value")

    for ax in axes[n_panels:]:
        ax.axis("off")

    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved KAN summary to {args.out_csv}")
    print(f"Saved KAN samples to {args.out_samples_csv}")
    print(f"Saved KAN figure to {out_fig}")


if __name__ == "__main__":
    main()
