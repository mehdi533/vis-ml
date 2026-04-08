import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.data_utils import load_dataset, split_data
from models.models import create_model
from models.testing import compute_prediction_metrics
from models.workflow_utils import build_model_kwargs, load_feature_name_registry, load_yaml, resolve_data_config


def _resolve_run_schema(run_cfg: dict, fallback_target_cols):
    resolved_cfg = run_cfg.get("resolved", {})
    run_feature_cols = list(resolved_cfg.get("feature_cols") or [])
    run_target_cols = list(
        resolved_cfg.get("target_cols")
        or run_cfg.get("data", {}).get("target_cols")
        or fallback_target_cols
    )
    if not run_feature_cols:
        raise ValueError(
            "Run config is missing resolved.feature_cols; boundary-region evaluation "
            "needs the exact saved training feature contract."
        )
    if not run_target_cols:
        raise ValueError(
            "Run config is missing target columns; boundary-region evaluation "
            "needs the exact saved training target contract."
        )
    return run_feature_cols, run_target_cols


def _load_run_test_split(eval_csv_path: str, run_cfg: dict, split_cfg: dict, fallback_target_cols):
    run_data_cfg = resolve_data_config(run_cfg["data"])
    run_feature_cols, run_target_cols = _resolve_run_schema(run_cfg, fallback_target_cols)
    X, y, _, _ = load_dataset(
        eval_csv_path,
        target_cols=run_target_cols,
        feature_cols=run_feature_cols,
        remove_cols=run_data_cfg.get("drop_cols"),
        remove_prefixes=run_data_cfg.get("drop_prefixes"),
        ignore_missing_remove_cols=bool(run_data_cfg.get("ignore_missing_drop_cols", False)),
        missing_fill_value=run_data_cfg.get("missing_fill_value"),
    )
    _, _, X_test, _, _, y_test = split_data(
        X,
        y,
        test_size=float(split_cfg.get("test_size", 0.3)),
        val_fraction=float(split_cfg.get("val_fraction", 0.5)),
        random_state=int(split_cfg.get("random_state", 42)),
    )
    return X_test, y_test, run_feature_cols, run_target_cols


def _infer_checkpoint_input_dim(state: dict) -> int | None:
    for tensor in state.values():
        if getattr(tensor, "ndim", 0) == 2:
            return int(tensor.shape[1])
    return None


def _load_model(run_dir: Path, cfg: dict, feature_cols, target_cols):
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
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        checkpoint_in_dim = _infer_checkpoint_input_dim(state)
        raise RuntimeError(
            f"{exc}\nRun directory: {run_dir}\n"
            f"Checkpoint input dim: {checkpoint_in_dim}\n"
            f"Current evaluator input dim: {len(feature_cols)}\n"
            "This usually means the evaluator is not using the exact feature "
            "contract saved in run_config.yaml."
        ) from exc
    model.to(device)
    model.eval()
    return model, device


def _predict(model, device, X_norm: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for start in range(0, X_norm.shape[0], batch_size):
            xb = torch.as_tensor(X_norm[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.vstack(preds)


def _subset_mask_from_quantile(values: np.ndarray, quantile: float, op: str) -> np.ndarray:
    threshold = float(np.quantile(values, quantile))
    if op == "le":
        return values <= threshold, threshold
    if op == "ge":
        return values >= threshold, threshold
    raise ValueError(f"Unsupported quantile op '{op}'.")


def _metrics_rows(model_name: str, subset_name: str, n_rows: int, target_cols, y_true, y_pred, y_true_norm, y_pred_norm):
    per_target, summary = compute_prediction_metrics(y_true, y_pred, y_true_norm, y_pred_norm, target_cols)
    subset_row = {
        "model": model_name,
        "subset": subset_name,
        "n_rows": int(n_rows),
        "agg_rmse_mean": summary["agg_rmse_mean"],
        "agg_mae_mean": summary["agg_mae_mean"],
        "agg_mse_mean": summary["agg_mse_mean"],
        "agg_rmse_norm_mean": summary["agg_rmse_norm_mean"],
        "agg_mae_norm_mean": summary["agg_mae_norm_mean"],
        "max_rmse": summary["max_rmse"],
        "max_mae": summary["max_mae"],
    }
    label_rows = []
    for row in per_target:
        label_rows.append(
            {
                "model": model_name,
                "subset": subset_name,
                "n_rows": int(n_rows),
                **row,
            }
        )
    return subset_row, label_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shortlisted models on stressed/boundary subsets.")
    parser.add_argument("--config", required=True, help="Boundary-region evaluation config.")
    parser.add_argument("--subset-csv", required=True, help="Output CSV for subset-level metrics.")
    parser.add_argument("--by-label-csv", required=True, help="Output CSV for subset label-level metrics.")
    parser.add_argument("--comparison-csv", required=True, help="Output CSV comparing subset vs global error.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = resolve_data_config(cfg["data"])
    eval_csv_path = data_cfg["csv_path"]
    X, y, feature_cols, target_cols = load_dataset(
        eval_csv_path,
        target_cols=data_cfg["target_cols"],
        remove_cols=data_cfg.get("drop_cols"),
        remove_prefixes=data_cfg.get("drop_prefixes"),
        ignore_missing_remove_cols=bool(data_cfg.get("ignore_missing_drop_cols", False)),
        missing_fill_value=data_cfg.get("missing_fill_value"),
    )
    split_cfg = cfg.get("split", {})
    _, _, X_test, _, _, y_test = split_data(
        X,
        y,
        test_size=float(split_cfg.get("test_size", 0.3)),
        val_fraction=float(split_cfg.get("val_fraction", 0.5)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    for idx, target in enumerate(target_cols):
        test_df[target] = y_test[:, idx]

    sweep_dir = Path(cfg["source_sweep_dir"])
    run_df = pd.read_csv(sweep_dir / "sweep_run_summary.csv")
    subset_rows = []
    label_rows = []
    threshold_rows = []

    region_cfg = cfg.get("regions", {})
    subset_specs = []
    for spec in region_cfg.get("input_quantiles", []):
        subset_specs.append(("input", spec))
    for spec in region_cfg.get("target_quantiles", []):
        subset_specs.append(("target", spec))

    subset_masks = {"global_test": np.ones(len(test_df), dtype=bool)}
    for source_kind, spec in subset_specs:
        column = spec["column"]
        values = test_df[column].to_numpy(dtype=np.float32)
        if spec.get("use_abs", False):
            values = np.abs(values)
        mask, threshold = _subset_mask_from_quantile(values, float(spec["quantile"]), str(spec["op"]))
        subset_masks[str(spec["name"])] = mask
        threshold_rows.append(
            {
                "subset": spec["name"],
                "source": source_kind,
                "column": column,
                "quantile": float(spec["quantile"]),
                "op": spec["op"],
                "threshold": threshold,
                "n_rows": int(mask.sum()),
            }
        )

    for _, run in run_df.iterrows():
        run_dir = Path(str(run["run_dir"]))
        run_cfg = load_yaml(run_dir / "run_config.yaml")
        model_name = str(run["model"])
        X_run_test, y_run_test, run_feature_cols, run_target_cols = _load_run_test_split(
            eval_csv_path,
            run_cfg,
            split_cfg,
            target_cols,
        )
        x_scaler = joblib.load(run_dir / "x_scaler.pkl")
        y_scaler = joblib.load(run_dir / "y_scaler.pkl")
        model, device = _load_model(run_dir, run_cfg, run_feature_cols, run_target_cols)

        X_test_norm = x_scaler.transform(X_run_test)
        y_test_norm = y_scaler.transform(y_run_test)
        y_pred_norm = _predict(model, device, X_test_norm)
        y_pred = y_scaler.inverse_transform(y_pred_norm)
        y_true = y_run_test

        for subset_name, mask in subset_masks.items():
            if int(mask.sum()) == 0:
                continue
            subset_row, subset_label_rows = _metrics_rows(
                model_name,
                subset_name,
                int(mask.sum()),
                run_target_cols,
                y_true[mask],
                y_pred[mask],
                y_test_norm[mask],
                y_pred_norm[mask],
            )
            subset_rows.append(subset_row)
            label_rows.extend(subset_label_rows)

    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(args.subset_csv, index=False)
    pd.DataFrame(label_rows).to_csv(args.by_label_csv, index=False)

    global_df = subset_df[subset_df["subset"] == "global_test"][["model", "agg_rmse_mean", "agg_mae_mean"]].rename(
        columns={"agg_rmse_mean": "global_agg_rmse_mean", "agg_mae_mean": "global_agg_mae_mean"}
    )
    comparison_df = subset_df[subset_df["subset"] != "global_test"].merge(global_df, on="model", how="left")
    comparison_df["rmse_ratio_to_global"] = comparison_df["agg_rmse_mean"] / comparison_df["global_agg_rmse_mean"]
    comparison_df["mae_ratio_to_global"] = comparison_df["agg_mae_mean"] / comparison_df["global_agg_mae_mean"]
    comparison_df.to_csv(args.comparison_csv, index=False)

    thresholds_csv = Path(args.subset_csv).with_name("boundary_region_eval_thresholds.csv")
    pd.DataFrame(threshold_rows).to_csv(thresholds_csv, index=False)
    print(
        f"Saved boundary-region summaries to {args.subset_csv}, {args.by_label_csv}, "
        f"{args.comparison_csv}, and {thresholds_csv}"
    )


if __name__ == "__main__":
    main()
