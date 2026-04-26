# train_sweep.py
# Main training entrypoint for model/loss/scaler/seed sweep experiments.

from __future__ import annotations

import argparse
import csv
import json
import time
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from models.data_utils import (
    load_dataset,
    make_dataloaders,
    scale_data,
    split_data,
)
from models.models import create_model
from models.plotting import plot_losses, plot_scatter_per_target
from models.testing import evaluate_model
from models.training import save_model, train_model
from models.utils import (
    build_model_kwargs,
    build_override_grid,
    checkpoint_filenames,
    clone_cfg_with_overrides,
    format_run_value,
    load_feature_name_registry,
    load_yaml,
    normalize_arg_list,
    resolve_data_config,
    sanitize_name,
    write_yaml,
)


# -----------------------------
# Sweep / logging helpers
# -----------------------------

def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _describe_model(model) -> Dict[str, object]:
    n_parameters_total = int(sum(param.numel() for param in model.parameters()))
    n_parameters_trainable = int(
        sum(param.numel() for param in model.parameters() if param.requires_grad)
    )
    return {
        "model_class": model.__class__.__name__,
        "n_parameters_total": n_parameters_total,
        "n_parameters_trainable": n_parameters_trainable,
    }


def _sum_int_list(values) -> int:
    return int(sum(int(v) for v in (values or [])))


def _estimate_relu_units(model_type: str, model_cfg: Dict, n_targets: int) -> Optional[int]:
    if model_type == "MLP":
        return _sum_int_list(model_cfg.get("hidden_sizes"))
    if model_type in {"FICNN", "PICNN"}:
        return _sum_int_list(model_cfg.get("hidden_sizes"))
    if model_type in {"MTLSH", "PICNN_MTLSH"}:
        return _sum_int_list(model_cfg.get("shared_sizes")) + n_targets * _sum_int_list(
            model_cfg.get("head_sizes")
        )
    if model_type in {"MTLGSH", "MTLGSH_ATT"}:
        group_count = len(model_cfg.get("group_head_indices") or [])
        return (
            _sum_int_list(model_cfg.get("shared_sizes"))
            + group_count * _sum_int_list(model_cfg.get("group_shared_sizes"))
            + n_targets * _sum_int_list(model_cfg.get("head_sizes"))
        )
    if model_type in {"MTLGSH_KAN_SHARED", "MTLGSH_KAN"}:
        return None
    return None


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _override_fields(prefix: str, overrides: Dict[str, object]) -> Dict[str, object]:
    return {f"{prefix}_{key}": overrides[key] for key in sorted(overrides.keys())}


def _override_fieldnames(prefix: str, override_list: Sequence[Dict]) -> list[str]:
    keys = sorted({key for overrides in override_list for key in overrides.keys()})
    return [f"{prefix}_{key}" for key in keys]


def _build_plot_context(run_cfg: Dict) -> Dict[str, object]:
    plotting_cfg = dict(run_cfg.get("plotting", {}) or {})
    system_cfg = dict(run_cfg.get("system", {}) or {})
    data_cfg = dict(run_cfg.get("data", {}) or {})

    context: Dict[str, object] = {}
    if plotting_cfg.get("nominal_frequency_hz") is not None:
        context["nominal_frequency_hz"] = plotting_cfg.get("nominal_frequency_hz")
    elif system_cfg.get("nominal_frequency_hz") is not None:
        context["nominal_frequency_hz"] = system_cfg.get("nominal_frequency_hz")

    if plotting_cfg.get("system_base_mva") is not None:
        context["system_base_mva"] = plotting_cfg.get("system_base_mva")
    elif system_cfg.get("system_base_mva") is not None:
        context["system_base_mva"] = system_cfg.get("system_base_mva")

    if plotting_cfg.get("ibr_device_base_mva") is not None:
        context["ibr_device_base_mva"] = plotting_cfg.get("ibr_device_base_mva")

    if plotting_cfg.get("case_path") is not None:
        context["case_path"] = plotting_cfg.get("case_path")
    elif data_cfg.get("case_path") is not None:
        context["case_path"] = data_cfg.get("case_path")
    elif system_cfg.get("case") is not None:
        context["case_path"] = system_cfg.get("case")

    return context


# -----------------------------
# Single-run execution
# -----------------------------

def train_one(
    cfg: Dict,
    run_dir: Path,
    *,
    model_type: str,
    loss_type: str,
    scaler_type: str,
    seed: int,
    model_overrides: Optional[Dict] = None,
    train_overrides: Optional[Dict] = None,
    data_overrides: Optional[Dict] = None,
):
    run_cfg = clone_cfg_with_overrides(
        cfg,
        model_overrides=model_overrides,
        training_overrides=train_overrides,
        data_overrides=data_overrides,
    )
    _set_seed(seed)

    data_cfg = resolve_data_config(run_cfg["data"])
    feature_name_registry = load_feature_name_registry(data_cfg.get("feature_names_path"))
    targets = normalize_arg_list(data_cfg.get("target_cols"))
    feature_cols_cfg = normalize_arg_list(data_cfg.get("feature_cols"))
    drops = normalize_arg_list(data_cfg.get("drop_cols"))
    drop_prefixes = normalize_arg_list(data_cfg.get("drop_prefixes"), default=[])

    X, y, feature_cols, target_cols = load_dataset(
        data_cfg["csv_path"],
        target_cols=targets,
        feature_cols=feature_cols_cfg,
        allowed_feature_cols=data_cfg.get("allowed_feature_cols"),
        allowed_feature_prefixes=data_cfg.get("allowed_feature_prefixes"),
        remove_cols=drops,
        remove_prefixes=drop_prefixes,
        ignore_missing_remove_cols=bool(data_cfg.get("ignore_missing_drop_cols", False)),
        missing_fill_value=data_cfg.get("missing_fill_value"),
    )

    split_cfg = run_cfg.get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        test_size=float(split_cfg.get("test_size", 0.3)),
        val_fraction=float(split_cfg.get("val_fraction", 0.5)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    (
        X_train_n,
        X_val_n,
        X_test_n,
        y_train_n,
        y_val_n,
        y_test_n,
        y_scaler,
    ) = scale_data(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        x_scaler_path=str(run_dir / "x_scaler.pkl"),
        y_scaler_path=str(run_dir / "y_scaler.pkl"),
        scaler_type=scaler_type,
    )

    train_cfg = dict(run_cfg["training"])
    (
        train_loader,
        val_loader,
        test_loader,
        train_ds,
        val_ds,
        test_ds,
    ) = make_dataloaders(
        X_train_n,
        X_val_n,
        X_test_n,
        y_train_n,
        y_val_n,
        y_test_n,
        batch_size_train=int(train_cfg.get("batch_train", 64)),
        batch_size_eval=int(train_cfg.get("batch_eval", 128)),
    )

    model_cfg = run_cfg.get("model", {})
    model_kwargs = build_model_kwargs(
        model_cfg,
        feature_cols,
        train_cfg=train_cfg,
        feature_name_registry=feature_name_registry,
    )
    model, device = create_model(
        model_type,
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        **model_kwargs,
    )

    run_cfg["resolved"] = {
        "model_type": model_type,
        "loss_type": loss_type,
        "scaler_type": scaler_type,
        "seed": int(seed),
        "feature_cols": list(feature_cols),
        "target_cols": list(target_cols),
    }
    write_yaml(run_dir / "run_config.yaml", run_cfg)

    model_txt = run_dir / "model.txt"
    model_txt.write_text(str(model), encoding="utf-8")

    model_stats = _describe_model(model)
    relu_units_estimate = _estimate_relu_units(model_type, model_cfg, len(target_cols))
    model_stats.update(
        {
            "n_features": int(len(feature_cols)),
            "n_targets": int(len(target_cols)),
            "target_cols": list(target_cols),
            "relu_units_estimate": relu_units_estimate,
        }
    )
    _write_json(run_dir / "model_stats.json", model_stats)
    checkpoint_names = checkpoint_filenames(model_type)
    artifact_manifest = {
        "model_type": model_type,
        "checkpoint_files": checkpoint_names,
        "primary_best_state_dict": checkpoint_names["best_state_dict"],
        "primary_final_state_dict": checkpoint_names["final_state_dict"],
    }
    _write_json(run_dir / "artifact_manifest.json", artifact_manifest)

    train_started = time.perf_counter()
    model, train_losses, train_eval_losses, val_losses = train_model(
        model,
        device,
        train_loader,
        val_loader,
        train_ds,
        val_ds,
        str(run_dir),
        n_epochs=int(train_cfg.get("epochs", 200)),
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        loss_type=loss_type,
        ce_weights=list(train_cfg.get("loss_weights", [])),
        use_lr_scheduler=bool(train_cfg.get("use_lr_scheduler", False)),
        lr_scheduler_patience=int(train_cfg.get("lr_scheduler_patience", 10)),
        lr_scheduler_factor=float(train_cfg.get("lr_scheduler_factor", 0.1)),
        lr_scheduler_min_lr=float(train_cfg.get("lr_scheduler_min_lr", 1e-6)),
        best_model_path=str(run_dir / checkpoint_names["best_state_dict"]),
    )
    train_wall_time_sec = float(time.perf_counter() - train_started)

    eval_started = time.perf_counter()
    (
        y_true,
        y_pred,
        y_true_norm,
        y_pred_norm,
        rmse,
        rmse_norm,
        metrics_by_target,
        metrics_summary,
    ) = evaluate_model(
        model,
        device,
        test_loader,
        y_scaler,
        target_cols,
        str(run_dir),
        save_predictions=bool(train_cfg.get("save_test_predictions", False)),
        return_metrics=True,
    )
    eval_wall_time_sec = float(time.perf_counter() - eval_started)
    test_mse = float(metrics_summary["agg_mse_norm_mean"])

    plot_losses(
        train_losses,
        val_losses,
        test_mse,
        train_eval_losses=train_eval_losses,
        out_path=str(run_dir / "loss_curve.png"),
    )
    plot_scatter_per_target(
        y_true,
        y_pred,
        target_cols,
        out_dir=str(run_dir),
        plot_context=_build_plot_context(run_cfg),
    )
    save_model(model, path=str(run_dir / checkpoint_names["final_state_dict"]))

    label_rows = []
    for row in metrics_by_target:
        metric_row = {
            "run_dir": str(run_dir),
            "model": model_type,
            "loss": loss_type,
            "scaler": scaler_type,
            "seed": int(seed),
            "label": row["label"],
            "rmse": row["rmse"],
            "mae": row["mae"],
            "mse": row["mse"],
            "raw_unit": row["raw_unit"],
            "display_unit": row["display_unit"],
            "display_scale": row["display_scale"],
            "rmse_display": row["rmse_display"],
            "mae_display": row["mae_display"],
            "mse_display": row["mse_display"],
            "rmse_norm": row["rmse_norm"],
            "mae_norm": row["mae_norm"],
            "mse_norm": row["mse_norm"],
            "agg_rmse_mean": metrics_summary["agg_rmse_mean"],
            "agg_mae_mean": metrics_summary["agg_mae_mean"],
            "agg_rmse_norm_mean": metrics_summary["agg_rmse_norm_mean"],
            "agg_mae_norm_mean": metrics_summary["agg_mae_norm_mean"],
            "test_mse": test_mse,
        }
        metric_row.update(_override_fields("model", model_overrides or {}))
        metric_row.update(_override_fields("train", train_overrides or {}))
        metric_row.update(_override_fields("data", data_overrides or {}))
        label_rows.append(metric_row)

    run_row = {
        "run_dir": str(run_dir),
        "model": model_type,
        "loss": loss_type,
        "scaler": scaler_type,
        "seed": int(seed),
        "n_features": int(len(feature_cols)),
        "n_targets": int(len(target_cols)),
        "agg_rmse_mean": metrics_summary["agg_rmse_mean"],
        "agg_mae_mean": metrics_summary["agg_mae_mean"],
        "agg_mse_mean": metrics_summary["agg_mse_mean"],
        "agg_rmse_norm_mean": metrics_summary["agg_rmse_norm_mean"],
        "agg_mae_norm_mean": metrics_summary["agg_mae_norm_mean"],
        "agg_mse_norm_mean": metrics_summary["agg_mse_norm_mean"],
        "max_rmse": metrics_summary["max_rmse"],
        "max_mae": metrics_summary["max_mae"],
        "best_val_loss": float(min(val_losses)) if val_losses else "",
        "final_val_loss": float(val_losses[-1]) if val_losses else "",
        "final_train_loss": float(train_losses[-1]) if train_losses else "",
        "final_train_eval_loss": float(train_eval_losses[-1]) if train_eval_losses else "",
        "train_wall_time_sec": train_wall_time_sec,
        "eval_wall_time_sec": eval_wall_time_sec,
        "total_wall_time_sec": train_wall_time_sec + eval_wall_time_sec,
        "n_parameters_total": model_stats["n_parameters_total"],
        "n_parameters_trainable": model_stats["n_parameters_trainable"],
        "relu_units_estimate": model_stats["relu_units_estimate"],
    }
    run_row.update(_override_fields("model", model_overrides or {}))
    run_row.update(_override_fields("train", train_overrides or {}))
    run_row.update(_override_fields("data", data_overrides or {}))

    return label_rows, [run_row]


# -----------------------------
# CLI entrypoint
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple models from a YAML sweep.")
    parser.add_argument("config_path", nargs="?", help="Optional positional path to sweep config.")
    parser.add_argument("--config", default="configs/model/train_sweep.yaml", help="Path to sweep config.")
    args = parser.parse_args()

    config_path = args.config_path or args.config
    cfg = load_yaml(config_path)
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    sweep = cfg.get("sweep", {})
    models = sweep.get("models", [])
    losses = sweep.get("losses", [])
    scalers = sweep.get("scalers", [])
    seeds = sweep.get("seeds", [int(cfg.get("seed", 42))])
    model_overrides_list = build_override_grid(sweep.get("model"))
    train_overrides_list = build_override_grid(sweep.get("training"))
    data_overrides_list = build_override_grid(sweep.get("data"))

    summary_path = output_root / "sweep_results.csv"
    run_summary_path = output_root / "sweep_run_summary.csv"
    override_fields = (
        _override_fieldnames("model", model_overrides_list)
        + _override_fieldnames("train", train_overrides_list)
        + _override_fieldnames("data", data_overrides_list)
    )

    label_fieldnames = [
        "run_dir",
        "model",
        "loss",
        "scaler",
        "seed",
        "label",
        "raw_unit",
        "display_unit",
        "display_scale",
        "rmse",
        "mae",
        "mse",
        "rmse_display",
        "mae_display",
        "mse_display",
        "rmse_norm",
        "mae_norm",
        "mse_norm",
        "agg_rmse_mean",
        "agg_mae_mean",
        "agg_rmse_norm_mean",
        "agg_mae_norm_mean",
        "test_mse",
    ] + override_fields
    _write_csv_header(summary_path, label_fieldnames)

    run_fieldnames = [
        "run_dir",
        "model",
        "loss",
        "scaler",
        "seed",
        "n_features",
        "n_targets",
        "agg_rmse_mean",
        "agg_mae_mean",
        "agg_mse_mean",
        "agg_rmse_norm_mean",
        "agg_mae_norm_mean",
        "agg_mse_norm_mean",
        "max_rmse",
        "max_mae",
        "best_val_loss",
        "final_val_loss",
        "final_train_loss",
        "final_train_eval_loss",
        "train_wall_time_sec",
        "eval_wall_time_sec",
        "total_wall_time_sec",
        "n_parameters_total",
        "n_parameters_trainable",
        "relu_units_estimate",
    ] + override_fields
    _write_csv_header(run_summary_path, run_fieldnames)

    for model_type, loss_type, scaler_type, seed, model_overrides, train_overrides, data_overrides in product(
        models,
        losses,
        scalers,
        seeds,
        model_overrides_list,
        train_overrides_list,
        data_overrides_list,
    ):
        run_name_parts = [
            sanitize_name(model_type),
            sanitize_name(loss_type),
            sanitize_name(scaler_type),
            f"seed{seed}",
        ]
        for prefix, overrides in (
            ("model", model_overrides),
            ("train", train_overrides),
            ("data", data_overrides),
        ):
            for key in sorted(overrides.keys()):
                run_name_parts.append(
                    sanitize_name(f"{prefix}_{key}{format_run_value(overrides[key])}")
                )
        run_name = "__".join(run_name_parts)
        run_dir = output_root / run_name
        if run_dir.exists() and cfg.get("skip_if_exists", True):
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        label_rows, run_rows = train_one(
            cfg,
            run_dir,
            model_type=model_type,
            loss_type=loss_type,
            scaler_type=scaler_type,
            seed=int(seed),
            model_overrides=model_overrides,
            train_overrides=train_overrides,
            data_overrides=data_overrides,
        )
        _append_csv_rows(summary_path, label_fieldnames, label_rows)
        _append_csv_rows(run_summary_path, run_fieldnames, run_rows)


if __name__ == "__main__":
    main()
