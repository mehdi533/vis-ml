import argparse
import csv
import os
from itertools import product
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import yaml
import torch

from models.data_utils import (
    load_dataset,
    split_data,
    scale_data,
    scale_data_with_recommendations,
    make_dataloaders,
)
from models.models import create_model
from models.training import train_model, save_model
from models.testing import evaluate_model
from models.plotting import plot_losses, plot_scatter_per_target


def _normalize_arg_list(values, default=None):
    if values is None:
        return default
    items = []
    for v in values:
        if isinstance(v, str):
            items.extend([p.strip() for p in v.split(",") if p.strip()])
        else:
            items.append(v)
    return items


def _parse_head_indices(values):
    if values is None:
        return None
    groups = []
    current = []
    for v in values:
        end_group = False
        if isinstance(v, str) and v.endswith(","):
            v = v[:-1]
            end_group = True
        if v != "":
            current.append(int(v))
        if end_group and current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups if groups else None


def _listify(value):
    if isinstance(value, list):
        return value
    return [value]


def _parse_size_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
        return [int(v) for v in items] if items else None
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, (list, tuple)) for v in value):
            return [[int(x) for x in group] for group in value]
        return [int(v) for v in value]
    return [int(value)]


def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _resolve_feature_indices(
    feature_cols: Sequence[str],
    *,
    idx_key: str,
    col_key: str,
    cfg: Dict,
):
    idx_values = cfg.get(idx_key)
    col_values = cfg.get(col_key)

    if idx_values is not None and col_values is not None:
        raise ValueError(f"Specify only one of '{idx_key}' or '{col_key}'.")

    if col_values is not None:
        names = [str(v) for v in col_values]
        name_to_idx = {name: i for i, name in enumerate(feature_cols)}
        missing = [name for name in names if name not in name_to_idx]
        if missing:
            raise KeyError(f"Unknown feature names in '{col_key}': {missing}")
        return [name_to_idx[name] for name in names]

    if idx_values is not None:
        idx = [int(v) for v in idx_values]
        bad = [i for i in idx if i < 0 or i >= len(feature_cols)]
        if bad:
            raise IndexError(f"Out-of-range indices in '{idx_key}': {bad}")
        return idx

    return None


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def train_one(
    cfg: Dict,
    run_dir: Path,
    *,
    model_type: str,
    loss_type: str,
    scaler_type: str,
    seed: int,
    train_overrides: Optional[Dict] = None,
):
    _set_seed(seed)

    data_cfg = cfg["data"]
    targets = _normalize_arg_list(data_cfg.get("target_cols"))
    drops = _normalize_arg_list(data_cfg.get("drop_cols"))

    X, y, feature_cols, target_cols = load_dataset(
        data_cfg["csv_path"],
        target_cols=targets,
        remove_cols=drops,
    )

    split_cfg = cfg.get("split", {})
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        test_size=float(split_cfg.get("test_size", 0.3)),
        val_fraction=float(split_cfg.get("val_fraction", 0.5)),
        random_state=int(split_cfg.get("random_state", 42)),
    )

    use_reco = bool(data_cfg.get("use_recommended_scalers", False))
    if use_reco:
        csv_dir = os.path.dirname(data_cfg["csv_path"])
        x_scaler_csv = data_cfg.get("x_scaler_csv") or os.path.join(csv_dir, "scaler_recommendations.csv")
        y_scaler_csv = data_cfg.get("y_scaler_csv") or os.path.join(csv_dir, "label_scaler_recommendations.csv")
        (
            X_train_n, X_val_n, X_test_n,
            y_train_n, y_val_n, y_test_n,
            y_scaler,
        ) = scale_data_with_recommendations(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_cols=feature_cols,
            target_cols=target_cols,
            x_scaler_csv=x_scaler_csv,
            y_scaler_csv=y_scaler_csv,
            x_scaler_path=str(run_dir / "x_scaler.pkl"),
            y_scaler_path=str(run_dir / "y_scaler.pkl"),
            default_scaler_type=scaler_type,
        )
    else:
        (
            X_train_n, X_val_n, X_test_n,
            y_train_n, y_val_n, y_test_n,
            y_scaler,
        ) = scale_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            x_scaler_path=str(run_dir / "x_scaler.pkl"),
            y_scaler_path=str(run_dir / "y_scaler.pkl"),
            scaler_type=scaler_type,
        )

    train_cfg = dict(cfg["training"])
    if train_overrides:
        train_cfg.update(train_overrides)
    (
        train_loader, val_loader, test_loader,
        train_ds, val_ds, test_ds
    ) = make_dataloaders(
        X_train_n, X_val_n, X_test_n,
        y_train_n, y_val_n, y_test_n,
        batch_size_train=int(train_cfg.get("batch_train", 64)),
        batch_size_eval=int(train_cfg.get("batch_eval", 128)),
    )

    group_head_indices = _parse_head_indices(train_cfg.get("head_indices"))
    model_cfg = cfg.get("model", {})
    shared_sizes = _parse_size_list(model_cfg.get("shared_sizes"))
    head_sizes = _parse_size_list(model_cfg.get("head_sizes"))
    hidden_sizes = _parse_size_list(model_cfg.get("hidden_sizes"))
    group_shared_sizes = _parse_size_list(model_cfg.get("group_shared_sizes"))
    dropout = float(model_cfg.get("dropout", 0.0))
    activation = str(model_cfg.get("activation", "relu"))
    u_feature_idx = _resolve_feature_indices(
        feature_cols,
        idx_key="u_feature_idx",
        col_key="u_feature_cols",
        cfg=model_cfg,
    )
    v_feature_idx = _resolve_feature_indices(
        feature_cols,
        idx_key="v_feature_idx",
        col_key="v_feature_cols",
        cfg=model_cfg,
    )
    model, device = create_model(
        model_type,
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        group_head_indices=group_head_indices,
        shared_sizes=shared_sizes,
        head_sizes=head_sizes,
        hidden_sizes=hidden_sizes,
        group_shared_sizes=group_shared_sizes,
        dropout=dropout,
        kan_grid_size=int(train_cfg.get("kan_grid_size", 8)),
        kan_grid_min=float(train_cfg.get("kan_grid_min", -1.0)),
        kan_grid_max=float(train_cfg.get("kan_grid_max", 1.0)),
        u_feature_idx=u_feature_idx,
        v_feature_idx=v_feature_idx,
        activation=activation,
    )

    model_txt = run_dir / "model.txt"
    model_txt.write_text(str(model), encoding="utf-8")

    model, train_losses, train_eval_losses, val_losses = train_model(
        model, device,
        train_loader, val_loader,
        train_ds, val_ds,
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
        best_model_path=str(run_dir / "vis_mlp_state_dict_best.pt"),
    )

    y_true, y_pred, y_true_norm, y_pred_norm, rmse, rmse_norm = evaluate_model(
        model, device, test_loader, y_scaler, target_cols, str(run_dir)
    )
    test_mse = float(np.mean((y_pred_norm - y_true_norm) ** 2))

    plot_losses(
        train_losses,
        val_losses,
        test_mse,
        train_eval_losses=train_eval_losses,
        out_path=str(run_dir / "loss_curve.png"),
    )
    plot_scatter_per_target(y_true, y_pred, target_cols, out_dir=str(run_dir))
    save_model(model, path=str(run_dir / "vis_mlp_state_dict.pt"))

    rmse_rows = []
    for label, rmse_val, norm_val in zip(target_cols, rmse, rmse_norm):
        row = {
            "model": model_type,
            "loss": loss_type,
            "scaler": scaler_type,
            "seed": seed,
            "label": label,
            "rmse": float(rmse_val),
            "norm": float(norm_val),
            "test_mse": test_mse,
        }
        if train_overrides:
            for key, val in train_overrides.items():
                row[f"train_{key}"] = val
        rmse_rows.append(row)

    return rmse_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiple models from a YAML sweep.")
    parser.add_argument("--config", default="to_export/train_sweep.yaml", help="Path to sweep config.")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    output_root = Path(cfg["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    sweep = cfg["sweep"]
    models = sweep.get("models", [])
    losses = sweep.get("losses", [])
    scalers = sweep.get("scalers", [])
    seeds = sweep.get("seeds", [int(cfg.get("seed", 42))])
    train_grid_cfg = sweep.get("training", {})

    if train_grid_cfg:
        keys = list(train_grid_cfg.keys())
        values = [_listify(train_grid_cfg[k]) for k in keys]
        train_overrides_list = [dict(zip(keys, combo)) for combo in product(*values)]
    else:
        train_overrides_list = [{}]

    summary_path = output_root / "sweep_results.csv"
    base_fields = [
        "model",
        "loss",
        "scaler",
        "seed",
        "label",
        "rmse",
        "norm",
        "test_mse",
    ]
    override_fields = [f"train_{key}" for key in sorted(train_grid_cfg.keys())]
    fieldnames = base_fields + override_fields
    _write_csv_header(summary_path, fieldnames)

    for model_type in models:
        for loss_type in losses:
            for scaler_type in scalers:
                for seed in seeds:
                    for train_overrides in train_overrides_list:
                        run_name_parts = [
                            _sanitize_name(model_type),
                            _sanitize_name(loss_type),
                            _sanitize_name(scaler_type),
                            f"seed{seed}",
                        ]
                        for key in sorted(train_overrides.keys()):
                            val = train_overrides[key]
                            run_name_parts.append(f"{key}{val}")
                        run_name = "__".join(run_name_parts)
                        run_dir = output_root / run_name
                        if run_dir.exists() and cfg.get("skip_if_exists", True):
                            continue
                        run_dir.mkdir(parents=True, exist_ok=True)
                        rmse_rows = train_one(
                            cfg,
                            run_dir,
                            model_type=model_type,
                            loss_type=loss_type,
                            scaler_type=scaler_type,
                            seed=int(seed),
                            train_overrides=train_overrides,
                        )
                        _append_csv_rows(summary_path, fieldnames, rmse_rows)


if __name__ == "__main__":
    main()
