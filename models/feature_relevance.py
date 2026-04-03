import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import joblib
import numpy as np
import pandas as pd
import torch

from models.data_utils import load_dataset, split_data
from models.models import create_model
from models.workflow_utils import (
    build_model_kwargs,
    load_feature_name_registry,
    load_yaml,
    resolve_data_config,
)


def _resolve_state_dict_path(model_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    manifest_path = model_dir / "artifact_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        best_name = manifest.get("primary_best_state_dict")
        final_name = manifest.get("primary_final_state_dict")
        for name in (best_name, final_name):
            if name and (model_dir / name).exists():
                return model_dir / name

    for name in ("vis_mlp_state_dict_best.pt", "vis_mlp_state_dict.pt"):
        path = model_dir / name
        if path.exists():
            return path

    return model_dir / "vis_mlp_state_dict_best.pt"


def _load_group_config(path: str | None) -> dict:
    if path is None:
        return {}
    cfg = load_yaml(path)
    return cfg.get("groups", cfg)


def _resolve_group_features(group_cfg: dict, feature_cols: list[str]) -> dict[str, list[str]]:
    name_to_feature = set(feature_cols)
    resolved = {}
    for group_name, spec in group_cfg.items():
        if isinstance(spec, list):
            exact = [str(item) for item in spec]
            prefixes = []
        else:
            exact = [str(item) for item in spec.get("exact", [])]
            prefixes = [str(item) for item in spec.get("prefixes", [])]

        members = []
        for name in exact:
            if name in name_to_feature and name not in members:
                members.append(name)
        for col in feature_cols:
            if any(col.startswith(prefix) for prefix in prefixes) and col not in members:
                members.append(col)
        if members:
            resolved[str(group_name)] = members
    return resolved


def _predict_numpy(model, device, X_array: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_array.shape[0], batch_size):
            xb = torch.as_tensor(X_array[start : start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.vstack(outputs)


def _collect_attention(model, device, X_array: np.ndarray, batch_size: int) -> np.ndarray:
    if not hasattr(model, "attention"):
        raise ValueError("Selected model does not expose an attention module.")

    attn_rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X_array.shape[0], batch_size):
            xb = torch.as_tensor(X_array[start : start + batch_size], dtype=torch.float32, device=device)
            try:
                _, attn = model(xb, return_attn=True)
            except TypeError:
                model(xb)
                attn = getattr(model.attention, "last_attn", None)
                if attn is None:
                    raise ValueError("Attention weights are not available for this model.")
            attn_rows.append(attn.detach().cpu().numpy())
    return np.vstack(attn_rows)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=fieldnames).to_csv(path, index=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_attention_rows(attn_mean: np.ndarray, feature_cols: list[str]) -> list[dict]:
    return [
        {"feature": feature, "attention_mean": float(attn_mean[idx])}
        for idx, feature in enumerate(feature_cols)
    ]


def _build_attention_group_rows(attn_mean: np.ndarray, feature_cols: list[str], groups: dict[str, list[str]]) -> list[dict]:
    rows = []
    name_to_idx = {name: idx for idx, name in enumerate(feature_cols)}
    for group_name, members in groups.items():
        idx = [name_to_idx[name] for name in members if name in name_to_idx]
        if not idx:
            continue
        rows.append(
            {
                "group": group_name,
                "n_features": len(idx),
                "attention_mean_sum": float(np.sum(attn_mean[idx])),
                "attention_mean_avg": float(np.mean(attn_mean[idx])),
            }
        )
    return rows


def _permutation_importance(
    model,
    device,
    X_norm: np.ndarray,
    y_norm: np.ndarray,
    target_cols: list[str],
    feature_cols: list[str],
    batch_size: int,
    num_repeats: int,
    rng: np.random.Generator,
):
    baseline_pred = _predict_numpy(model, device, X_norm, batch_size)
    baseline_mse = np.mean((baseline_pred - y_norm) ** 2, axis=0)

    rows = []
    for feat_idx, feature in enumerate(feature_cols):
        deltas = []
        for _ in range(num_repeats):
            X_perm = X_norm.copy()
            perm = rng.permutation(X_perm.shape[0])
            X_perm[:, feat_idx] = X_perm[perm, feat_idx]
            pred_perm = _predict_numpy(model, device, X_perm, batch_size)
            perm_mse = np.mean((pred_perm - y_norm) ** 2, axis=0)
            deltas.append(perm_mse - baseline_mse)

        delta_mean = np.mean(np.vstack(deltas), axis=0)
        row = {
            "feature": feature,
            "importance_mse_norm_mean": float(np.mean(delta_mean)),
        }
        for target_idx, target_name in enumerate(target_cols):
            row[f"importance_mse_norm__{target_name}"] = float(delta_mean[target_idx])
        rows.append(row)

    return baseline_mse, rows


def _group_permutation_importance(
    model,
    device,
    X_norm: np.ndarray,
    y_norm: np.ndarray,
    target_cols: list[str],
    feature_cols: list[str],
    groups: dict[str, list[str]],
    batch_size: int,
    num_repeats: int,
    rng: np.random.Generator,
):
    baseline_pred = _predict_numpy(model, device, X_norm, batch_size)
    baseline_mse = np.mean((baseline_pred - y_norm) ** 2, axis=0)
    name_to_idx = {name: idx for idx, name in enumerate(feature_cols)}

    rows = []
    for group_name, members in groups.items():
        idx = [name_to_idx[name] for name in members if name in name_to_idx]
        if not idx:
            continue

        deltas = []
        for _ in range(num_repeats):
            X_perm = X_norm.copy()
            perm = rng.permutation(X_perm.shape[0])
            X_perm[:, idx] = X_perm[perm][:, idx]
            pred_perm = _predict_numpy(model, device, X_perm, batch_size)
            perm_mse = np.mean((pred_perm - y_norm) ** 2, axis=0)
            deltas.append(perm_mse - baseline_mse)

        delta_mean = np.mean(np.vstack(deltas), axis=0)
        n_features = len(idx)
        row = {
            "group": group_name,
            "n_features": n_features,
            "importance_mse_norm_mean": float(np.mean(delta_mean)),
            "importance_mse_norm_avg_per_feature": float(np.mean(delta_mean) / n_features),
        }
        for target_idx, target_name in enumerate(target_cols):
            delta_value = float(delta_mean[target_idx])
            row[f"importance_mse_norm__{target_name}"] = delta_value
            row[f"importance_mse_norm_avg_per_feature__{target_name}"] = delta_value / n_features
        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute attention and permutation feature relevance.")
    parser.add_argument("--config", required=True, help="Training YAML used to define the dataset/model schema.")
    parser.add_argument("--model-dir", required=True, help="Trained run directory.")
    parser.add_argument("--csv", default=None, help="Optional CSV override. Default: data.csv_path from config.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to analyze when using the training CSV.",
    )
    parser.add_argument(
        "--mode",
        choices=["attention", "permutation", "both"],
        default="both",
        help="Which relevance diagnostics to run.",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size used for forward passes.")
    parser.add_argument("--num-repeats", type=int, default=5, help="Permutation repeats per feature/group.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional row cap for faster analysis.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation sampling.")
    parser.add_argument("--group-config", default=None, help="Optional YAML mapping feature groups to names/prefixes.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: model-dir/feature_relevance.")
    parser.add_argument("--model-type", default=None, help="Optional model type override.")
    parser.add_argument("--state-dict", default=None, help="Optional state dict override.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "feature_relevance"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = resolve_data_config(cfg["data"])
    feature_name_registry = load_feature_name_registry(data_cfg.get("feature_names_path"))
    csv_path = args.csv or data_cfg["csv_path"]
    targets = list(data_cfg.get("target_cols", []))
    feature_cols_cfg = list(data_cfg.get("feature_cols", [])) or None
    drops = list(data_cfg.get("drop_cols", []))
    drop_prefixes = list(data_cfg.get("drop_prefixes", []))
    fill = data_cfg.get("missing_fill_value")

    X, y, feature_cols, target_cols = load_dataset(
        csv_path,
        target_cols=targets,
        feature_cols=feature_cols_cfg,
        allowed_feature_cols=data_cfg.get("allowed_feature_cols"),
        allowed_feature_prefixes=data_cfg.get("allowed_feature_prefixes"),
        remove_cols=drops,
        remove_prefixes=drop_prefixes,
        ignore_missing_remove_cols=bool(data_cfg.get("ignore_missing_drop_cols", False)),
        missing_fill_value=fill,
    )

    if args.csv is None and args.split != "all":
        split_cfg = cfg.get("split", {})
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X,
            y,
            test_size=float(split_cfg.get("test_size", 0.3)),
            val_fraction=float(split_cfg.get("val_fraction", 0.5)),
            random_state=int(split_cfg.get("random_state", 42)),
        )
        if args.split == "train":
            X, y = X_train, y_train
        elif args.split == "val":
            X, y = X_val, y_val
        else:
            X, y = X_test, y_test

    if args.max_samples is not None and X.shape[0] > int(args.max_samples):
        X = X[: int(args.max_samples)]
        y = y[: int(args.max_samples)]

    x_scaler = joblib.load(model_dir / "x_scaler.pkl")
    y_scaler = joblib.load(model_dir / "y_scaler.pkl")
    X_norm = x_scaler.transform(X)
    y_norm = y_scaler.transform(y)

    model_type = args.model_type or cfg.get("sweep", {}).get("models", ["MLP"])[0]
    model_cfg = cfg.get("model", {})
    model, device = create_model(
        model_type,
        in_dim=len(feature_cols),
        out_dim=len(target_cols),
        **build_model_kwargs(
            model_cfg,
            feature_cols,
            train_cfg=cfg.get("training", {}),
            feature_name_registry=feature_name_registry,
        ),
    )

    state_path = _resolve_state_dict_path(model_dir, args.state_dict)
    if not state_path.exists():
        raise FileNotFoundError(f"State dict not found: {state_path}")

    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    group_cfg = _load_group_config(args.group_config)
    groups = _resolve_group_features(group_cfg, list(feature_cols))
    rng = np.random.default_rng(args.seed)

    metadata = {
        "mode": args.mode,
        "split": args.split,
        "n_rows": int(X_norm.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_targets": int(len(target_cols)),
        "interpretation_note": (
            "Attention scores and permutation scores are explanatory diagnostics. "
            "They do not imply causality and should not be read as physical proofs."
        ),
    }

    if args.mode in {"attention", "both"}:
        attn = _collect_attention(model, device, X_norm, int(args.batch_size))
        attn_mean = attn.mean(axis=0)
        feature_rows = _build_attention_rows(attn_mean, list(feature_cols))
        feature_path = out_dir / "attention_feature_relevance.csv"
        _write_csv(feature_path, feature_rows, ["feature", "attention_mean"])

        if groups:
            group_rows = _build_attention_group_rows(attn_mean, list(feature_cols), groups)
            _write_csv(
                out_dir / "attention_group_relevance.csv",
                group_rows,
                ["group", "n_features", "attention_mean_sum", "attention_mean_avg"],
            )

    if args.mode in {"permutation", "both"}:
        baseline_mse, feature_rows = _permutation_importance(
            model,
            device,
            X_norm,
            y_norm,
            list(target_cols),
            list(feature_cols),
            int(args.batch_size),
            int(args.num_repeats),
            rng,
        )
        feature_fieldnames = ["feature", "importance_mse_norm_mean"] + [
            f"importance_mse_norm__{target}" for target in target_cols
        ]
        _write_csv(out_dir / "permutation_feature_relevance.csv", feature_rows, feature_fieldnames)

        if groups:
            group_rows = _group_permutation_importance(
                model,
                device,
                X_norm,
                y_norm,
                list(target_cols),
                list(feature_cols),
                groups,
                int(args.batch_size),
                int(args.num_repeats),
                rng,
            )
            group_fieldnames = [
                "group",
                "n_features",
                "importance_mse_norm_mean",
                "importance_mse_norm_avg_per_feature",
            ] + [
                f"importance_mse_norm__{target}" for target in target_cols
            ] + [
                f"importance_mse_norm_avg_per_feature__{target}" for target in target_cols
            ]
            _write_csv(out_dir / "permutation_group_relevance.csv", group_rows, group_fieldnames)

        metadata["baseline_mse_norm"] = {
            str(target): float(baseline_mse[idx]) for idx, target in enumerate(target_cols)
        }

    _write_json(out_dir / "feature_relevance_summary.json", metadata)
    print(f"Saved feature relevance outputs under {out_dir}")


if __name__ == "__main__":
    main()
