# eval_on_csv.py
# Re-evaluate a trained run directory against a new CSV dataset.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.data_utils import load_dataset
from models.models import create_model
from models.testing import evaluate_model
from models.utils import (
    build_model_kwargs,
    load_feature_name_registry,
    load_yaml,
    resolve_data_config,
)


# -----------------------------
# Path / schema resolution
# -----------------------------

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

    for name in (
        "vis_mlp_state_dict_best.pt",
        "vis_mlp_state_dict.pt",
    ):
        path = model_dir / name
        if path.exists():
            return path

    return model_dir / "vis_mlp_state_dict_best.pt"


def _resolve_run_schema(cfg: dict, fallback_targets):
    resolved_cfg = cfg.get("resolved", {})
    feature_cols = list(resolved_cfg.get("feature_cols") or [])
    target_cols = list(
        resolved_cfg.get("target_cols")
        or cfg.get("data", {}).get("target_cols")
        or fallback_targets
    )
    if not feature_cols:
        raise ValueError(
            "Config is missing resolved.feature_cols; evaluation needs the exact "
            "saved training feature contract."
        )
    if not target_cols:
        raise ValueError(
            "Config is missing target columns; evaluation needs the exact saved "
            "training target contract."
        )
    return feature_cols, target_cols


def _infer_checkpoint_input_dim(state: dict) -> int | None:
    for tensor in state.values():
        if getattr(tensor, "ndim", 0) == 2:
            return int(tensor.shape[1])
    return None


# -----------------------------
# CLI entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model directory on another CSV.")
    parser.add_argument("--config", required=True, help="Training YAML (for schema + model architecture).")
    parser.add_argument("--model-dir", required=True, help="Trained run directory with state_dict + scalers.")
    parser.add_argument("--csv", required=True, help="New CSV dataset for evaluation.")
    parser.add_argument("--model-type", default=None, help="Override model type (default: first sweep model).")
    parser.add_argument("--batch-size", type=int, default=1024, help="Eval batch size.")
    parser.add_argument(
        "--state-dict",
        default=None,
        help="Path to state dict. Default: artifact_manifest.json best/final checkpoint, then legacy fallback names.",
    )
    parser.add_argument("--out-dir", default=None, help="Output dir for RMSE/results.")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also export per-row predictions for downstream thesis plots/tables.",
    )
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / f"eval_{Path(args.csv).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = resolve_data_config(cfg["data"])
    feature_name_registry = load_feature_name_registry(data_cfg.get("feature_names_path"))
    targets = list(data_cfg.get("target_cols", []))
    feature_cols_cfg, target_cols_cfg = _resolve_run_schema(cfg, targets)
    drops = list(data_cfg.get("drop_cols", []))
    drop_prefixes = list(data_cfg.get("drop_prefixes", []))
    fill = data_cfg.get("missing_fill_value")

    X, y, feature_cols, target_cols = load_dataset(
        args.csv,
        target_cols=target_cols_cfg,
        feature_cols=feature_cols_cfg,
        remove_cols=drops,
        remove_prefixes=drop_prefixes,
        ignore_missing_remove_cols=bool(data_cfg.get("ignore_missing_drop_cols", False)),
        missing_fill_value=fill,
    )

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
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        checkpoint_in_dim = _infer_checkpoint_input_dim(state)
        raise RuntimeError(
            f"{exc}\nModel directory: {model_dir}\n"
            f"Checkpoint input dim: {checkpoint_in_dim}\n"
            f"Current evaluator input dim: {len(feature_cols)}\n"
            "This usually means the evaluator is not using the exact feature "
            "contract saved in run_config.yaml."
        ) from exc
    model.to(device)

    ds = TensorDataset(
        torch.as_tensor(X_norm, dtype=torch.float32),
        torch.as_tensor(y_norm, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0)

    evaluate_model(
        model,
        device,
        loader,
        y_scaler,
        target_cols,
        str(out_dir),
        save_predictions=args.save_predictions,
    )
    print(f"Done. Results saved under: {out_dir}")


if __name__ == "__main__":
    main()
