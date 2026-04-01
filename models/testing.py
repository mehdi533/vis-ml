import csv
import json
from pathlib import Path

import numpy as np
import torch


def compute_prediction_metrics(y_true, y_pred, y_true_norm, y_pred_norm, target_cols):
    diff = y_pred - y_true
    diff_norm = y_pred_norm - y_true_norm

    mse = np.mean(diff ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=0)

    mse_norm = np.mean(diff_norm ** 2, axis=0)
    rmse_norm = np.sqrt(mse_norm)
    mae_norm = np.mean(np.abs(diff_norm), axis=0)

    metrics_by_target = []
    for idx, name in enumerate(target_cols):
        metrics_by_target.append(
            {
                "label": str(name),
                "rmse": float(rmse[idx]),
                "mae": float(mae[idx]),
                "mse": float(mse[idx]),
                "rmse_norm": float(rmse_norm[idx]),
                "mae_norm": float(mae_norm[idx]),
                "mse_norm": float(mse_norm[idx]),
            }
        )

    summary = {
        "n_targets": int(len(target_cols)),
        "agg_rmse_mean": float(np.mean(rmse)),
        "agg_mae_mean": float(np.mean(mae)),
        "agg_mse_mean": float(np.mean(mse)),
        "agg_rmse_norm_mean": float(np.mean(rmse_norm)),
        "agg_mae_norm_mean": float(np.mean(mae_norm)),
        "agg_mse_norm_mean": float(np.mean(mse_norm)),
        "max_rmse": float(np.max(rmse)),
        "max_mae": float(np.max(mae)),
        "targets": metrics_by_target,
    }
    return metrics_by_target, summary


def _write_metrics_files(output_dir: Path, target_cols, metrics_by_target) -> None:
    rmse_path = output_dir / "rmse_results.txt"
    metrics_csv = output_dir / "metrics_by_target.csv"

    with rmse_path.open("w", encoding="utf-8") as f:
        for row in metrics_by_target:
            line = (
                f"Test RMSE({row['label']}) = {row['rmse']:.4f} | "
                f"norm: {row['rmse_norm']:.4f} | real_diff: {row['mae']:.4f}"
            )
            print(line)
            f.write(line + "\n")

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "rmse", "mae", "mse", "rmse_norm", "mae_norm", "mse_norm"],
        )
        writer.writeheader()
        writer.writerows(metrics_by_target)

    print(f"Saved RMSE results to {rmse_path}")
    print(f"Saved detailed metrics to {metrics_csv}")


def _write_summary_json(output_dir: Path, summary: dict) -> None:
    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved aggregate metrics to {summary_path}")


def _write_predictions_csv(output_dir: Path, target_cols, y_true, y_pred, y_true_norm, y_pred_norm) -> None:
    pred_path = output_dir / "test_predictions.csv"
    fieldnames = ["row_idx"]
    for name in target_cols:
        fieldnames.extend(
            [
                f"{name}__true",
                f"{name}__pred",
                f"{name}__true_norm",
                f"{name}__pred_norm",
            ]
        )

    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(y_true.shape[0]):
            row = {"row_idx": idx}
            for col_idx, name in enumerate(target_cols):
                row[f"{name}__true"] = float(y_true[idx, col_idx])
                row[f"{name}__pred"] = float(y_pred[idx, col_idx])
                row[f"{name}__true_norm"] = float(y_true_norm[idx, col_idx])
                row[f"{name}__pred_norm"] = float(y_pred_norm[idx, col_idx])
            writer.writerow(row)

    print(f"Saved test predictions to {pred_path}")


def evaluate_model(
    model,
    device,
    test_loader,
    y_scaler,
    target_cols,
    output_dir,
    *,
    save_predictions: bool = False,
    return_metrics: bool = False,
):
    model.eval()
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            y_true_list.append(yb.numpy())
            y_pred_list.append(preds)

    y_true_norm = np.vstack(y_true_list)
    y_pred_norm = np.vstack(y_pred_list)

    y_true = y_scaler.inverse_transform(y_true_norm)
    y_pred = y_scaler.inverse_transform(y_pred_norm)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_target, summary = compute_prediction_metrics(
        y_true,
        y_pred,
        y_true_norm,
        y_pred_norm,
        target_cols,
    )
    _write_metrics_files(out_dir, target_cols, metrics_by_target)
    _write_summary_json(out_dir, summary)

    if save_predictions:
        _write_predictions_csv(out_dir, target_cols, y_true, y_pred, y_true_norm, y_pred_norm)

    rmse = np.array([row["rmse"] for row in metrics_by_target], dtype=np.float32)
    rmse_norm = np.array([row["rmse_norm"] for row in metrics_by_target], dtype=np.float32)

    if return_metrics:
        return y_true, y_pred, y_true_norm, y_pred_norm, rmse, rmse_norm, metrics_by_target, summary

    return y_true, y_pred, y_true_norm, y_pred_norm, rmse, rmse_norm
