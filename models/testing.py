import os
import numpy as np
import torch


def evaluate_model(model, device, test_loader, y_scaler, target_cols, output_dir):
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
    rmse_norm = np.sqrt(np.mean((y_pred_norm - y_true_norm) ** 2, axis=0))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))

    rmse_path = os.path.join(output_dir, "rmse_results.txt")

    with open(rmse_path, "w", encoding="utf-8") as f:
        for name, val, val_norm in zip(target_cols, rmse, rmse_norm):
            line = f"Test RMSE({name}) = {val:.4f} | norm: {val_norm:.4f}"
            print(line)
            f.write(line + "\n")

    print(f"Saved RMSE results to {rmse_path}")

    return y_true, y_pred, y_true_norm, y_pred_norm, rmse, rmse_norm
