import os
import numpy as np
import matplotlib.pyplot as plt


def plot_losses(train_losses, val_losses, test_mse, train_eval_losses=None, out_path="loss_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    if train_eval_losses is not None:
        plt.plot(epochs, train_eval_losses, label="Train eval loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.axhline(test_mse, linestyle="--", label=f"Test MSE = {test_mse:.4e}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curves to {out_path}")


def plot_scatter_per_target(y_true, y_pred, target_cols, out_dir="output/runx/"):
    os.makedirs(out_dir, exist_ok=True)
    for i, name in enumerate(target_cols):
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, s=10)
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Predicted {name}")
        plt.title(f"Test set: {name} (True vs Pred)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"scatter_{name}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved scatter plot {out_path}")
