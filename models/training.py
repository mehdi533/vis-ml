import os
import numpy as np
import torch

import sys as _sys
from pathlib import Path as _Path

from models.losses import build_loss


def train_model(
    model,
    device,
    train_loader,
    val_loader,
    train_ds,
    val_ds,
    output_dir,
    n_epochs=500,
    lr=1e-4,
    weight_decay=1e-5,
    loss_type: str = "mse",
    ce_weights: list = None,
    use_lr_scheduler: bool = False,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_min_lr: float = 1e-6,
    best_model_path: str | None = None,
):
    ce_weights = ce_weights or []
    out_dim = train_ds[0][1].shape[0]

    logs_path = os.path.join(output_dir, "training_summary.txt")
    with open(logs_path, "w", encoding="utf-8") as f:
        f.write("")

    def log(msg: str):
        print(msg)
        with open(logs_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _log_target_stats():
        y_train = train_ds.tensors[1].detach().cpu().numpy()
        y_val = val_ds.tensors[1].detach().cpu().numpy()

        def _stats(arr):
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            q05 = np.quantile(arr, 0.05, axis=0)
            q95 = np.quantile(arr, 0.95, axis=0)
            return mean, std, q05, q95

        train_mean, train_std, train_q05, train_q95 = _stats(y_train)
        val_mean, val_std, val_q05, val_q95 = _stats(y_val)
        std_ratio = val_std / (train_std + 1e-12)

        log("Target stats (scaled):")
        log(f"train mean={np.round(train_mean, 4)}")
        log(f"train std={np.round(train_std, 4)}")
        log(f"train q05={np.round(train_q05, 4)}")
        log(f"train q95={np.round(train_q95, 4)}")
        log(f"val mean={np.round(val_mean, 4)}")
        log(f"val std={np.round(val_std, 4)}")
        log(f"val q05={np.round(val_q05, 4)}")
        log(f"val q95={np.round(val_q95, 4)}")
        log(f"val/train std ratio={np.round(std_ratio, 4)}")

    loss_fn, extra_params = build_loss(loss_type, ce_weights, device, out_dim)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + extra_params,
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr,
        )

    log(f"Loss type: {loss_type} | weights={ce_weights}")
    _log_target_stats()
    if use_lr_scheduler:
        log(
            "LR scheduler: ReduceLROnPlateau "
            f"(factor={lr_scheduler_factor}, patience={lr_scheduler_patience}, min_lr={lr_scheduler_min_lr})"
        )

    best_val_loss = float("inf")
    best_state = None
    train_losses, train_eval_losses, val_losses = [], [], []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_ds)

        train_eval_loss = 0.0
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                train_eval_loss += loss.item() * xb.size(0)
        train_eval_loss /= len(train_ds)

        train_losses.append(train_loss)
        train_eval_losses.append(train_eval_loss)
        val_losses.append(val_loss)

        if hasattr(loss_fn, "log_vars"):
            logvars = loss_fn.log_vars.detach().cpu().numpy()
            logvars_str = f"log_vars={logvars}"
        else:
            logvars_str = ""

        lr_str = ""
        if use_lr_scheduler:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_str = f" | lr={current_lr:.2e}"

        line = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4e} | "
            f"train_eval_loss={train_eval_loss:.4e} | "
            f"val_loss={val_loss:.4e} | "
            f"{logvars_str}".strip()
        )
        line = f"{line}{lr_str}"

        log(line)

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if best_model_path:
                best_path = _Path(best_model_path)
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, best_model_path)
                log(f"Saved best model to {best_model_path}")

    log(f"Saved training logs to {logs_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, train_eval_losses, val_losses


def save_model(model, path="vis_mlp_state_dict.pt"):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")
