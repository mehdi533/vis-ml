from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from scheduling.utils import activation_masks_mtlshared


def relu_activation_pattern_mtlshared(model, x_np: np.ndarray):
    return activation_masks_mtlshared(model, x_np)


def plot_diff_norm(steps: np.ndarray, diffs: np.ndarray, plot_dir: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(steps, diffs, marker="o")
    plt.xlabel("step_scale")
    plt.ylabel("||Pg_baseline - Pg_nn||")
    plt.title("Dispatch difference vs step_scale")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "scan_pg_diff_norm.png", dpi=150)
    plt.close()


def plot_pg_delta_per_gen(pg_delta_arr: np.ndarray, plot_dir: Path):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(pg_delta_arr.shape[1]) + 1, np.mean(pg_delta_arr, axis=0), marker="o")
    plt.xlabel("Generator index")
    plt.ylabel("Mean Pg_baseline - Pg_nn")
    plt.title("Mean dispatch delta per generator")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "scan_pg_delta_per_gen.png", dpi=150)
    plt.close()


def plot_pg_delta_bars(
    pg_delta_arr: np.ndarray,
    plot_dir: Path,
    ibr_idx: Sequence[int],
    max_plots: int = 20,
):
    for i in range(min(pg_delta_arr.shape[0], max_plots)):
        plt.figure(figsize=(8, 4))
        bar_colors = ["red" if j in set(ibr_idx) else "tab:blue" for j in range(pg_delta_arr.shape[1])]
        plt.bar(np.arange(pg_delta_arr.shape[1]) + 1, pg_delta_arr[i], color=bar_colors)
        plt.xlabel("Generator index")
        plt.ylabel("Pg_baseline - Pg_nn")
        plt.title(f"Dispatch delta (step idx {i:02d})")
        plt.tight_layout()
        plt.savefig(plot_dir / f"scan_pg_delta_bar_step_{i:02d}.png", dpi=150)
        plt.close()


def save_scan_results(
    steps: np.ndarray,
    diffs: np.ndarray,
    cost_diffs: np.ndarray,
    pg_baseline: np.ndarray,
    pg_nn: np.ndarray,
    pg_delta: np.ndarray,
    out_path: Path,
):
    np.savez(
        out_path,
        steps=steps,
        diffs=diffs,
        cost_diffs=cost_diffs,
        pg_baseline=pg_baseline,
        pg_nn=pg_nn,
        pg_delta=pg_delta,
    )


def plot_m_d_ibrs(
    m_vals: np.ndarray,
    d_vals: np.ndarray,
    ibr_idx: Sequence[int],
    plot_dir: Path,
):
    idx = np.arange(len(m_vals))
    ibr_set = set(int(i) for i in ibr_idx)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(idx + 1, m_vals)
    axes[0].hlines(8, 0.5, 4.5, color="gray", linestyle="--", label="max M")
    axes[0].hlines(2, 0.5, 4.5, color="gray", linestyle="--", label="min M")
    axes[0].set_ylabel("M")
    axes[0].set_title("Chosen M (IBR indices in red)")
    axes[0].set_title("Chosen M values")

    axes[1].bar(idx + 1, d_vals)
    axes[1].hlines(4, 0.5, 4.5, color="gray", linestyle="--", label="max D")
    axes[1].hlines(0, 0.5, 4.5, color="gray", linestyle="--", label="min D")
    axes[1].set_ylabel("D")
    axes[1].set_xlabel("IBR index")
    axes[1].set_title("Chosen D values")

    plt.tight_layout()
    plt.savefig(plot_dir / "scan_m_d_ibrs.png", dpi=150)
    plt.close()


def plot_pred_vs_opt(
    y_nn_scaled: np.ndarray,
    y_nn_unscaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_pred_unscaled: np.ndarray,
    plot_dir: Path,
):
    x = np.arange(len(y_nn_scaled))
    width = 0.4
    x_left = x - width / 2
    x_right = x + width / 2

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].bar(x_left, y_nn_scaled, width=width, label="opt (scaled)")
    axes[0].bar(x_right, y_pred_scaled, width=width, label="nn pred (scaled)")
    axes[0].set_ylabel("scaled")
    axes[0].legend()

    axes[1].bar(x_left, y_nn_unscaled, width=width, label="opt (unscaled)")
    axes[1].bar(x_right, y_pred_unscaled, width=width, label="nn pred (unscaled)")
    axes[1].set_ylabel("unscaled")
    axes[1].set_xlabel("output index")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(plot_dir / "scan_pred_vs_opt.png", dpi=150)
    plt.close()
