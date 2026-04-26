from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/thesis_model_results/figures/mtlsh_activation_motion"


@dataclass
class ActivationMotionBundle:
    base_dir: Path
    pair_meta: dict[str, Any]
    pair_meta_table: pd.DataFrame
    sched_changes: pd.DataFrame
    shared_table: pd.DataFrame
    head_table: pd.DataFrame
    animation_table: pd.DataFrame | None
    focus_target: str


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_activation_motion_bundle(base_dir: str | Path = DEFAULT_OUTPUT_DIR) -> ActivationMotionBundle:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Activation-motion output directory not found: {base}")

    meta_path = base / "selected_pair_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    pair_meta = json.loads(meta_path.read_text())
    pair_meta_table = pd.read_csv(base / "selected_pair_metadata.csv")
    sched_changes = pd.read_csv(base / "selected_pair_changed_sched_features.csv")
    shared_table = pd.read_csv(base / "shared_activation_table.csv")
    head_table = pd.read_csv(base / "head_activation_table.csv")
    animation_table = _read_csv_if_exists(base / "animation_frame_summary.csv")

    focus_target = str(pair_meta.get("focus_target") or "dev_COI")
    if "target" in head_table.columns and focus_target not in set(head_table["target"].astype(str)):
        focus_target = str(head_table["target"].astype(str).iloc[0])

    return ActivationMotionBundle(
        base_dir=base,
        pair_meta=pair_meta,
        pair_meta_table=pair_meta_table,
        sched_changes=sched_changes,
        shared_table=shared_table,
        head_table=head_table,
        animation_table=animation_table,
        focus_target=focus_target,
    )


def apply_presentation_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (12, 7),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "DejaVu Serif",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4b433d",
            "axes.labelcolor": "#2b2826",
            "xtick.color": "#2b2826",
            "ytick.color": "#2b2826",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.color": "#8a8179",
            "legend.frameon": False,
        }
    )


def pair_summary(bundle: ActivationMotionBundle) -> pd.Series:
    row = bundle.pair_meta_table.iloc[0].copy()
    row["focus_target"] = bundle.focus_target
    return row


def switched_shared_neurons(bundle: ActivationMotionBundle, changed_only: bool = True) -> pd.DataFrame:
    wide = (
        bundle.shared_table.pivot(index="shared_neuron", columns="sample", values="active")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if {"sample_a", "sample_b"} <= set(wide.columns):
        wide["changed"] = wide["sample_a"] != wide["sample_b"]
    else:
        wide["changed"] = False
    if changed_only:
        wide = wide.loc[wide["changed"]].copy()
    return wide.sort_values(["changed", "shared_neuron"], ascending=[False, True]).reset_index(drop=True)


def switched_head_neurons(
    bundle: ActivationMotionBundle,
    target: str | None = None,
    changed_only: bool = True,
) -> pd.DataFrame:
    target_name = str(target or bundle.focus_target)
    subset = bundle.head_table.loc[bundle.head_table["target"].astype(str) == target_name].copy()
    wide = (
        subset.pivot(index="head_neuron", columns="sample", values="active")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    if {"sample_a", "sample_b"} <= set(wide.columns):
        wide["changed"] = wide["sample_a"] != wide["sample_b"]
    else:
        wide["changed"] = False
    if changed_only:
        wide = wide.loc[wide["changed"]].copy()
    return wide.sort_values(["changed", "head_neuron"], ascending=[False, True]).reset_index(drop=True)


def plot_sched_change_bars(
    bundle: ActivationMotionBundle,
    *,
    top_k: int | None = None,
    ax: plt.Axes | None = None,
    color: str = "#b56576",
    annotate: bool = True,
    sort_desc: bool = True,
) -> plt.Axes:
    apply_presentation_style()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 5))

    df = bundle.sched_changes.copy()
    df = df.sort_values("abs_norm_delta", ascending=not sort_desc)
    if top_k is not None:
        df = df.head(int(top_k))
    df = df.iloc[::-1]

    ax.barh(df["feature"], df["abs_norm_delta"], color=color, alpha=0.88)
    ax.set_title("Schedulable M/D changes", loc="left", fontsize=13, fontweight="semibold")
    ax.set_xlabel("Absolute normalized change")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.0)

    if annotate:
        for patch, (_, row) in zip(ax.patches, df.iterrows()):
            ax.text(
                patch.get_width() + 0.01,
                patch.get_y() + patch.get_height() / 2.0,
                f"{row['sample_a_raw']:.2f} -> {row['sample_b_raw']:.2f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#4b433d",
            )

    if created:
        plt.tight_layout()
    return ax


def plot_binary_comparison(
    bundle: ActivationMotionBundle,
    *,
    target: str | None = None,
    order: str = "changed_first",
    show_pre_activation: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    apply_presentation_style()
    target_name = str(target or bundle.focus_target)

    shared = bundle.shared_table.copy()
    shared_wide = shared.pivot(index="shared_neuron", columns="sample", values="active")
    shared_pre = shared.pivot(index="shared_neuron", columns="sample", values="pre_activation")
    shared_order = shared_wide.index.to_numpy()
    if order == "changed_first":
        shared_change = (shared_wide["sample_a"] != shared_wide["sample_b"]).astype(int)
        shared_order = shared_change.sort_values(ascending=False, kind="stable").index.to_numpy()

    head = bundle.head_table.loc[bundle.head_table["target"].astype(str) == target_name].copy()
    head_wide = head.pivot(index="head_neuron", columns="sample", values="active")
    head_pre = head.pivot(index="head_neuron", columns="sample", values="pre_activation")
    head_order = head_wide.index.to_numpy()
    if order == "changed_first":
        head_change = (head_wide["sample_a"] != head_wide["sample_b"]).astype(int)
        head_order = head_change.sort_values(ascending=False, kind="stable").index.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.3, 1.4]})

    shared_img = shared_wide.loc[shared_order, ["sample_a", "sample_b"]].to_numpy(dtype=float)
    im0 = axes[0].imshow(shared_img, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0].set_title("Shared ReLU mask", loc="left", fontsize=13, fontweight="semibold")
    axes[0].set_xticks([0, 1], labels=["Sample A", "Sample B"])
    axes[0].set_yticks(np.arange(len(shared_order)), labels=[str(i) for i in shared_order])
    axes[0].set_ylabel("Shared neuron")

    head_img = head_wide.loc[head_order, ["sample_a", "sample_b"]].to_numpy(dtype=float)
    axes[1].imshow(head_img, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{target_name} head mask", loc="left", fontsize=13, fontweight="semibold")
    axes[1].set_xticks([0, 1], labels=["Sample A", "Sample B"])
    axes[1].set_yticks(np.arange(len(head_order)), labels=[str(i) for i in head_order])
    axes[1].set_ylabel("Head neuron")

    if show_pre_activation:
        for ax, pre_df, ord_idx in [
            (axes[0], shared_pre, shared_order),
            (axes[1], head_pre, head_order),
        ]:
            for row_pos, idx in enumerate(ord_idx):
                values = pre_df.loc[idx, ["sample_a", "sample_b"]].to_numpy(dtype=float)
                for col_pos, val in enumerate(values):
                    ax.text(
                        col_pos,
                        row_pos,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color="#1f1f1f" if abs(val) < 0.8 else "white",
                    )

    changed_shared = int((shared_wide["sample_a"] != shared_wide["sample_b"]).sum())
    changed_head = int((head_wide["sample_a"] != head_wide["sample_b"]).sum())
    fig.suptitle(
        f"Binary pattern comparison: shared flips={changed_shared}, {target_name} head flips={changed_head}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig, (axes[0], axes[1])


def plot_animation_summary(
    bundle: ActivationMotionBundle,
    *,
    ax: plt.Axes | None = None,
    color_shared: str = "#355070",
    color_head: str = "#b56576",
) -> plt.Axes:
    if bundle.animation_table is None:
        raise FileNotFoundError("animation_frame_summary.csv is missing from the output directory.")

    apply_presentation_style()
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 4))

    df = bundle.animation_table.copy()
    ax.plot(df["lambda"], df["shared_flip_vs_frame0"], marker="o", color=color_shared, label="Shared flips vs frame 0")
    if "focus_head_flip_vs_frame0" in df.columns:
        ax.plot(
            df["lambda"],
            df["focus_head_flip_vs_frame0"],
            marker="o",
            color=color_head,
            label=f"{bundle.focus_target} head flips vs frame 0",
        )
    ax.set_title("Interpolation path summary", loc="left", fontsize=13, fontweight="semibold")
    ax.set_xlabel("Interpolation parameter lambda")
    ax.set_ylabel("Number of changed binaries")
    ax.legend(loc="best")

    if created:
        plt.tight_layout()
    return ax


def plot_activation_dashboard(
    bundle: ActivationMotionBundle,
    *,
    target: str | None = None,
    top_k_sched: int = 8,
    show_pre_activation: bool = False,
    figsize: tuple[float, float] = (15, 10),
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    apply_presentation_style()
    target_name = str(target or bundle.focus_target)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.25)

    ax_sched = fig.add_subplot(gs[:, 0])
    plot_sched_change_bars(bundle, top_k=top_k_sched, ax=ax_sched)

    subgs = gs[0, 1].subgridspec(1, 2, width_ratios=[2.2, 1.3], wspace=0.25)
    ax_shared = fig.add_subplot(subgs[0, 0])
    ax_head = fig.add_subplot(subgs[0, 1])
    tmp_fig, _ = plot_binary_comparison(
        bundle,
        target=target_name,
        show_pre_activation=show_pre_activation,
        figsize=(10, 4),
    )
    plt.close(tmp_fig)
    # Re-render directly onto the provided axes.
    shared = bundle.shared_table.pivot(index="shared_neuron", columns="sample", values="active")
    shared_change = (shared["sample_a"] != shared["sample_b"]).astype(int)
    shared_order = shared_change.sort_values(ascending=False, kind="stable").index.to_numpy()
    shared_img = shared.loc[shared_order, ["sample_a", "sample_b"]].to_numpy(dtype=float)
    ax_shared.imshow(shared_img, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax_shared.set_title("Shared mask", loc="left", fontsize=13, fontweight="semibold")
    ax_shared.set_xticks([0, 1], labels=["A", "B"])
    ax_shared.set_yticks(np.arange(len(shared_order)), labels=[str(i) for i in shared_order])
    ax_shared.set_ylabel("Neuron")

    head = bundle.head_table.loc[bundle.head_table["target"].astype(str) == target_name].pivot(index="head_neuron", columns="sample", values="active")
    head_change = (head["sample_a"] != head["sample_b"]).astype(int)
    head_order = head_change.sort_values(ascending=False, kind="stable").index.to_numpy()
    head_img = head.loc[head_order, ["sample_a", "sample_b"]].to_numpy(dtype=float)
    ax_head.imshow(head_img, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax_head.set_title(f"{target_name} head mask", loc="left", fontsize=13, fontweight="semibold")
    ax_head.set_xticks([0, 1], labels=["A", "B"])
    ax_head.set_yticks(np.arange(len(head_order)), labels=[str(i) for i in head_order])
    ax_head.set_ylabel("Neuron")

    ax_anim = fig.add_subplot(gs[1, 1])
    plot_animation_summary(bundle, ax=ax_anim)

    fig.suptitle(
        f"Activation-motion dashboard for {target_name}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig, {"sched": ax_sched, "shared": ax_shared, "head": ax_head, "animation": ax_anim}


def save_current_figure(path: str | Path, dpi: int = 300) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(out, dpi=dpi, bbox_inches="tight")
    return out
