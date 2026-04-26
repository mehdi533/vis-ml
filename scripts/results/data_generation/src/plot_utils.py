from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    from .config_analysis import COLUMN_LABELS, FIGURES_DIR, SCENARIO_COLORS, SCENARIO_LABELS, SCENARIO_ORDER
except ImportError:
    from config_analysis import (  # type: ignore
        COLUMN_LABELS,
        FIGURES_DIR,
        SCENARIO_COLORS,
        SCENARIO_LABELS,
        SCENARIO_ORDER,
    )


def set_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.4),
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.facecolor": "#fbfaf8",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "grid.color": "#6e6259",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.edgecolor": "#4b433d",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "axes.labelcolor": "#2b2826",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": "#2b2826",
            "ytick.color": "#2b2826",
            "legend.fontsize": 9,
            "legend.frameon": False,
        }
    )


def format_column_label(column: str) -> str:
    return COLUMN_LABELS.get(column, column.replace("_", " "))


def scenario_display_name(scenario_family: str) -> str:
    return SCENARIO_LABELS.get(scenario_family, scenario_family.replace("_", " "))


def scenario_color(scenario_family: str) -> str:
    return SCENARIO_COLORS.get(scenario_family, "#5b6c7d")


def ordered_scenarios(scenario_frames: Mapping[str, pd.DataFrame]) -> Sequence[str]:
    ordered = [name for name in SCENARIO_ORDER if name in scenario_frames]
    tail = sorted(name for name in scenario_frames if name not in ordered)
    return ordered + tail


def save_figure(fig: plt.Figure, stem: str, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    target_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    png_path = target_dir / f"{stem}.png"
    pdf_path = target_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_scenario_counts(df: pd.DataFrame, *, ax: Optional[plt.Axes] = None, title: str = "") -> plt.Axes:
    axis = ax or plt.gca()
    plot_df = df.copy()
    labels = plot_df.get("scenario_label", plot_df["scenario_family"]).astype(str)
    colors = [scenario_color(value) for value in plot_df["scenario_family"].astype(str)]
    bars = axis.bar(labels, plot_df["count"], color=colors, width=0.7)
    for bar, value in zip(bars, plot_df["count"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#2b2826",
        )
    axis.set_ylabel("Retained cases [-]")
    axis.set_xlabel("")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=18)
    return axis


def plot_hist_grid(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    bins: int = 30,
    ncols: int = 2,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    cols = [col for col in columns if col in df.columns]
    n = len(cols)
    nrows = max(int(np.ceil(n / ncols)), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.8 * ncols, 2.9 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, col in zip(axes_arr.ravel(), cols):
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(series, bins=bins, color="#3b6ea8", alpha=0.9, edgecolor="white", linewidth=0.35)
        ax.set_title(format_column_label(col))
        ax.set_ylabel("Count [-]")
        ax.set_xlabel(format_column_label(col))
    for ax in axes_arr.ravel()[len(cols) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.01)
    return fig, axes_arr


def plot_scenario_hist_panels(
    scenario_frames: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    *,
    bins: int = 24,
    normalize: bool = False,
    column_labels: Optional[Mapping[str, str]] = None,
    row_labels: Optional[Mapping[str, str]] = None,
    share_x_by_column: bool = True,
    share_y_by_row: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    scenarios = ordered_scenarios(scenario_frames)
    cols = list(columns)
    fig, axes = plt.subplots(
        nrows=len(scenarios),
        ncols=len(cols),
        figsize=(3.25 * len(cols), 1.95 * len(scenarios) + 0.3),
        squeeze=False,
    )

    label_map = dict(column_labels or {})
    row_label_map = dict(row_labels or {})
    column_ranges: Dict[str, Tuple[float, float]] = {}
    if share_x_by_column:
        for column in cols:
            values = []
            for scenario_family in scenarios:
                frame = scenario_frames[scenario_family]
                if column not in frame.columns:
                    continue
                series = pd.to_numeric(frame[column], errors="coerce").dropna()
                if series.empty:
                    continue
                if np.allclose(series.to_numpy(dtype=float), -1.0):
                    continue
                values.append(series.to_numpy(dtype=float))
            if values:
                stacked = np.concatenate(values)
                xmin = float(np.nanmin(stacked))
                xmax = float(np.nanmax(stacked))
                if np.isclose(xmin, xmax):
                    pad = max(abs(xmin) * 0.05, 1e-3)
                    xmin -= pad
                    xmax += pad
                column_ranges[column] = (xmin, xmax)

    row_ymax: Dict[int, float] = {}

    for row_idx, scenario_family in enumerate(scenarios):
        frame = scenario_frames[scenario_family]
        for col_idx, column in enumerate(cols):
            ax = axes[row_idx, col_idx]
            if column not in frame.columns:
                ax.axis("off")
                continue
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            if series.empty:
                ax.axis("off")
                continue
            if np.allclose(series.to_numpy(dtype=float), -1.0):
                ax.text(0.5, 0.5, "not applicable", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                if row_idx == 0:
                    ax.set_title(label_map.get(column, format_column_label(column)))
                if col_idx == 0:
                    ax.set_ylabel(row_label_map.get(scenario_family, scenario_display_name(scenario_family)))
                continue
            if series.nunique() <= 1:
                constant = float(series.iloc[0])
                ax.axvline(constant, color=scenario_color(scenario_family), linewidth=2.0)
                ax.text(0.5, 0.84, f"fixed at {constant:.3g}", ha="center", va="center", transform=ax.transAxes, fontsize=8.5)
                if column in column_ranges:
                    ax.set_xlim(*column_ranges[column])
                row_ymax[row_idx] = max(row_ymax.get(row_idx, 0.0), ax.get_ylim()[1])
            else:
                hist_kwargs = {
                    "bins": bins,
                    "density": normalize,
                    "color": scenario_color(scenario_family),
                    "alpha": 0.92,
                    "edgecolor": "white",
                    "linewidth": 0.35,
                }
                if column in column_ranges:
                    hist_kwargs["range"] = column_ranges[column]
                ax.hist(
                    series,
                    **hist_kwargs,
                )
                row_ymax[row_idx] = max(row_ymax.get(row_idx, 0.0), ax.get_ylim()[1])
            if row_idx == 0:
                ax.set_title(label_map.get(column, format_column_label(column)))
            if col_idx == 0:
                ax.set_ylabel(row_label_map.get(scenario_family, scenario_display_name(scenario_family)))
            else:
                ax.set_ylabel("")
            if row_idx == len(scenarios) - 1:
                ax.set_xlabel(label_map.get(column, format_column_label(column)))
            else:
                ax.set_xlabel("")
            ax.tick_params(axis="x", labelrotation=0)
            if column in column_ranges:
                ax.set_xlim(*column_ranges[column])

    if share_y_by_row:
        for row_idx in range(len(scenarios)):
            ymax = row_ymax.get(row_idx)
            if ymax is None or ymax <= 0:
                continue
            for col_idx in range(len(cols)):
                ax = axes[row_idx, col_idx]
                if ax.axison:
                    ax.set_ylim(0.0, ymax)
    return fig, axes


def scatter_with_binned_mean(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    bins: int = 12,
    sample_size: int = 5000,
    title: Optional[str] = None,
    color: str = "#3b6ea8",
) -> Tuple[plt.Figure, plt.Axes]:
    subset = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty:
        raise ValueError(f"No finite data available for {x} vs {y}.")

    if subset.shape[0] > sample_size:
        subset = subset.sample(sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(subset[x], subset[y], s=11, alpha=0.18, color=color, edgecolors="none")
    _draw_binned_summary(ax, subset[x].to_numpy(dtype=float), subset[y].to_numpy(dtype=float), bins=bins)
    ax.set_xlabel(format_column_label(x))
    ax.set_ylabel(format_column_label(y))
    if title:
        ax.set_title(title)
    return fig, ax


def _draw_binned_summary(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, *, bins: int) -> None:
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    if x_values.size == 0:
        return
    bin_edges = np.linspace(np.nanmin(x_values), np.nanmax(x_values), bins + 1)
    if np.unique(bin_edges).size <= 1:
        return

    bin_ids = np.digitize(x_values, bin_edges[1:-1], right=False)
    centers = []
    means = []
    stds = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        centers.append(float(np.nanmean(x_values[mask])))
        means.append(float(np.nanmean(y_values[mask])))
        stds.append(float(np.nanstd(y_values[mask])))

    if not centers:
        return

    centers_arr = np.asarray(centers, dtype=float)
    means_arr = np.asarray(means, dtype=float)
    stds_arr = np.asarray(stds, dtype=float)
    ax.plot(centers_arr, means_arr, color="#101820", linewidth=2.0)
    ax.fill_between(
        centers_arr,
        means_arr - stds_arr,
        means_arr + stds_arr,
        color="#101820",
        alpha=0.12,
    )


def plot_relationship_by_scenario(
    scenario_frames: Mapping[str, pd.DataFrame],
    *,
    x: str,
    y: str,
    bins: int = 10,
    sample_size: int = 3000,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    row_labels: Optional[Mapping[str, str]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    scenarios = ordered_scenarios(scenario_frames)
    n = len(scenarios)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4.0 * n, 3.8),
        squeeze=False,
        sharey=True,
    )
    rho_by_scenario: Dict[str, float] = {}
    row_label_map = dict(row_labels or {})
    x_limits = []
    for scenario_family in scenarios:
        frame = scenario_frames[scenario_family]
        if x not in frame.columns or y not in frame.columns:
            continue
        subset = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if subset.empty:
            continue
        x_limits.append((float(subset[x].min()), float(subset[x].max())))
    global_x = None
    if x_limits:
        xmin = min(item[0] for item in x_limits)
        xmax = max(item[1] for item in x_limits)
        if np.isclose(xmin, xmax):
            pad = max(abs(xmin) * 0.05, 1e-3)
            xmin -= pad
            xmax += pad
        global_x = (xmin, xmax)

    for idx, scenario_family in enumerate(scenarios):
        ax = axes[0, idx]
        frame = scenario_frames[scenario_family]
        if x not in frame.columns or y not in frame.columns:
            ax.axis("off")
            continue
        subset = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if subset.empty or subset[x].nunique() <= 1 or subset[y].nunique() <= 1:
            ax.axis("off")
            continue
        if subset.shape[0] > sample_size:
            subset = subset.sample(sample_size, random_state=42)
        color = scenario_color(scenario_family)
        ax.scatter(subset[x], subset[y], s=12, alpha=0.22, color=color, edgecolors="none")
        _draw_binned_summary(ax, subset[x].to_numpy(dtype=float), subset[y].to_numpy(dtype=float), bins=bins)
        rho = subset[x].corr(subset[y], method="spearman")
        rho_by_scenario[scenario_family] = float(rho)
        ax.set_title(row_label_map.get(scenario_family, scenario_display_name(scenario_family)))
        ax.set_xlabel(x_label or format_column_label(x))
        if idx == 0:
            ax.set_ylabel(y_label or format_column_label(y))
        if global_x is not None:
            ax.set_xlim(*global_x)
        ax.text(
            0.03,
            0.96,
            f"Spearman $\\rho={rho:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#2b2826",
        )

    return fig, axes


def plot_split_histograms(
    split_frames: Dict[str, pd.DataFrame],
    column: str,
    *,
    bins: int = 30,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    colors = {"train": "#2f6690", "val": "#c65d3b", "test": "#4f7c5d"}
    for split_name in ("train", "val", "test"):
        frame = split_frames.get(split_name)
        if frame is None or column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        ax.hist(
            series,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            label=split_name,
            color=colors.get(split_name, "#5b6c7d"),
        )
    ax.set_xlabel(format_column_label(column))
    ax.set_ylabel("Density [-]")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")
    return fig, ax


def plot_split_boxplot(
    split_frames: Dict[str, pd.DataFrame],
    column: str,
    *,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    labels = []
    values = []
    colors = []
    for split_name in ("train", "val", "test"):
        frame = split_frames.get(split_name)
        if frame is None or column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        labels.append(split_name)
        values.append(series.to_numpy(dtype=float))
        colors.append({"train": "#2f6690", "val": "#c65d3b", "test": "#4f7c5d"}[split_name])
    box = ax.boxplot(values, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#2b2826")
    for median in box["medians"]:
        median.set_color("#101820")
        median.set_linewidth(1.6)
    ax.set_ylabel(format_column_label(column))
    if title:
        ax.set_title(title)
    return fig, ax
