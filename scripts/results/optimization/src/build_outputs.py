from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

try:
    from .analysis_utils import (
        baseline_formulation_id,
        build_dispatch_comparison_tables,
        build_formulation_catalog,
        cost_summary_table,
        formulation_order,
        line_security_summary,
        load_manifest_tables,
        load_optimization_runs_master,
        load_replay_outputs,
        load_replay_runs_master,
        load_run_catalog,
        method_cost_breakdown,
        merge_dynamic_and_line_security,
        replay_metric_summary,
        reserve_summary_by_method,
        resolve_analysis_config,
        retained_formulation_id,
        save_plot_data,
        scenario_coverage_summary,
        solver_complexity_summary,
        vis_allocation_summary,
        write_latex_table,
        write_markdown_table,
    )
    from .benchmark_utils import write_csv_and_parquet
    from .config_analysis import (
        BENCHMARK_CONFIG,
        FIGURES_DIR,
        MERGED_RESULTS_DIR,
        PLOT_DATA_DIR,
        TABLES_DIR,
        ensure_output_dirs,
        OptimizationResultsNotReadyError,
    )
    from .plot_utils import formulation_color, save_figure, set_thesis_style
    from .validation_utils import replay_breakdown_by_method, replay_breakdown_by_metric, summarize_replay_feasibility
except ImportError:
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from analysis_utils import (  # type: ignore
        baseline_formulation_id,
        build_dispatch_comparison_tables,
        build_formulation_catalog,
        cost_summary_table,
        formulation_order,
        line_security_summary,
        load_manifest_tables,
        load_optimization_runs_master,
        load_replay_outputs,
        load_replay_runs_master,
        load_run_catalog,
        method_cost_breakdown,
        merge_dynamic_and_line_security,
        replay_metric_summary,
        reserve_summary_by_method,
        resolve_analysis_config,
        retained_formulation_id,
        save_plot_data,
        scenario_coverage_summary,
        solver_complexity_summary,
        vis_allocation_summary,
        write_latex_table,
        write_markdown_table,
    )
    from benchmark_utils import write_csv_and_parquet  # type: ignore
    from config_analysis import BENCHMARK_CONFIG, FIGURES_DIR, MERGED_RESULTS_DIR, PLOT_DATA_DIR, TABLES_DIR, ensure_output_dirs, OptimizationResultsNotReadyError  # type: ignore
    from plot_utils import formulation_color, save_figure, set_thesis_style  # type: ignore
    from validation_utils import replay_breakdown_by_method, replay_breakdown_by_metric, summarize_replay_feasibility  # type: ignore


FORMULATION_PLOT_LABELS = {
    "ed": "ED",
    "ed_line": "ED + Line",
    "ed_line_n1": "ED + Line + N-1",
    "ed_surrogate": "ED + Surrogate",
    "ed_line_n1_surrogate": "Full preventive",
    "ed_line_n1_surrogate_redispatch": "Full + redispatch",
}

FORMULATION_LEGEND_LABELS = {
    "ed": "A",
    "ed_surrogate": "D",
    "ed_line_n1_surrogate": "E",
    "ed_line_n1_surrogate_redispatch": "Full + redispatch",
    "ed_line": "B",
    "ed_line_n1": "C",
}

METRIC_LABELS = {
    "rocof_COI": r"$\mathrm{RoCoF}_{\mathrm{COI}}$",
    "dev_COI": r"$\Delta f_{\mathrm{COI}}$",
    "Delta_P_IBR_1": r"$\Delta P_{\mathrm{IBR},1}$",
    "Delta_P_IBR_2": r"$\Delta P_{\mathrm{IBR},2}$",
    "Delta_P_IBR_3": r"$\Delta P_{\mathrm{IBR},3}$",
    "Delta_P_IBR_4": r"$\Delta P_{\mathrm{IBR},4}$",
}

FORMULATION_MARKERS = {
    "ed_surrogate": "o",
    "ed_line_n1_surrogate": "s",
    "ed_line_n1_surrogate_redispatch": "D",
}

FORMULATION_LETTERS = {
    "ed": "A",
    "ed_line": "B",
    "ed_line_n1": "C",
    "ed_surrogate": "D",
    "ed_line_n1_surrogate": "E",
}

EXCLUDED_CHAPTER_FORMULATIONS = {"ed_line_n1_surrogate_redispatch"}


def _write_table_bundle(stem: str, df: pd.DataFrame, *, tables_dir: Path = TABLES_DIR) -> dict[str, str]:
    csv_path = tables_dir / f"{stem}.csv"
    parquet_path = tables_dir / f"{stem}.parquet"
    md_path = tables_dir / f"{stem}.md"
    tex_path = tables_dir / f"{stem}.tex"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_and_parquet(df, csv_path, parquet_path)
    write_markdown_table(md_path, df)
    write_latex_table(tex_path, df)
    return {"csv": str(csv_path), "parquet": str(parquet_path), "markdown": str(md_path), "latex": str(tex_path)}


def _plot_label(formulation_id: str, fallback: str | None = None) -> str:
    key = str(formulation_id)
    return FORMULATION_PLOT_LABELS.get(key, fallback or key)


def _legend_label(formulation_id: str) -> str:
    key = str(formulation_id)
    return FORMULATION_LEGEND_LABELS.get(key, key)


def _formulation_letter(formulation_id: str) -> str:
    key = str(formulation_id)
    return FORMULATION_LETTERS.get(key, key)


def _read_markdown_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        return pd.DataFrame()
    filtered: list[str] = []
    for idx, line in enumerate(lines):
        if idx == 1 and set(line.replace("|", "").replace(" ", "")) <= {"-"}:
            continue
        filtered.append(line)
    if not filtered:
        return pd.DataFrame()
    df = pd.read_csv(StringIO("\n".join(filtered)), sep="|", engine="python")
    df = df.loc[:, [str(col).strip() != "" and not str(col).startswith("Unnamed") for col in df.columns]]
    df.columns = [str(col).strip() for col in df.columns]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    return df


def _plot_retained_feasibility_boundary() -> list[str]:
    base = Path("results/thesis_optimization_results/results/benchmark/main")
    if not base.exists():
        return []

    families = ["global", "zone", "line"]
    family_labels = {
        "global": "Global mismatch",
        "zone": "Zone mismatch",
        "line": "Line outage",
    }
    scale_values = [0.60, 0.75, 0.80, 0.90, 1.00]
    counts: dict[tuple[str, float], tuple[int, int]] = {}
    tally: dict[tuple[str, float], list[int]] = {}

    for path in base.rglob("*_summary.json"):
        payload = json.loads(path.read_text())
        scenario_id = str(payload.get("scenario_id") or path.parts[-3])
        status = str(payload.get("status") or "").lower()
        if scenario_id.startswith("global_"):
            family = "global"
        elif scenario_id.startswith("zone_"):
            family = "zone"
        elif scenario_id.startswith("line_"):
            family = "line"
        else:
            continue
        scale = None
        for token in scenario_id.split("_"):
            if token.startswith("b") and "p" in token:
                scale = float(token[1:].replace("p", "."))
                break
        if scale is None:
            continue
        key = (family, scale)
        tally.setdefault(key, [0, 0])
        tally[key][1] += 1
        if status == "optimal":
            tally[key][0] += 1

    if not tally:
        return []

    rates = np.full((len(families), len(scale_values)), np.nan, dtype=float)
    labels = np.empty((len(families), len(scale_values)), dtype=object)
    for i, family in enumerate(families):
        for j, scale in enumerate(scale_values):
            vals = tally.get((family, scale))
            if vals is None:
                labels[i, j] = "n/a"
                continue
            feasible, total = vals
            rates[i, j] = feasible / total if total else np.nan
            labels[i, j] = f"{feasible}/{total}"

    set_thesis_style()
    cmap = plt.cm.get_cmap("RdYlGn").copy()
    cmap.set_bad("#ece7df")
    fig, ax = plt.subplots(figsize=(8.6, 3.8), constrained_layout=True)
    im = ax.imshow(rates, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(scale_values), dtype=float))
    ax.set_xticklabels([f"{v:.2f}" for v in scale_values])
    ax.set_yticks(np.arange(len(families), dtype=float))
    ax.set_yticklabels([family_labels[f] for f in families])
    ax.set_xlabel("Base scale")
    ax.set_title("Retained Formulation Feasibility by Scenario Family and Base Loading")
    for i in range(len(families)):
        for j in range(len(scale_values)):
            text = labels[i, j]
            color = "#1f1d1b" if np.isnan(rates[i, j]) or rates[i, j] < 0.7 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color, fontweight="semibold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Feasibility rate")
    paths = save_figure(fig, "retained_feasibility_map", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_zone_targeted_md_schedule() -> list[str]:
    path = TABLES_DIR / "zone_mismatch_vis_sensitivity_by_scenario.md"
    df = _read_markdown_table(path)
    if df.empty:
        return []
    df = df.loc[df["formulation_id"].isin(["retained_vis", "retained_vis_area_tied"])].copy()
    if df.empty:
        return []

    scenario_order = ["global_uniform", "zone_owner_1", "zone_owner_2", "zone_owner_3", "zone_owner_4"]
    scenario_labels = {
        "global_uniform": "Global",
        "zone_owner_1": "Zone 1",
        "zone_owner_2": "Zone 2",
        "zone_owner_3": "Zone 3",
        "zone_owner_4": "Zone 4",
    }
    label_map = {
        "retained_vis": "Retained VIS",
        "retained_vis_area_tied": "Area-tied VIS",
    }
    color_map = {
        "retained_vis": "#2f7f6d",
        "retained_vis_area_tied": "#8f6f4b",
    }

    set_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    metrics = [("M_mean", "Mean scheduled inertia M [p.u.]"), ("D_mean", "Mean scheduled damping D [p.u.]")]
    width = 0.36
    x = np.arange(len(scenario_order), dtype=float)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, formulation_id in enumerate(["retained_vis", "retained_vis_area_tied"]):
            subset = (
                df.loc[df["formulation_id"] == formulation_id, ["scenario_id", metric]]
                .set_index("scenario_id")
                .reindex(scenario_order)
            )
            vals = pd.to_numeric(subset[metric], errors="coerce").fillna(0.0).to_numpy()
            offset = (idx - 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=label_map[formulation_id],
                color=color_map[formulation_id],
                alpha=0.92,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([scenario_labels[sid] for sid in scenario_order])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
        if metric == "M_mean":
            ax.set_ylim(0.0, 8.0)
        elif metric == "D_mean":
            ax.set_ylim(0.0, 6.0)
    for ax in axes:
        ax.set_xlabel("Targeted mismatch scenario")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    paths = save_figure(fig, "zone_targeted_md_schedule", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_zone_targeted_md_unit_allocation() -> list[str]:
    """Heatmap of per-unit M and D allocations across zone-targeted scenarios (retained VIS only)."""
    path = TABLES_DIR / "zone_mismatch_vis_sensitivity_unit_allocations.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    df = df.loc[
        (df["row_type"] == "ibr_summary") & (df["formulation_id"] == "retained_vis")
    ].copy()
    if df.empty:
        return []

    scenario_order = ["global_uniform", "zone_owner_1", "zone_owner_2", "zone_owner_3", "zone_owner_4"]
    scenario_labels = {
        "global_uniform": "Global",
        "zone_owner_1": "Zone 1",
        "zone_owner_2": "Zone 2",
        "zone_owner_3": "Zone 3",
        "zone_owner_4": "Zone 4",
    }
    unit_order = ["REGCV1_1", "REGCV1_2", "REGCV1_3", "REGCV1_4"]
    unit_labels = ["REGCV1_1\n(WEST)", "REGCV1_2\n(EAST)", "REGCV1_3\n(WEST)", "REGCV1_4\n(EAST)"]

    # Build M and D matrices: rows = scenarios, cols = units
    m_matrix = np.zeros((len(scenario_order), len(unit_order)))
    d_matrix = np.zeros((len(scenario_order), len(unit_order)))
    for i, scen in enumerate(scenario_order):
        for j, unit in enumerate(unit_order):
            row = df.loc[(df["scenario_id"] == scen) & (df["regcv1_name"] == unit)]
            if not row.empty:
                m_matrix[i, j] = float(row["M_opt"].iloc[0])
                d_matrix[i, j] = float(row["D_opt"].iloc[0])

    set_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), constrained_layout=True)

    for ax, matrix, cbar_label, vmax in zip(
        axes,
        [m_matrix, d_matrix],
        ["Scheduled inertia $M$ [p.u.]", "Scheduled damping $D$ [p.u.]"],
        [8.0, 6.0],
    ):
        im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(unit_order)))
        ax.set_xticklabels(unit_labels, fontsize=8.5)
        ax.set_yticks(range(len(scenario_order)))
        ax.set_yticklabels([scenario_labels[s] for s in scenario_order])
        ax.set_xlabel("IBR unit")
        # Annotate cells
        for si in range(len(scenario_order)):
            for ui in range(len(unit_order)):
                val = matrix[si, ui]
                color = "white" if val > vmax * 0.55 else "#2b2826"
                ax.text(ui, si, f"{val:.1f}", ha="center", va="center", fontsize=9, color=color, fontweight="semibold")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=8)

    paths = save_figure(fig, "zone_targeted_md_unit_allocation", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_mtlsh_topk_screening() -> list[str]:
    base = Path("results/thesis_optimization_results/results/mtlsh_topk_screening/global")
    if not base.exists():
        return []
    order = [
        "ed_line_n1_surrogate_screen_top01",
        "ed_line_n1_surrogate_screen_top03",
        "ed_line_n1_surrogate_screen_top05",
        "ed_line_n1_surrogate_screen_all10",
    ]
    short = {
        "ed_line_n1_surrogate_screen_top01": "Top 1",
        "ed_line_n1_surrogate_screen_top03": "Top 3",
        "ed_line_n1_surrogate_screen_top05": "Top 5",
        "ed_line_n1_surrogate_screen_all10": "All 10",
    }
    rows: dict[str, dict[str, tuple[str, float | None]]] = {}
    for path in base.rglob("*_summary.json"):
        payload = json.loads(path.read_text())
        formulation_id = str(payload.get("formulation_id") or path.parts[-2])
        scenario_id = str(payload.get("scenario_id") or path.parts[-3])
        if formulation_id not in short:
            continue
        rows.setdefault(scenario_id, {})[formulation_id] = (
            str(payload.get("status") or "").lower(),
            payload.get("objective"),
        )
    if not rows:
        return []

    feasible_counts = []
    total_counts = []
    common_means = []
    common = []
    for scenario_id, vals in rows.items():
        if len(vals) == len(order) and all(v[0] == "optimal" and v[1] is not None for v in vals.values()):
            common.append(vals)
    for formulation_id in order:
        statuses = [vals.get(formulation_id, ("", None))[0] for vals in rows.values()]
        feasible_counts.append(sum(status == "optimal" for status in statuses))
        total_counts.append(sum(status != "" for status in statuses))
        common_vals = [float(vals[formulation_id][1]) for vals in common]
        common_means.append(float(np.mean(common_vals)) if common_vals else np.nan)

    set_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    colors = ["#4c6a92", "#7a8da6", "#8f6f4b", "#2f7f6d"]
    x = np.arange(len(order), dtype=float)

    bars = axes[0].bar(x, feasible_counts, color=colors, alpha=0.92)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([short[fid] for fid in order])
    axes[0].set_xlabel("Screened line-contingency set")
    axes[0].set_ylabel("Feasible runs")
    axes[0].set_ylim(0, max(total_counts) + 2)
    for bar, feasible, total in zip(bars, feasible_counts, total_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4, f"{feasible}/{total}", ha="center", va="bottom", fontsize=8.5)

    bars = axes[1].bar(x, common_means, color=colors, alpha=0.92)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([short[fid] for fid in order])
    axes[1].set_xlabel("Screened line-contingency set")
    axes[1].set_ylabel("Mean total objective on common feasible subset")
    for bar, value in zip(bars, common_means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}", ha="center", va="bottom", fontsize=8.5)

    paths = save_figure(fig, "mtlsh_topk_screening", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_cost_impact(cost_df: pd.DataFrame) -> list[str]:
    feasible = cost_df.loc[cost_df["is_feasible"] >= 0.5].copy()
    feasible = feasible.loc[~feasible["formulation_id"].astype(str).isin(EXCLUDED_CHAPTER_FORMULATIONS)].copy()
    if feasible.empty:
        return []

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.9), constrained_layout=True)
    scenario_count = feasible["scenario_id"].nunique()
    order = formulation_order()
    if scenario_count <= 1:
        plot_df = feasible.copy()
        plot_df["formulation_id"] = pd.Categorical(plot_df["formulation_id"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("formulation_id")
        yvals = pd.to_numeric(plot_df["dispatch_cost_component"], errors="coerce").fillna(0.0).to_numpy()
        labels = [_formulation_letter(fid) for fid in plot_df["formulation_id"].astype(str)]
        colors = [formulation_color(fid) for fid in plot_df["formulation_id"].astype(str)]
        xpos = np.arange(plot_df.shape[0], dtype=float)
        bars = ax.bar(xpos, yvals, color=colors, alpha=0.92, width=0.72)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("One-hour dispatch-cost component [$]")
        ax.set_xlabel("")
        ax.set_title("Dispatch Cost by Formulation")
        ymin = 10000.0
        ymax = max(float(np.nanmax(yvals)) if len(yvals) else ymin, ymin)
        ax.set_ylim(ymin, ymax * 1.08)
        for bar, value in zip(bars, yvals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max((ymax - ymin) * 0.015, 120.0),
                f"{value:.0f}",
                va="bottom",
                ha="center",
                fontsize=9,
                color="#2b2826",
            )
        legend_lines = [
            f"{_formulation_letter(fid)} = {_plot_label(fid, name)}"
            for fid, name in zip(plot_df["formulation_id"].astype(str), plot_df["formulation_name"].astype(str))
        ]
        ax.text(
            0.98,
            0.98,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#2b2826",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fbf8f2", "edgecolor": "#d6cec2", "alpha": 0.96},
        )
    else:
        grouped = (
            feasible.groupby(["formulation_id", "formulation_name"], dropna=False)
            .agg(
                mean_total_cost=("total_cost", "mean"),
                min_total_cost=("total_cost", "min"),
                max_total_cost=("total_cost", "max"),
            )
            .reset_index()
        )
        grouped["formulation_id"] = pd.Categorical(grouped["formulation_id"], categories=order, ordered=True)
        grouped = grouped.sort_values("formulation_id")
        x = np.arange(grouped.shape[0], dtype=float)
        y = pd.to_numeric(grouped["mean_total_cost"], errors="coerce").to_numpy()
        yerr = np.vstack(
            [
                y - pd.to_numeric(grouped["min_total_cost"], errors="coerce").to_numpy(),
                pd.to_numeric(grouped["max_total_cost"], errors="coerce").to_numpy() - y,
            ]
        )
        colors = [formulation_color(fid) for fid in grouped["formulation_id"].astype(str)]
        ax.bar(x, y, color=colors, alpha=0.9, yerr=yerr, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([_formulation_letter(fid) for fid in grouped["formulation_id"].astype(str)])
        ax.set_ylabel("Mean total objective")
        ax.set_title("Total Cost Across Scenarios")
        legend_lines = [
            f"{_formulation_letter(fid)} = {_plot_label(fid, name)}"
            for fid, name in zip(grouped["formulation_id"].astype(str), grouped["formulation_name"].astype(str))
        ]
        ax.text(
            0.98,
            0.98,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#2b2826",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fbf8f2", "edgecolor": "#d6cec2", "alpha": 0.96},
        )

    ax.grid(axis="y", alpha=0.18)
    paths = save_figure(fig, "cost_impact_by_formulation", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_dispatch_vis(generator_df: pd.DataFrame, ibr_df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    if generator_df.empty or ibr_df.empty:
        return []

    order = formulation_order(cfg)
    baseline_id = baseline_formulation_id(cfg)
    retained_id = retained_formulation_id(cfg)
    default_formulations = [
        fid
        for fid in [baseline_id, "ed_line", "ed_line_n1", "ed_surrogate", retained_id]
        if fid in set(generator_df["formulation_id"])
    ]
    selected_formulations = list(cfg.get("dispatch_figure_formulations") or default_formulations)
    available_scenarios = [value for value in generator_df["scenario_id"].dropna().astype(str).unique().tolist() if value]
    scenario_id = str(cfg.get("dispatch_figure_scenario_id") or (available_scenarios[0] if available_scenarios else ""))
    if not scenario_id:
        return []

    gen_plot = generator_df.loc[
        (generator_df["scenario_id"].astype(str) == scenario_id)
        & (generator_df["formulation_id"].astype(str).isin(selected_formulations))
    ].copy()
    ibr_plot = ibr_df.loc[
        (ibr_df["scenario_id"].astype(str) == scenario_id)
        & (ibr_df["formulation_id"].astype(str).isin(selected_formulations))
    ].copy()
    if gen_plot.empty or ibr_plot.empty:
        return []

    set_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2), constrained_layout=True)
    ax_dispatch = axes[0, 0]
    ax_headroom = axes[0, 1]
    ax_m = axes[1, 0]
    ax_d = axes[1, 1]

    def _grouped_bar(ax, df, *, x_col: str, y_col: str, title: str, ylabel: str) -> None:
        ids = [fid for fid in selected_formulations if fid in set(df["formulation_id"].astype(str))]
        x_values = sorted(df[x_col].dropna().astype(int).unique().tolist())
        if not ids or not x_values:
            ax.set_visible(False)
            return
        x = np.arange(len(x_values), dtype=float)
        width = 0.8 / max(len(ids), 1)
        scale = 100.0 if y_col in {"pg_delta_vs_baseline", "headroom_up_vs_baseline"} else 1.0
        for idx, formulation_id in enumerate(ids):
            subset = df.loc[df["formulation_id"].astype(str) == formulation_id].copy()
            subset = subset.set_index(x_col)
            values = (
                pd.to_numeric(subset.reindex(x_values)[y_col], errors="coerce").fillna(0.0).to_numpy() * scale
            )
            offset = (idx - (len(ids) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=_legend_label(formulation_id),
                color=formulation_color(formulation_id),
                alpha=0.9,
            )
        ax.set_xticks(x)
        prefix = "G" if x_col == "index" and len(x_values) > 4 and title.startswith("Dispatch") else "IBR"
        ax.set_xticklabels([f"{prefix}{int(v) + 1}" for v in x_values])
        ax.set_xlabel("Generator unit" if prefix == "G" else "IBR unit")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0)

    _grouped_bar(
        ax_dispatch,
        gen_plot,
        x_col="index",
        y_col="pg_delta_vs_baseline",
        title="Dispatch Shift vs ED",
        ylabel="Dispatch Delta [MW]",
    )
    _grouped_bar(
        ax_headroom,
        ibr_plot,
        x_col="index",
        y_col="headroom_up_vs_baseline",
        title="IBR Upward Headroom Shift",
        ylabel="Headroom Delta [MW]",
    )
    _grouped_bar(ax_m, ibr_plot, x_col="index", y_col="M_opt", title="VIS Inertia Allocation", ylabel="M [p.u.]")
    _grouped_bar(ax_d, ibr_plot, x_col="index", y_col="D_opt", title="VIS Damping Allocation", ylabel="D [p.u.]")
    ax_m.set_ylim(0.0, 8.0)
    ax_d.set_ylim(0.0, 6.0)

    ax_dispatch.axhline(0.0, color="#777777", linewidth=0.9, linestyle=":")
    ax_headroom.axhline(0.0, color="#777777", linewidth=0.9, linestyle=":")
    ax_m.axhline(4.0, color="#777777", linewidth=0.9, linestyle=":")
    ax_d.axhline(2.0, color="#777777", linewidth=0.9, linestyle=":")

    handles, labels = ax_dispatch.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 3),
            bbox_to_anchor=(0.5, 1.08),
            fontsize=10.3,
            handlelength=1.8,
            columnspacing=1.2,
            frameon=False,
        )
    paths = save_figure(fig, f"dispatch_vis_comparison_{scenario_id}", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_predicted_vs_replayed(detail_df: pd.DataFrame) -> list[str]:
    if detail_df.empty:
        return []
    detail_df = detail_df.loc[~detail_df["formulation_id"].astype(str).isin(EXCLUDED_CHAPTER_FORMULATIONS)].copy()
    retained_id = retained_formulation_id()
    detail_df = detail_df.loc[detail_df["formulation_id"].astype(str) == retained_id].copy()
    if detail_df.empty:
        return []

    metrics = detail_df["metric_name"].dropna().astype(str).unique().tolist()
    if not metrics:
        return []
    save_plot_data("predicted_vs_replayed_metrics", detail_df)

    set_thesis_style()
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    order = [retained_id]
    def _unsafe_mask(frame: pd.DataFrame) -> pd.Series:
        violated = (
            pd.to_numeric(frame["violated_in_replay"], errors="coerce")
            if "violated_in_replay" in frame.columns
            else pd.Series(0.0, index=frame.index, dtype=float)
        )
        false_safe = (
            pd.to_numeric(frame["false_safe_flag"], errors="coerce")
            if "false_safe_flag" in frame.columns
            else pd.Series(0.0, index=frame.index, dtype=float)
        )
        return (violated.fillna(0.0) >= 0.5) | (false_safe.fillna(0.0) >= 0.5)

    def _series_or_default(frame: pd.DataFrame, column: str) -> pd.Series:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    for ax, metric in zip(axes_arr.ravel(), metrics):
        subset = detail_df.loc[detail_df["metric_name"].astype(str) == metric].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        subset = subset.loc[~_unsafe_mask(subset)].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        limit_low = _series_or_default(subset, "limit_low").dropna()
        limit_high = _series_or_default(subset, "limit_high").dropna()
        low_bound = float(limit_low.iloc[0]) if not limit_low.empty else np.nan
        high_bound = float(limit_high.iloc[0]) if not limit_high.empty else np.nan
        for formulation_id in [fid for fid in order if fid in set(subset["formulation_id"].astype(str))]:
            frame = subset.loc[subset["formulation_id"].astype(str) == formulation_id]
            scale = 100.0 if str(metric).startswith("Delta_P_IBR_") else 1.0
            x_vals = pd.to_numeric(frame["predicted_value"], errors="coerce") * scale
            y_vals = pd.to_numeric(frame["replayed_value"], errors="coerce") * scale
            ax.scatter(
                x_vals,
                y_vals,
                s=48,
                alpha=0.9,
                color=formulation_color(formulation_id),
                marker=FORMULATION_MARKERS.get(formulation_id, "o"),
                label=_legend_label(formulation_id),
                edgecolors="white",
                linewidths=0.6,
            )
        vals = pd.concat(
            [
                pd.to_numeric(subset["predicted_value"], errors="coerce") * scale,
                pd.to_numeric(subset["replayed_value"], errors="coerce") * scale,
            ],
            axis=0,
        ).dropna()
        if not vals.empty:
            lo = float(vals.min())
            hi = float(vals.max())
            if np.isfinite(low_bound):
                lo = min(lo, low_bound * scale)
            if np.isfinite(high_bound):
                hi = max(hi, high_bound * scale)
            pad = 0.08 * (hi - lo) if hi > lo else 0.1 * max(1.0, abs(lo), abs(hi))
            lo -= pad
            hi += pad
            if np.isfinite(low_bound) and np.isfinite(high_bound) and high_bound > low_bound:
                rect = Rectangle(
                    (low_bound * scale, low_bound * scale),
                    (high_bound - low_bound) * scale,
                    (high_bound - low_bound) * scale,
                    facecolor="#dff0df",
                    edgecolor="#7ea37e",
                    linewidth=0.9,
                    alpha=0.18,
                    zorder=0,
                )
                ax.add_patch(rect)
                ax.axvline(low_bound * scale, color="#7ea37e", linewidth=0.9, linestyle=":")
                ax.axvline(high_bound * scale, color="#7ea37e", linewidth=0.9, linestyle=":")
                ax.axhline(low_bound * scale, color="#7ea37e", linewidth=0.9, linestyle=":")
                ax.axhline(high_bound * scale, color="#7ea37e", linewidth=0.9, linestyle=":")
            ax.plot([lo, hi], [lo, hi], color="#333333", linestyle="--", linewidth=1.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        axis_label = _metric_axis_label(metric)
        ax.set_xlabel(f"Predicted {axis_label}")
        ax.set_ylabel(f"Replayed {axis_label}")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_arr.ravel()[len(metrics) :]:
        ax.set_visible(False)
    paths = save_figure(fig, "predicted_vs_replayed_metrics", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_replay_frequency_by_formulation(opt_master: pd.DataFrame, replay_master: pd.DataFrame) -> list[str]:
    if opt_master.empty or replay_master.empty:
        return []

    feasible_runs = opt_master.loc[
        (opt_master["source_group"].astype(str) == "benchmark_cross_method_subset")
        & (opt_master["solver_status"].astype(str).str.startswith("optimal")),
        ["scenario_id", "formulation_id", "formulation_name"],
    ].drop_duplicates()
    if feasible_runs.empty:
        return []

    plot_df = replay_master.loc[
        replay_master["source_group"].astype(str) == "benchmark_replay_cross_method_subset"
    ].copy()
    plot_df = plot_df.merge(
        feasible_runs,
        left_on=["scenario_id", "formulation"],
        right_on=["scenario_id", "formulation_id"],
        how="inner",
    )
    plot_df = plot_df.loc[
        ~plot_df["formulation"].astype(str).isin(EXCLUDED_CHAPTER_FORMULATIONS)
    ].copy()
    if plot_df.empty:
        return []

    plot_df["abs_replayed_value"] = pd.to_numeric(plot_df["replayed_value"], errors="coerce").abs()
    plot_df = plot_df.loc[np.isfinite(plot_df["abs_replayed_value"])].copy()
    if plot_df.empty:
        return []
    save_plot_data("replay_frequency_by_formulation", plot_df)

    retained_id = retained_formulation_id()
    plot_df = plot_df.loc[plot_df["formulation"].astype(str) == retained_id].copy()
    if plot_df.empty:
        return []

    metric_order = [
        metric
        for metric in ["rocof_COI", "dev_COI", "Delta_P_IBR_1", "Delta_P_IBR_2", "Delta_P_IBR_3", "Delta_P_IBR_4"]
        if metric in set(plot_df["metric_name"].astype(str))
    ]
    if not metric_order:
        return []

    rng = np.random.default_rng(42)
    set_thesis_style()
    fig, ax_rocof = plt.subplots(1, 1, figsize=(11.6, 4.9), constrained_layout=True)
    ax_dp = ax_rocof.twinx()
    ax_dev = ax_rocof.twinx()
    ax_dev.spines["right"].set_visible(False)
    ax_dev.spines["left"].set_visible(True)
    ax_dev.yaxis.set_label_position("left")
    ax_dev.yaxis.tick_left()
    ax_dev.spines["left"].set_position(("outward", 72))

    xpos = np.arange(len(metric_order), dtype=float)
    rocof_max = 0.0
    dev_max = 0.0
    dp_max = 0.0

    for idx, metric_name in enumerate(metric_order):
        metric_df = plot_df.loc[plot_df["metric_name"].astype(str) == metric_name].copy()
        if metric_df.empty:
            continue
        scale = 100.0 if str(metric_name).startswith("Delta_P_IBR_") else 1.0
        if metric_name == "rocof_COI":
            target_ax = ax_rocof
        elif metric_name == "dev_COI":
            target_ax = ax_dev
        else:
            target_ax = ax_dp
        limit_low = (
            pd.to_numeric(metric_df["limit_low"], errors="coerce")
            if "limit_low" in metric_df.columns
            else pd.Series(np.nan, index=metric_df.index, dtype=float)
        )
        limit_high = (
            pd.to_numeric(metric_df["limit_high"], errors="coerce")
            if "limit_high" in metric_df.columns
            else pd.Series(np.nan, index=metric_df.index, dtype=float)
        )
        limit_candidates = pd.concat(
            [
                limit_low.abs(),
                limit_high.abs(),
            ],
            axis=0,
        ).dropna()
        limit_value = float(limit_candidates.max()) if not limit_candidates.empty else np.nan
        vals = pd.to_numeric(metric_df["abs_replayed_value"], errors="coerce").dropna() * scale
        if vals.empty:
            continue
        bp = target_ax.boxplot(
            [vals.to_numpy()],
            positions=[xpos[idx]],
            widths=0.58,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(formulation_color(retained_id))
            patch.set_alpha(0.55)
            patch.set_edgecolor("#2b2826")
        for median in bp["medians"]:
            median.set_color("#2b2826")
            median.set_linewidth(1.2)
        jitter = rng.normal(loc=xpos[idx], scale=0.045, size=vals.shape[0])
        target_ax.scatter(
            jitter,
            vals.to_numpy(),
            s=26,
            alpha=0.78,
            color=formulation_color(retained_id),
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        peak = float(vals.max())
        limit_peak = (limit_value * scale) if np.isfinite(limit_value) else 0.0
        if metric_name == "rocof_COI":
            rocof_max = max(rocof_max, peak, limit_peak)
        elif metric_name == "dev_COI":
            dev_max = max(dev_max, peak, limit_peak)
        else:
            dp_max = max(dp_max, peak, limit_peak)

    ax_rocof.set_xticks(xpos)
    ax_rocof.set_xticklabels([METRIC_LABELS.get(metric, metric) for metric in metric_order], rotation=18, ha="right")
    ax_rocof.set_ylabel(r"$|\mathrm{RoCoF}_{\mathrm{COI}}|$ [Hz/s]")
    ax_dev.set_ylabel(r"$|\Delta f_{\mathrm{COI}}|$ [Hz]")
    ax_dp.set_ylabel(r"$|\Delta P_{\mathrm{IBR}}|$ [MW]")
    ax_rocof.grid(False)
    ax_rocof.set_ylim(0.0, rocof_max * 1.08 if rocof_max > 0 else 1.0)
    ax_dev.set_ylim(0.0, dev_max * 1.08 if dev_max > 0 else 1.0)
    ax_dp.set_ylim(0.0, dp_max * 1.08 if dp_max > 0 else 1.0)
    paths = save_figure(fig, "replay_frequency_by_formulation", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_violation_breakdown(replay_summary_df: pd.DataFrame) -> list[str]:
    if replay_summary_df.empty:
        return []
    plot_df = replay_summary_df.copy()
    plot_df = plot_df.loc[~plot_df["formulation_id"].astype(str).isin(EXCLUDED_CHAPTER_FORMULATIONS)].copy()
    plot_df = plot_df.loc[plot_df["status_from_optimization"].astype(str).str.startswith("optimal")].copy()
    focus_order = [fid for fid in ["ed_surrogate", retained_formulation_id()] if fid in set(plot_df["formulation_id"].astype(str))]
    if not focus_order:
        focus_order = [fid for fid in formulation_order() if fid in set(plot_df["formulation_id"].astype(str))]
    plot_df = plot_df.loc[plot_df["formulation_id"].astype(str).isin(focus_order)].copy()
    if plot_df.empty:
        return []

    for col in [
        "replay_violation_count",
        "n_false_safe",
        "scheduled_headroom_exceedance_count",
        "physical_limit_exceedance_count",
    ]:
        if col not in plot_df.columns:
            plot_df[col] = 0

    summary = (
        plot_df.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            feasible_replayed_runs=("run_id", "count"),
            replay_violation_runs=("replay_violation_count", lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(0.0) > 0))),
            false_safe_runs=("n_false_safe", lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(0.0) > 0))),
            headroom_exceed_runs=("scheduled_headroom_exceedance_count", lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(0.0) > 0))),
            physical_limit_exceed_runs=("physical_limit_exceedance_count", lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(0.0) > 0))),
        )
        .reset_index()
    )
    summary["replay_safe_runs"] = (
        summary["feasible_replayed_runs"]
        - summary["replay_violation_runs"]
        - summary["false_safe_runs"]
    ).clip(lower=0)
    summary["replay_safe_rate"] = np.where(
        pd.to_numeric(summary["feasible_replayed_runs"], errors="coerce").fillna(0.0) > 0,
        pd.to_numeric(summary["replay_safe_runs"], errors="coerce")
        / pd.to_numeric(summary["feasible_replayed_runs"], errors="coerce"),
        np.nan,
    )
    summary["formulation_id"] = pd.Categorical(summary["formulation_id"], categories=focus_order, ordered=True)
    summary = summary.sort_values("formulation_id")

    set_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    ax_safe, ax_issue = axes

    labels = [_plot_label(fid, name) for fid, name in zip(summary["formulation_id"].astype(str), summary["formulation_name"].astype(str))]
    y = np.arange(summary.shape[0], dtype=float)
    safe_pct = 100.0 * pd.to_numeric(summary["replay_safe_rate"], errors="coerce").fillna(0.0).to_numpy()
    unsafe_pct = 100.0 - safe_pct
    ax_safe.barh(y, np.full_like(y, 100.0), color="#ece7dd", alpha=0.9, height=0.58)
    ax_safe.barh(y, safe_pct, color="#2f7f6d", alpha=0.95, height=0.58, label="Replay-safe")
    ax_safe.barh(y, unsafe_pct, left=safe_pct, color="#c46646", alpha=0.92, height=0.58, label="Replay-bad")
    ax_safe.set_xlim(0.0, 100.0)
    ax_safe.set_yticks(y)
    ax_safe.set_yticklabels(labels)
    ax_safe.set_xlabel("Replay-safe share [%]")
    ax_safe.set_title("Replay Safety by Formulation")
    ax_safe.invert_yaxis()
    ax_safe.legend(frameon=False, loc="lower right")
    for yi, safe, n_safe, n_total in zip(
        y,
        safe_pct,
        pd.to_numeric(summary["replay_safe_runs"], errors="coerce").fillna(0.0).astype(int),
        pd.to_numeric(summary["feasible_replayed_runs"], errors="coerce").fillna(0.0).astype(int),
    ):
        ax_safe.text(
            min(safe + 2.0, 98.0),
            yi,
            f"{n_safe}/{n_total}",
            va="center",
            ha="left" if safe < 94 else "right",
            fontsize=8.8,
            color="#2b2826",
        )

    x = np.arange(summary.shape[0], dtype=float)
    width = 0.22
    replay_bad = pd.to_numeric(summary["replay_violation_runs"], errors="coerce").fillna(0.0).to_numpy()
    false_safe = pd.to_numeric(summary["false_safe_runs"], errors="coerce").fillna(0.0).to_numpy()
    headroom = pd.to_numeric(summary["headroom_exceed_runs"], errors="coerce").fillna(0.0).to_numpy()
    ax_issue.bar(x - width, replay_bad, width=width, color="#c46646", label="Replay violations")
    ax_issue.bar(x, false_safe, width=width, color="#d89c3d", label="False-safe runs")
    ax_issue.bar(x + width, headroom, width=width, color="#7a8da6", label="Headroom exceedances")
    ax_issue.set_xticks(x)
    ax_issue.set_xticklabels(labels, rotation=16, ha="right")
    ax_issue.set_ylabel("Run count [-]")
    ax_issue.set_title("Residual Issues")
    ax_issue.legend(frameon=False, loc="upper right")
    for xpos, values in [(x - width, replay_bad), (x, false_safe), (x + width, headroom)]:
        for xx, val in zip(xpos, values):
            ax_issue.text(xx, val, f"{int(val)}", ha="center", va="bottom", fontsize=8.5, color="#2b2826")

    paths = save_figure(fig, "constraint_satisfaction_breakdown", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _metric_axis_label(metric_name: str) -> str:
    if metric_name == "rocof_COI":
        return "RoCoF [Hz/s]"
    if metric_name == "dev_COI":
        return "Frequency Deviation [Hz]"
    if str(metric_name).startswith("Delta_P_IBR_"):
        return "IBR Power Excursion [MW]"
    return str(metric_name)


def _plot_cost_breakdown(method_df: pd.DataFrame) -> list[str]:
    if method_df.empty:
        return []
    plot_df = method_df.copy()
    order = formulation_order()
    plot_df["formulation_id"] = pd.Categorical(plot_df["formulation_id"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("formulation_id")
    save_plot_data("cost_breakdown_by_formulation", plot_df)

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    x = np.arange(plot_df.shape[0], dtype=float)
    dispatch = pd.to_numeric(plot_df["dispatch_cost_mean"], errors="coerce").fillna(0.0).to_numpy()
    reserve_up = pd.to_numeric(plot_df["reserve_up_cost_mean"], errors="coerce").fillna(0.0).to_numpy()
    reserve_post = pd.to_numeric(plot_df["reserve_postcont_cost_mean"], errors="coerce").fillna(0.0).to_numpy()
    total = pd.to_numeric(plot_df["total_cost_mean"], errors="coerce").fillna(0.0).to_numpy()
    ax.bar(x, dispatch, color="#4c6a92", label="Dispatch cost")
    ax.bar(x, reserve_up, bottom=dispatch, color="#c46646", label="Reserve-up cost")
    ax.bar(x, reserve_post, bottom=dispatch + reserve_up, color="#2f7f6d", label="Post-contingency reserve cost")
    ax.plot(x, total, color="#2b2826", marker="o", linewidth=1.2, label="Total cost")
    ax.set_xticks(x)
    ax.set_xticklabels([_plot_label(fid, name) for fid, name in zip(plot_df["formulation_id"].astype(str), plot_df["formulation_name"].astype(str))], rotation=18, ha="right")
    ax.set_ylabel("Single-Interval Scheduling Cost")
    ax.set_title("Cost Breakdown by Formulation")
    ax.legend(ncol=2, frameon=False)
    paths = save_figure(fig, "cost_breakdown_by_formulation", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_reserve_headroom_by_ibr(opt_master: pd.DataFrame, replay_master: pd.DataFrame) -> list[str]:
    if opt_master.empty:
        return []
    rows: list[pd.DataFrame] = []
    for _, run in opt_master.iterrows():
        path_raw = str(run.get("dispatch_impact_csv", "")).strip()
        if not path_raw:
            continue
        path = Path(path_raw)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [str(col).strip() for col in df.columns]
        if "row_type" not in df.columns:
            continue
        df = df.loc[df["row_type"].astype(str) == "ibr_summary"].copy()
        if df.empty:
            continue
        df["formulation_id"] = run.get("formulation_id")
        df["formulation_name"] = run.get("formulation_name")
        df["scenario_id"] = run.get("scenario_id")
        rows.append(df)
    if not rows:
        return []
    ibr_df = pd.concat(rows, ignore_index=True, sort=False)
    for col in ("headroom_up", "reserve_up"):
        ibr_df[col] = pd.to_numeric(ibr_df.get(col), errors="coerce")

    replay_df = replay_master.copy() if not replay_master.empty else pd.DataFrame()
    if not replay_df.empty:
        replay_df = replay_df.loc[replay_df["metric_name"].astype(str).str.startswith("Delta_P_IBR_")].copy()
        replay_df["index"] = replay_df["metric_name"].astype(str).str.extract(r"(\d+)$").astype(float) - 1.0
        replay_agg = (
            replay_df.groupby(["formulation", "index"], dropna=False)
            .agg(max_headroom_exceedance=("scheduled_headroom_violation_magnitude", "max"))
            .reset_index()
            .rename(columns={"formulation": "formulation_id"})
        )
    else:
        replay_agg = pd.DataFrame(columns=["formulation_id", "index", "max_headroom_exceedance"])

    plot_df = (
        ibr_df.groupby(["formulation_id", "formulation_name", "index"], dropna=False)
        .agg(
            reserve_up_mean=("reserve_up", "mean"),
            headroom_up_mean=("headroom_up", "mean"),
        )
        .reset_index()
        .merge(replay_agg, on=["formulation_id", "index"], how="left")
    )
    save_plot_data("reserve_headroom_by_ibr", plot_df)

    order = [fid for fid in formulation_order() if fid in set(plot_df["formulation_id"].astype(str))]
    x_values = sorted(plot_df["index"].dropna().astype(int).unique().tolist())
    if not order or not x_values:
        return []

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    x = np.arange(len(x_values), dtype=float)
    width = 0.8 / max(len(order), 1)
    for idx, formulation_id in enumerate(order):
        sub = plot_df.loc[plot_df["formulation_id"].astype(str) == formulation_id].copy().set_index("index")
        reserve_vals = pd.to_numeric(sub.reindex(x_values)["reserve_up_mean"], errors="coerce").fillna(0.0).to_numpy()
        headroom_vals = pd.to_numeric(sub.reindex(x_values)["headroom_up_mean"], errors="coerce").fillna(0.0).to_numpy()
        exceed_vals = pd.to_numeric(sub.reindex(x_values)["max_headroom_exceedance"], errors="coerce").fillna(0.0).to_numpy()
        offset = (idx - (len(order) - 1) / 2.0) * width
        ax.bar(x + offset, headroom_vals, width=width, color=formulation_color(formulation_id), alpha=0.35)
        ax.bar(x + offset, reserve_vals, width=width * 0.68, color=formulation_color(formulation_id), alpha=0.9, label=_legend_label(formulation_id))
        ax.scatter(x + offset, reserve_vals + exceed_vals, color="#2b2826", s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"IBR {int(v) + 1}" for v in x_values])
    ax.set_ylabel("Scheduled Reserve / Headroom [p.u.]")
    ax.set_title("Reserve and Headroom by IBR")
    ax.legend(ncol=min(len(order), 3), frameon=False)
    paths = save_figure(fig, "reserve_headroom_by_ibr", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_replay_violation_counts_by_metric(replay_master: pd.DataFrame) -> list[str]:
    if replay_master.empty:
        return []
    plot_df = (
        replay_master.groupby(["metric_name", "metric_category"], dropna=False)
        .agg(
            violation_count=("violated_in_replay", "sum"),
            violation_rate=("violated_in_replay", "mean"),
        )
        .reset_index()
    )
    save_plot_data("replay_violation_counts_by_metric", plot_df)

    set_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    x = np.arange(plot_df.shape[0], dtype=float)
    axes[0].bar(x, pd.to_numeric(plot_df["violation_count"], errors="coerce").fillna(0.0), color="#c46646")
    axes[1].bar(x, pd.to_numeric(plot_df["violation_rate"], errors="coerce").fillna(0.0), color="#7a8da6")
    for ax, ylabel in zip(axes, ["Violation Count", "Violation Rate"]):
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABELS.get(name, name) for name in plot_df["metric_name"].astype(str)], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Replay Violations by Metric")
    axes[1].set_title("Replay Violation Rates")
    paths = save_figure(fig, "replay_violation_counts_by_metric", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_cost_vs_replay_safety_scatter(opt_master: pd.DataFrame, replay_master: pd.DataFrame) -> list[str]:
    if opt_master.empty or replay_master.empty:
        return []
    replay_by_run = (
        replay_master.groupby("run_id", dropna=False)
        .agg(
            replay_violation_count=("violated_in_replay", "sum"),
            replay_violation_rate=("violated_in_replay", "mean"),
        )
        .reset_index()
    )
    plot_df = opt_master.merge(replay_by_run, on="run_id", how="inner")
    if plot_df.empty:
        return []
    save_plot_data("cost_vs_replay_safety_scatter", plot_df)

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    order = formulation_order()
    for formulation_id in [fid for fid in order if fid in set(plot_df["formulation_id"].astype(str))]:
        sub = plot_df.loc[plot_df["formulation_id"].astype(str) == formulation_id]
        ax.scatter(
            pd.to_numeric(sub["objective_total"], errors="coerce"),
            pd.to_numeric(sub["replay_violation_rate"], errors="coerce"),
            color=formulation_color(formulation_id),
            alpha=0.82,
            s=42,
            label=_legend_label(formulation_id),
        )
    ax.set_xlabel("Total Cost")
    ax.set_ylabel("Replay Violation Rate")
    ax.set_title("Cost vs Replay Safety")
    ax.legend(ncol=min(len(order), 3), frameon=False)
    paths = save_figure(fig, "cost_vs_replay_safety_scatter", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_solve_time_vs_binary_count(opt_master: pd.DataFrame) -> list[str]:
    if opt_master.empty:
        return []
    plot_df = opt_master.copy()
    save_plot_data("solve_time_vs_binary_count", plot_df)

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    order = formulation_order()
    for formulation_id in [fid for fid in order if fid in set(plot_df["formulation_id"].astype(str))]:
        sub = plot_df.loc[plot_df["formulation_id"].astype(str) == formulation_id]
        ax.scatter(
            pd.to_numeric(sub["binaries"], errors="coerce"),
            pd.to_numeric(sub["solve_time_sec"], errors="coerce"),
            color=formulation_color(formulation_id),
            alpha=0.8,
            s=38,
            label=_legend_label(formulation_id),
        )
    ax.set_xlabel("Binary Variables")
    ax.set_ylabel("Solve Time [s]")
    ax.set_title("Solve Time vs Binary Count")
    ax.set_yscale("log")
    ax.legend(ncol=min(len(order), 3), frameon=False)
    paths = save_figure(fig, "solve_time_vs_binary_count", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_predicted_vs_replayed_rocof(replay_master: pd.DataFrame) -> list[str]:
    if replay_master.empty:
        return []
    plot_df = replay_master.loc[replay_master["metric_name"].astype(str) == "rocof_COI"].copy()
    if plot_df.empty:
        return []
    save_plot_data("predicted_vs_replayed_rocof", plot_df)

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.8), constrained_layout=True)
    order = formulation_order()
    vals = pd.concat(
        [pd.to_numeric(plot_df["predicted_value"], errors="coerce"), pd.to_numeric(plot_df["replayed_value"], errors="coerce")],
        axis=0,
    ).dropna()
    lo, hi = (-1.0, 1.0)
    if not vals.empty:
        lo = float(vals.min())
        hi = float(vals.max())
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        lo -= pad
        hi += pad
    for formulation_id in [fid for fid in order if fid in set(plot_df["formulation"].astype(str))]:
        sub = plot_df.loc[plot_df["formulation"].astype(str) == formulation_id]
        ax.scatter(
            pd.to_numeric(sub["predicted_value"], errors="coerce"),
            pd.to_numeric(sub["replayed_value"], errors="coerce"),
            color=formulation_color(formulation_id),
            alpha=0.82,
            s=40,
            label=_legend_label(formulation_id),
        )
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#333333", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted RoCoF [Hz/s]")
    ax.set_ylabel("Replayed RoCoF [Hz/s]")
    ax.set_title("Predicted vs Replayed RoCoF")
    ax.legend(frameon=False)
    paths = save_figure(fig, "predicted_vs_replayed_rocof", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_scenario_coverage_heatmaps(manifest_df: pd.DataFrame) -> list[str]:
    if manifest_df.empty:
        return []
    save_plot_data("scenario_coverage_heatmaps", manifest_df)

    global_df = manifest_df.loc[manifest_df["scenario_family"].astype(str) == "global_load_mismatch"].copy()
    zone_df = manifest_df.loc[manifest_df["scenario_family"].astype(str) == "zone_load_mismatch"].copy()
    line_df = manifest_df.loc[manifest_df["scenario_family"].astype(str) == "line_outage"].copy()

    set_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), constrained_layout=True)
    for ax, frame, title, index_col, col_col in [
        (axes[0], global_df, "Global Mismatch", "base_scale", "step_scale"),
        (axes[1], zone_df, "Zone Mismatch", "base_scale", "step_scale"),
        (axes[2], line_df, "Line Outage", "severity_bin", "base_scale"),
    ]:
        if frame.empty:
            ax.set_visible(False)
            continue
        pivot = frame.pivot_table(index=index_col, columns=col_col, values="scenario_id", aggfunc="count", fill_value=0)
        ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
        ax.set_xticks(np.arange(pivot.shape[1], dtype=float))
        ax.set_xticklabels([str(v) for v in pivot.columns.tolist()], rotation=18, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0], dtype=float))
        ax.set_yticklabels([str(v) for v in pivot.index.tolist()])
        ax.set_title(title)
    paths = save_figure(fig, "scenario_coverage_heatmaps", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_method_comparison_boxplots(opt_master: pd.DataFrame, replay_master: pd.DataFrame) -> list[str]:
    if opt_master.empty:
        return []
    order = [fid for fid in formulation_order() if fid in set(opt_master["formulation_id"].astype(str))]
    if not order:
        return []

    replay_error = (
        replay_master.groupby("run_id", dropna=False)
        .agg(mean_abs_prediction_error=("abs_prediction_error", "mean"))
        .reset_index()
        if not replay_master.empty
        else pd.DataFrame(columns=["run_id", "mean_abs_prediction_error"])
    )
    plot_df = opt_master.merge(replay_error, on="run_id", how="left")
    save_plot_data("method_comparison_boxplots", plot_df)

    set_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4), constrained_layout=True)
    metrics = [
        ("objective_total", "Total Cost"),
        ("solve_time_sec", "Solve Time [s]"),
        ("mean_abs_prediction_error", "Mean Abs Replay Error"),
    ]
    labels = [_plot_label(fid, fid) for fid in order]
    for ax, (metric, title) in zip(axes, metrics):
        series = [
            pd.to_numeric(plot_df.loc[plot_df["formulation_id"].astype(str) == fid, metric], errors="coerce").dropna().to_numpy()
            for fid in order
        ]
        series = [vals if len(vals) else np.asarray([np.nan]) for vals in series]
        bp = ax.boxplot(series, patch_artist=True, labels=labels, showfliers=False)
        for patch, fid in zip(bp["boxes"], order):
            patch.set_facecolor(formulation_color(fid))
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        if metric == "solve_time_sec":
            ax.set_yscale("log")
    paths = save_figure(fig, "method_comparison_boxplots", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def build_outputs(
    config_path: str | Path | None = None,
    *,
    require_replay: bool = False,
    from_existing_results: bool = True,
    rebuild_tables: bool = True,
    rebuild_figures: bool = True,
    benchmark_config: str | Path | None = None,
    output_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    cfg = resolve_analysis_config(Path(config_path) if config_path is not None else None)
    ensure_output_dirs()

    outputs: dict[str, Any] = {"tables": {}, "figures": [], "warnings": []}

    if rebuild_tables:
        catalog_df = build_formulation_catalog()
        outputs["tables"]["formulation_catalog"] = _write_table_bundle("formulation_catalog", catalog_df)

        run_df = load_run_catalog()
        run_df.to_csv(MERGED_RESULTS_DIR / "formulation_run_catalog.csv", index=False)
        outputs["tables"]["formulation_kpis"] = _write_table_bundle("formulation_kpis", cost_summary_table(run_df))
    else:
        run_df = load_run_catalog()

    if rebuild_figures:
        cost_figures = _plot_cost_impact(cost_summary_table(run_df))
        outputs["figures"].extend(cost_figures)

    generator_df, ibr_df = build_dispatch_comparison_tables(run_df)
    if rebuild_tables:
        generator_df.to_csv(MERGED_RESULTS_DIR / "dispatch_generator_comparison.csv", index=False)
        ibr_df.to_csv(MERGED_RESULTS_DIR / "dispatch_ibr_comparison.csv", index=False)
        outputs["tables"]["dispatch_generator_comparison"] = _write_table_bundle("dispatch_generator_comparison", generator_df)
        outputs["tables"]["dispatch_ibr_comparison"] = _write_table_bundle("dispatch_ibr_comparison", ibr_df)
    if rebuild_figures:
        outputs["figures"].extend(_plot_dispatch_vis(generator_df, ibr_df, cfg))

    try:
        replay_detail_df, replay_summary_df = load_replay_outputs()
        replay_detail_df.to_csv(MERGED_RESULTS_DIR / "predicted_vs_replayed_metrics.csv", index=False)
        replay_summary_df.to_csv(MERGED_RESULTS_DIR / "replay_run_summary.csv", index=False)
        metric_summary_df = replay_metric_summary(replay_detail_df)
        if rebuild_tables:
            outputs["tables"]["replay_metric_summary"] = _write_table_bundle("replay_metric_summary", metric_summary_df)
        if rebuild_figures:
            outputs["figures"].extend(_plot_predicted_vs_replayed(replay_detail_df))

        by_run_df, by_formulation_df = summarize_replay_feasibility(replay_detail_df)
        constraint_by_run = merge_dynamic_and_line_security(run_df, replay_summary_df).merge(
            by_run_df,
            on=["run_id", "formulation_id", "formulation_name", "scenario_id", "scenario_name"],
            how="left",
        )
        constraint_by_formulation = (
            constraint_by_run.groupby(["formulation_id", "formulation_name"], dropna=False)
            .agg(
                n_runs=("run_id", "count"),
                line_security_safe_rate=("line_security_safe_all", "mean"),
                replay_dynamic_safe_rate=("replay_dynamic_safe_all", "mean"),
                false_safe_count=("false_safe_count", "sum"),
                false_unsafe_count=("false_unsafe_count", "sum"),
                true_unsafe_count=("true_unsafe_count", "sum"),
                max_line_violation_pct=("line_security_max_violation_pct", "max"),
                max_replay_violation_magnitude=("max_replay_violation_magnitude", "max"),
            )
            .reset_index()
        )
        constraint_by_run.to_csv(MERGED_RESULTS_DIR / "constraint_satisfaction_by_run.csv", index=False)
        constraint_by_formulation.to_csv(MERGED_RESULTS_DIR / "constraint_satisfaction_by_formulation.csv", index=False)
        if rebuild_tables:
            outputs["tables"]["constraint_satisfaction_by_run"] = _write_table_bundle("constraint_satisfaction_by_run", constraint_by_run)
            outputs["tables"]["constraint_satisfaction_by_formulation"] = _write_table_bundle(
                "constraint_satisfaction_by_formulation",
                constraint_by_formulation,
            )
        if rebuild_figures:
            outputs["figures"].extend(_plot_violation_breakdown(replay_summary_df))
    except OptimizationResultsNotReadyError as exc:
        if require_replay:
            raise
        outputs["warnings"].append(str(exc))

    benchmark_cfg_path = Path(benchmark_config) if benchmark_config is not None else BENCHMARK_CONFIG
    opt_master = load_optimization_runs_master(benchmark_cfg_path)
    replay_master = load_replay_runs_master(benchmark_cfg_path)
    manifest_df, subset_df = load_manifest_tables(benchmark_cfg_path)

    if rebuild_tables and not opt_master.empty:
        opt_master.to_csv(MERGED_RESULTS_DIR / "optimization_runs_master.csv", index=False)
        write_csv_and_parquet(
            opt_master,
            TABLES_DIR / "optimization_runs_master.csv",
            TABLES_DIR / "optimization_runs_master.parquet",
        )
        outputs["tables"]["optimization_runs_master"] = {
            "csv": str(TABLES_DIR / "optimization_runs_master.csv"),
            "parquet": str(TABLES_DIR / "optimization_runs_master.parquet"),
        }

    if rebuild_tables and not replay_master.empty:
        replay_master.to_csv(MERGED_RESULTS_DIR / "replay_runs_master.csv", index=False)
        write_csv_and_parquet(
            replay_master,
            TABLES_DIR / "replay_runs_master.csv",
            TABLES_DIR / "replay_runs_master.parquet",
        )
        outputs["tables"]["replay_runs_master"] = {
            "csv": str(TABLES_DIR / "replay_runs_master.csv"),
            "parquet": str(TABLES_DIR / "replay_runs_master.parquet"),
        }

    replay_metric_breakdown_df = (
        replay_master.groupby(["formulation", "formulation_name", "metric_name", "metric_category"], dropna=False)
        .agg(
            n_rows=("metric_name", "count"),
            replay_violation_count=("violated_in_replay", "sum"),
            replay_violation_rate=("violated_in_replay", "mean"),
            false_safe_count=("false_safe_flag", "sum"),
            false_safe_rate=("false_safe_flag", "mean"),
            scheduled_headroom_exceedance_count=("scheduled_headroom_violation_flag", "sum"),
            physical_limit_exceedance_count=("physical_limit_violation_flag", "sum"),
            max_replay_violation_magnitude=("violation_magnitude", "max"),
            max_headroom_exceedance_magnitude=("scheduled_headroom_violation_magnitude", "max"),
            max_physical_limit_exceedance_magnitude=("physical_limit_violation_magnitude", "max"),
        )
        .reset_index()
        .rename(columns={"formulation": "formulation_id"})
        if not replay_master.empty
        else pd.DataFrame()
    )
    replay_method_breakdown_df = (
        replay_master.groupby(["formulation", "formulation_name"], dropna=False)
        .agg(
            n_rows=("metric_name", "count"),
            rocof_violation_count=("metric_category", lambda s: int(np.sum((pd.Series(s).astype(str) == "rocof") & (replay_master.loc[s.index, "violated_in_replay"] >= 0.5)))),
            frequency_violation_count=("metric_category", lambda s: int(np.sum((pd.Series(s).astype(str) == "frequency_deviation") & (replay_master.loc[s.index, "violated_in_replay"] >= 0.5)))),
            ibr_power_violation_count=("metric_category", lambda s: int(np.sum((pd.Series(s).astype(str) == "ibr_power") & (replay_master.loc[s.index, "violated_in_replay"] >= 0.5)))),
            replay_violation_count=("violated_in_replay", "sum"),
            replay_violation_rate=("violated_in_replay", "mean"),
            false_safe_count=("false_safe_flag", "sum"),
            false_safe_rate=("false_safe_flag", "mean"),
            scheduled_headroom_exceedance_count=("scheduled_headroom_violation_flag", "sum"),
            physical_limit_exceedance_count=("physical_limit_violation_flag", "sum"),
            max_replay_violation_magnitude=("violation_magnitude", "max"),
        )
        .reset_index()
        .rename(columns={"formulation": "formulation_id"})
        if not replay_master.empty
        else pd.DataFrame()
    )

    benchmark_tables = {
        "method_cost_breakdown": method_cost_breakdown(opt_master),
        "reserve_summary_by_method": reserve_summary_by_method(opt_master),
        "solver_complexity_summary": solver_complexity_summary(opt_master),
        "replay_violation_breakdown_by_metric": replay_metric_breakdown_df,
        "replay_violation_breakdown_by_method": replay_method_breakdown_df,
        "line_security_summary": line_security_summary(opt_master),
        "vis_allocation_summary": vis_allocation_summary(opt_master),
        "scenario_coverage_summary": scenario_coverage_summary(opt_master, replay_master, manifest_df),
    }
    if rebuild_tables:
        for stem, df in benchmark_tables.items():
            if df is not None and not df.empty:
                outputs["tables"][stem] = _write_table_bundle(stem, df)

    if rebuild_figures:
        outputs["figures"].extend(_plot_cost_breakdown(benchmark_tables["method_cost_breakdown"]))
        outputs["figures"].extend(_plot_reserve_headroom_by_ibr(opt_master, replay_master))
        outputs["figures"].extend(_plot_replay_violation_counts_by_metric(replay_master))
        outputs["figures"].extend(_plot_cost_vs_replay_safety_scatter(opt_master, replay_master))
        outputs["figures"].extend(_plot_solve_time_vs_binary_count(opt_master))
        outputs["figures"].extend(_plot_retained_feasibility_boundary())
        outputs["figures"].extend(_plot_zone_targeted_md_schedule())
        outputs["figures"].extend(_plot_zone_targeted_md_unit_allocation())
        outputs["figures"].extend(_plot_mtlsh_topk_screening())
        outputs["figures"].extend(_plot_replay_frequency_by_formulation(opt_master, replay_master))
        if not replay_master.empty:
            plot_df = replay_master.rename(columns={"formulation": "formulation_id"})
            outputs["figures"].extend(_plot_predicted_vs_replayed(plot_df))
            outputs["figures"].extend(_plot_predicted_vs_replayed_rocof(replay_master))
        outputs["figures"].extend(_plot_scenario_coverage_heatmaps(manifest_df))
        outputs["figures"].extend(_plot_method_comparison_boxplots(opt_master, replay_master))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-ready optimization tables and figures.")
    parser.add_argument(
        "--config",
        default="configs/scheduling/analysis/results_pack.yaml",
        help="Optional analysis config YAML.",
    )
    parser.add_argument(
        "--require-replay",
        action="store_true",
        help="Fail if replay-validation artifacts are not available yet.",
    )
    parser.add_argument("--from-existing-results", action="store_true", help="Rebuild from stored artifacts without rerunning optimization.")
    parser.add_argument("--rebuild-tables", action="store_true", help="Rebuild tables from stored artifacts.")
    parser.add_argument("--rebuild-figures", action="store_true", help="Rebuild figures from stored artifacts.")
    parser.add_argument(
        "--benchmark-config",
        default=str(BENCHMARK_CONFIG),
        help="Benchmark config YAML used for manifest-aware rebuilds.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional alternate output root. The current implementation still writes to the thesis optimization output tree.",
    )
    parser.add_argument("--force", action="store_true", help="Reserved flag for future rebuild policies.")
    args = parser.parse_args()

    try:
        rebuild_tables = bool(args.rebuild_tables) or not bool(args.rebuild_figures)
        rebuild_figures = bool(args.rebuild_figures) or not bool(args.rebuild_tables)
        outputs = build_outputs(
            args.config,
            require_replay=bool(args.require_replay),
            from_existing_results=bool(args.from_existing_results or True),
            rebuild_tables=rebuild_tables,
            rebuild_figures=rebuild_figures,
            benchmark_config=args.benchmark_config,
            output_root=args.output_root or None,
            force=bool(args.force),
        )
    except OptimizationResultsNotReadyError as exc:
        print(f"[build_outputs] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"[build_outputs] tables_dir={TABLES_DIR}")
    print(f"[build_outputs] figures_dir={FIGURES_DIR}")
    for warning in outputs.get("warnings", []):
        print(f"[build_outputs] warning={warning}")


if __name__ == "__main__":
    main()
