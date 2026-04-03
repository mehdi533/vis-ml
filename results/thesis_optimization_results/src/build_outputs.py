from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    from .analysis_utils import (
        baseline_formulation_id,
        build_dispatch_comparison_tables,
        build_formulation_catalog,
        cost_summary_table,
        formulation_order,
        load_replay_outputs,
        load_run_catalog,
        merge_dynamic_and_line_security,
        replay_metric_summary,
        resolve_analysis_config,
        retained_formulation_id,
        write_latex_table,
        write_markdown_table,
    )
    from .config_analysis import FIGURES_DIR, MERGED_RESULTS_DIR, TABLES_DIR, ensure_output_dirs, OptimizationResultsNotReadyError
    from .plot_utils import formulation_color, save_figure, set_thesis_style
    from .validation_utils import summarize_replay_feasibility
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
        load_replay_outputs,
        load_run_catalog,
        merge_dynamic_and_line_security,
        replay_metric_summary,
        resolve_analysis_config,
        retained_formulation_id,
        write_latex_table,
        write_markdown_table,
    )
    from config_analysis import FIGURES_DIR, MERGED_RESULTS_DIR, TABLES_DIR, ensure_output_dirs, OptimizationResultsNotReadyError  # type: ignore
    from plot_utils import formulation_color, save_figure, set_thesis_style  # type: ignore
    from validation_utils import summarize_replay_feasibility  # type: ignore


def _write_table_bundle(stem: str, df: pd.DataFrame) -> dict[str, str]:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    tex_path = TABLES_DIR / f"{stem}.tex"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_markdown_table(md_path, df)
    write_latex_table(tex_path, df)
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path)}


def _plot_cost_impact(cost_df: pd.DataFrame) -> list[str]:
    feasible = cost_df.loc[cost_df["is_feasible"] >= 0.5].copy()
    if feasible.empty:
        return []

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    scenario_count = feasible["scenario_id"].nunique()
    order = formulation_order()
    if scenario_count <= 1:
        plot_df = feasible.copy()
        plot_df["formulation_id"] = pd.Categorical(plot_df["formulation_id"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("formulation_id")
        x = np.arange(plot_df.shape[0], dtype=float)
        y = pd.to_numeric(plot_df["cost_increase_pct_vs_ed"], errors="coerce").fillna(0.0).to_numpy()
        labels = plot_df["formulation_name"].astype(str).tolist()
        colors = [formulation_color(fid) for fid in plot_df["formulation_id"].astype(str)]
        ax.bar(x, y, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Cost Increase vs ED [%]")
        ax.set_title("Cost Impact of Security Constraints")
    else:
        grouped = (
            feasible.groupby(["formulation_id", "formulation_name"], dropna=False)
            .agg(
                mean_cost_increase_pct=("cost_increase_pct_vs_ed", "mean"),
                min_cost_increase_pct=("cost_increase_pct_vs_ed", "min"),
                max_cost_increase_pct=("cost_increase_pct_vs_ed", "max"),
            )
            .reset_index()
        )
        grouped["formulation_id"] = pd.Categorical(grouped["formulation_id"], categories=order, ordered=True)
        grouped = grouped.sort_values("formulation_id")
        x = np.arange(grouped.shape[0], dtype=float)
        y = pd.to_numeric(grouped["mean_cost_increase_pct"], errors="coerce").to_numpy()
        yerr = np.vstack(
            [
                y - pd.to_numeric(grouped["min_cost_increase_pct"], errors="coerce").to_numpy(),
                pd.to_numeric(grouped["max_cost_increase_pct"], errors="coerce").to_numpy() - y,
            ]
        )
        colors = [formulation_color(fid) for fid in grouped["formulation_id"].astype(str)]
        ax.bar(x, y, color=colors, alpha=0.9, yerr=yerr, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["formulation_name"].astype(str).tolist(), rotation=20, ha="right")
        ax.set_ylabel("Mean Cost Increase vs ED [%]")
        ax.set_title("Cost Impact Across Scenarios")

    ax.axhline(0.0, color="black", linewidth=1.0)
    paths = save_figure(fig, "cost_impact_by_formulation", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_dispatch_vis(generator_df: pd.DataFrame, ibr_df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    if generator_df.empty or ibr_df.empty:
        return []

    order = formulation_order(cfg)
    baseline_id = baseline_formulation_id(cfg)
    retained_id = retained_formulation_id(cfg)
    default_formulations = [fid for fid in [baseline_id, "ed_surrogate", retained_id] if fid in set(generator_df["formulation_id"])]
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
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    ax_dispatch, ax_m, ax_d, ax_headroom = axes.ravel()

    def _grouped_bar(ax, df, *, x_col: str, y_col: str, title: str, ylabel: str) -> None:
        ids = [fid for fid in selected_formulations if fid in set(df["formulation_id"].astype(str))]
        x_values = sorted(df[x_col].dropna().astype(int).unique().tolist())
        if not ids or not x_values:
            ax.set_visible(False)
            return
        x = np.arange(len(x_values), dtype=float)
        width = 0.8 / max(len(ids), 1)
        for idx, formulation_id in enumerate(ids):
            subset = df.loc[df["formulation_id"].astype(str) == formulation_id].copy()
            subset = subset.set_index(x_col)
            values = pd.to_numeric(subset.reindex(x_values)[y_col], errors="coerce").fillna(0.0).to_numpy()
            offset = (idx - (len(ids) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=formulation_id,
                color=formulation_color(formulation_id),
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    _grouped_bar(
        ax_dispatch,
        gen_plot,
        x_col="index",
        y_col="pg_delta_vs_baseline",
        title="Dispatch Shift vs ED",
        ylabel="Dispatch Delta [p.u.]",
    )
    _grouped_bar(ax_m, ibr_plot, x_col="index", y_col="M_opt", title="VIS Inertia Allocation", ylabel="M [-]")
    _grouped_bar(ax_d, ibr_plot, x_col="index", y_col="D_opt", title="VIS Damping Allocation", ylabel="D [-]")
    _grouped_bar(
        ax_headroom,
        ibr_plot,
        x_col="index",
        y_col="headroom_up_vs_baseline",
        title="IBR Upward Headroom Shift",
        ylabel="Headroom Delta [p.u.]",
    )

    handles, labels = ax_dispatch.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(f"Dispatch, Headroom, and VIS Allocation ({scenario_id})", y=1.01)
    paths = save_figure(fig, f"dispatch_vis_comparison_{scenario_id}", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_predicted_vs_replayed(detail_df: pd.DataFrame) -> list[str]:
    if detail_df.empty:
        return []

    metrics = detail_df["metric_name"].dropna().astype(str).unique().tolist()
    if not metrics:
        return []

    set_thesis_style()
    ncols = min(3, len(metrics))
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    order = formulation_order()
    for ax, metric in zip(axes_arr.ravel(), metrics):
        subset = detail_df.loc[detail_df["metric_name"].astype(str) == metric].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        for formulation_id in [fid for fid in order if fid in set(subset["formulation_id"].astype(str))]:
            frame = subset.loc[subset["formulation_id"].astype(str) == formulation_id]
            ax.scatter(
                pd.to_numeric(frame["predicted_value"], errors="coerce"),
                pd.to_numeric(frame["replayed_value"], errors="coerce"),
                s=24,
                alpha=0.75,
                color=formulation_color(formulation_id),
                label=formulation_id,
            )
        vals = pd.concat(
            [
                pd.to_numeric(subset["predicted_value"], errors="coerce"),
                pd.to_numeric(subset["replayed_value"], errors="coerce"),
            ],
            axis=0,
        ).dropna()
        if not vals.empty:
            lo = float(vals.min())
            hi = float(vals.max())
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
        ax.set_title(metric)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Replayed")
    for ax in axes_arr.ravel()[len(metrics) :]:
        ax.set_visible(False)
    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle("Predicted vs Replayed Dynamic-Security Metrics", y=1.01)
    paths = save_figure(fig, "predicted_vs_replayed_metrics", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_violation_breakdown(by_formulation_df: pd.DataFrame) -> list[str]:
    if by_formulation_df.empty:
        return []

    set_thesis_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    plot_df = by_formulation_df.copy()
    order = formulation_order()
    plot_df["formulation_id"] = pd.Categorical(plot_df["formulation_id"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("formulation_id")
    x = np.arange(plot_df.shape[0], dtype=float)
    false_safe = pd.to_numeric(plot_df["false_safe_count"], errors="coerce").fillna(0.0).to_numpy()
    false_unsafe = pd.to_numeric(plot_df["false_unsafe_count"], errors="coerce").fillna(0.0).to_numpy()
    true_unsafe = pd.to_numeric(plot_df["true_unsafe_count"], errors="coerce").fillna(0.0).to_numpy()
    ax.bar(x, false_safe, color="#c46646", label="False safe")
    ax.bar(x, false_unsafe, bottom=false_safe, color="#d89c3d", label="False unsafe")
    ax.bar(x, true_unsafe, bottom=false_safe + false_unsafe, color="#7a8da6", label="True unsafe")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["formulation_name"].astype(str).tolist(), rotation=20, ha="right")
    ax.set_ylabel("Metric Count [-]")
    ax.set_title("Replay Constraint-Satisfaction Outcomes")
    ax.legend(frameon=False)
    paths = save_figure(fig, "constraint_satisfaction_breakdown", FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def build_outputs(config_path: str | Path | None = None, *, require_replay: bool = False) -> dict[str, Any]:
    cfg = resolve_analysis_config(Path(config_path) if config_path is not None else None)
    ensure_output_dirs()

    outputs: dict[str, Any] = {"tables": {}, "figures": [], "warnings": []}

    catalog_df = build_formulation_catalog()
    outputs["tables"]["formulation_catalog"] = _write_table_bundle("formulation_catalog", catalog_df)

    run_df = load_run_catalog()
    run_df.to_csv(MERGED_RESULTS_DIR / "formulation_run_catalog.csv", index=False)
    outputs["tables"]["formulation_kpis"] = _write_table_bundle("formulation_kpis", cost_summary_table(run_df))

    cost_figures = _plot_cost_impact(cost_summary_table(run_df))
    outputs["figures"].extend(cost_figures)

    generator_df, ibr_df = build_dispatch_comparison_tables(run_df)
    generator_df.to_csv(MERGED_RESULTS_DIR / "dispatch_generator_comparison.csv", index=False)
    ibr_df.to_csv(MERGED_RESULTS_DIR / "dispatch_ibr_comparison.csv", index=False)
    outputs["tables"]["dispatch_generator_comparison"] = _write_table_bundle("dispatch_generator_comparison", generator_df)
    outputs["tables"]["dispatch_ibr_comparison"] = _write_table_bundle("dispatch_ibr_comparison", ibr_df)
    outputs["figures"].extend(_plot_dispatch_vis(generator_df, ibr_df, cfg))

    try:
        replay_detail_df, replay_summary_df = load_replay_outputs()
        replay_detail_df.to_csv(MERGED_RESULTS_DIR / "predicted_vs_replayed_metrics.csv", index=False)
        replay_summary_df.to_csv(MERGED_RESULTS_DIR / "replay_run_summary.csv", index=False)
        metric_summary_df = replay_metric_summary(replay_detail_df)
        outputs["tables"]["replay_metric_summary"] = _write_table_bundle("replay_metric_summary", metric_summary_df)
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
        outputs["tables"]["constraint_satisfaction_by_run"] = _write_table_bundle("constraint_satisfaction_by_run", constraint_by_run)
        outputs["tables"]["constraint_satisfaction_by_formulation"] = _write_table_bundle(
            "constraint_satisfaction_by_formulation",
            constraint_by_formulation,
        )
        outputs["figures"].extend(_plot_violation_breakdown(by_formulation_df))
    except OptimizationResultsNotReadyError as exc:
        if require_replay:
            raise
        outputs["warnings"].append(str(exc))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-ready optimization tables and figures.")
    parser.add_argument(
        "--config",
        default="results/thesis_optimization_results/configs/analysis/results_pack.yaml",
        help="Optional analysis config YAML.",
    )
    parser.add_argument(
        "--require-replay",
        action="store_true",
        help="Fail if replay-validation artifacts are not available yet.",
    )
    args = parser.parse_args()

    try:
        outputs = build_outputs(args.config, require_replay=bool(args.require_replay))
    except OptimizationResultsNotReadyError as exc:
        print(f"[build_outputs] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"[build_outputs] tables_dir={TABLES_DIR}")
    print(f"[build_outputs] figures_dir={FIGURES_DIR}")
    for warning in outputs.get("warnings", []):
        print(f"[build_outputs] warning={warning}")


if __name__ == "__main__":
    main()
