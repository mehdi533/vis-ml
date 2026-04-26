from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    from .plot_utils import formulation_color as _base_formulation_color
    from .plot_utils import set_thesis_style
except ImportError:
    from plot_utils import formulation_color as _base_formulation_color  # type: ignore
    from plot_utils import set_thesis_style  # type: ignore


FORMULATION_ORDER = [
    "ed",
    "ed_line",
    "ed_line_n1",
    "ed_surrogate",
    "ed_line_n1_surrogate",
    "ed_line_n1_surrogate_redispatch",
]

FORMULATION_LABELS = {
    "ed": "A: ED",
    "ed_line": "B: ED + Line",
    "ed_line_n1": "C: ED + Line + N-1",
    "ed_surrogate": "D: ED + Surrogate",
    "ed_line_n1_surrogate": "E: Full preventive",
    "ed_line_n1_surrogate_redispatch": "Full + redispatch",
    "retained_vis": "Retained VIS formulation",
}

METRIC_LABELS = {
    "rocof_COI": r"$\mathrm{RoCoF}_{\mathrm{COI}}$",
    "dev_COI": r"$\Delta f_{\mathrm{COI}}$",
    "Delta_P_IBR_1": r"$\Delta P_{\mathrm{IBR},1}$",
    "Delta_P_IBR_2": r"$\Delta P_{\mathrm{IBR},2}$",
    "Delta_P_IBR_3": r"$\Delta P_{\mathrm{IBR},3}$",
    "Delta_P_IBR_4": r"$\Delta P_{\mathrm{IBR},4}$",
}

METRIC_LIMITS = {
    "rocof_COI": 1.0,
    "dev_COI": 0.8,
}

METRIC_SCALES = {
    "rocof_COI": 1.0,
    "dev_COI": 1.0,
    "Delta_P_IBR_1": 100.0,
    "Delta_P_IBR_2": 100.0,
    "Delta_P_IBR_3": 100.0,
    "Delta_P_IBR_4": 100.0,
}

TRACE_FAMILY_CATALOG: dict[str, dict[str, Any]] = {
    "tightfreq_custom_065": {
        "folder": "results/thesis_optimization_results/local_validation/replay_trace_tightfreq_custom_065",
        "rerun_config": "configs/scheduling/replay/focused_trace_tightfreq_custom_065.yaml",
        "default_stem": "replay_trace_tightfreq_custom_065",
        "chapter": "5.5",
        "kind": "combined",
    },
    "zone_local_frequency": {
        "folder": "results/thesis_optimization_results/local_validation/replay_trace_zone_local_frequency",
        "rerun_config": "configs/scheduling/replay/focused_trace_zone_local_frequency.yaml",
        "default_stem": "replay_trace_zone_local_frequency_panel",
        "chapter": "5.5",
        "kind": "frequency_panel",
    },
    "zone_local_ibr": {
        "folder": "results/thesis_optimization_results/local_validation/replay_trace_zone_local_frequency",
        "rerun_config": "configs/scheduling/replay/focused_trace_zone_local_frequency.yaml",
        "default_stem": "replay_trace_zone_local_ibr_panel",
        "chapter": "5.5",
        "kind": "ibr_panel",
    },
    "line_robustness_frequency": {
        "folder": "results/thesis_optimization_results/local_validation/replay_trace_line_robustness",
        "rerun_config": "configs/scheduling/replay/focused_trace_line_robustness.yaml",
        "default_stem": "replay_trace_topk_frequency_panel",
        "chapter": "5.5",
        "kind": "frequency_panel",
    },
    "line_robustness_ibr": {
        "folder": "results/thesis_optimization_results/local_validation/replay_trace_line_robustness",
        "rerun_config": "configs/scheduling/replay/focused_trace_line_robustness.yaml",
        "default_stem": "replay_trace_topk_ibr_panel",
        "chapter": "5.5",
        "kind": "ibr_panel",
    },
    "formulation_contrast": {
        "folder": "results/thesis_optimization_results/local_validation/replay_trace_formulation_contrast",
        "rerun_config": "configs/scheduling/replay/focused_trace_formulation_contrast.yaml",
        "default_stem": "replay_trace_formulation_contrast",
        "chapter": "5.5",
        "kind": "combined",
    },
}


def find_repo_root(start: Path | None = None) -> Path:
    base = Path(start).resolve() if start is not None else Path.cwd().resolve()
    for candidate in [base, *base.parents]:
        if (candidate / "results" / "thesis_optimization_results").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root from current working directory.")


def get_paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = find_repo_root(repo_root)
    opt_root = root / "results" / "thesis_optimization_results"
    return {
        "root": root,
        "opt_root": opt_root,
        "outputs_root": opt_root / "outputs",
        "plot_data_dir": opt_root / "outputs" / "plot_data",
        "tables_dir": opt_root / "outputs" / "tables",
        "figures_dir": opt_root / "outputs" / "figures",
        "results_dir": opt_root / "results",
        "local_validation_dir": opt_root / "local_validation",
        "src_dir": opt_root / "src",
        "thesis_root": root.parent / "Thesis",
    }


def chapter_figure_dir(chapter: str, repo_root: Path | None = None) -> Path:
    paths = get_paths(repo_root)
    return paths["thesis_root"] / "Figures" / "05_Results" / chapter


def configure_notebook_style(style_cfg: dict[str, Any] | None = None) -> None:
    cfg = dict(style_cfg or {})
    set_thesis_style()
    font_scale = float(cfg.get("font_scale", 1.0))
    dpi = float(cfg.get("dpi", plt.rcParams.get("figure.dpi", 140)))
    grid = bool(cfg.get("grid", True))
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["axes.grid"] = grid
    for key in [
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
    ]:
        try:
            plt.rcParams[key] = float(plt.rcParams[key]) * font_scale
        except Exception:
            pass


def default_export_cfg() -> dict[str, Any]:
    return {
        "write_results": True,
        "write_thesis": True,
        "export_png": True,
        "export_pdf": True,
        "overwrite": True,
    }


def default_style_cfg() -> dict[str, Any]:
    return {
        "figsize": (12.0, 4.8),
        "dpi": 160,
        "font_scale": 1.0,
        "legend_loc": "upper center",
        "legend_ncol": 3,
        "legend_bbox": None,
        "grid": True,
        "palette": {},
        "title_mode": "section",
    }


def default_layout_cfg() -> dict[str, Any]:
    return {
        "shared_axes": False,
        "label_rotation": 0,
        "margins": {"top": 0.9, "bottom": 0.12, "left": 0.08, "right": 0.98},
        "ylims": {},
        "annotate": True,
    }


def _resolve_rel(path_like: str | Path, repo_root: Path | None = None) -> Path:
    root = find_repo_root(repo_root)
    path = Path(path_like)
    return path if path.is_absolute() else (root / path).resolve()


def load_table(name: str, repo_root: Path | None = None) -> pd.DataFrame:
    path = get_paths(repo_root)["tables_dir"] / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path)


def load_plot_data(name: str, repo_root: Path | None = None) -> pd.DataFrame:
    path = get_paths(repo_root)["plot_data_dir"] / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing plot-data CSV: {path}")
    return pd.read_csv(path)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _close_float_match(series: pd.Series, target: float, tol: float = 1e-6) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return (values - float(target)).abs() <= tol


def _ordered_formulations(formulations: list[str] | None) -> list[str]:
    selected = [str(value) for value in _as_list(formulations)]
    if not selected:
        return list(FORMULATION_ORDER)
    front = [value for value in FORMULATION_ORDER if value in selected]
    tail = [value for value in selected if value not in front]
    return front + tail


def _step_scale_tag(step_scale: float) -> str:
    return f"ss{int(round(float(step_scale) * 100)):03d}"


def _zone_raw_results_root(step_scale: float, repo_root: Path | None = None) -> Path:
    paths = get_paths(repo_root)
    return paths["results_dir"] / f"zone_mismatch_vis_sensitivity_{_step_scale_tag(step_scale)}_formulations"


def _palette_color(key: str, style_cfg: dict[str, Any] | None = None) -> str:
    palette = dict((style_cfg or {}).get("palette", {}) or {})
    return str(palette.get(key, _base_formulation_color(key)))


def _legend_kwargs(style_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    legend_kwargs = {
        "loc": str(style["legend_loc"]),
        "ncol": int(style["legend_ncol"]),
    }
    bbox = style.get("legend_bbox")
    if bbox is not None:
        legend_kwargs["bbox_to_anchor"] = tuple(bbox)
    return legend_kwargs


def export_figure(
    fig: plt.Figure,
    *,
    stem: str,
    chapter: str,
    export_cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    cfg = default_export_cfg()
    cfg.update(dict(export_cfg or {}))
    paths = get_paths(repo_root)
    targets: list[tuple[str, Path]] = []
    if cfg["write_results"]:
        targets.append(("results", paths["figures_dir"] / stem))
    if cfg["write_thesis"]:
        targets.append(("thesis", chapter_figure_dir(chapter, repo_root) / stem))
    written: list[dict[str, str]] = []
    for kind, base in targets:
        base.parent.mkdir(parents=True, exist_ok=True)
        if cfg["export_png"]:
            png_path = base.with_suffix(".png")
            try:
                if cfg["overwrite"] or not png_path.exists():
                    fig.savefig(png_path, bbox_inches="tight")
                written.append({"target": kind, "path": str(png_path), "status": "written", "error": ""})
            except OSError as exc:
                written.append({"target": kind, "path": str(png_path), "status": "skipped", "error": str(exc)})
        if cfg["export_pdf"]:
            pdf_path = base.with_suffix(".pdf")
            try:
                if cfg["overwrite"] or not pdf_path.exists():
                    fig.savefig(pdf_path, bbox_inches="tight")
                written.append({"target": kind, "path": str(pdf_path), "status": "written", "error": ""})
            except OSError as exc:
                written.append({"target": kind, "path": str(pdf_path), "status": "skipped", "error": str(exc)})
    return pd.DataFrame(written)


def format_sentences(sentences: list[str]) -> str:
    return "\n".join(f"- {sentence}" for sentence in sentences)


def prepare_cost_impact(
    formulation_kpis: pd.DataFrame,
    cost_breakdown: pd.DataFrame,
    scenario_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = dict(scenario_cfg or {})
    scenario_id = cfg.get("scenario_id")
    formulations = [str(value) for value in _as_list(cfg.get("formulations"))]
    selected = formulation_kpis.copy()
    if scenario_id:
        selected = selected.loc[selected["scenario_id"].astype(str) == str(scenario_id)].copy()
    if formulations:
        selected = selected.loc[selected["formulation_id"].astype(str).isin(formulations)].copy()
    if selected.empty:
        selected = cost_breakdown.copy()
        selected["scenario_id"] = str(scenario_id or "aggregate_mean")
        selected["total_cost_raw"] = pd.to_numeric(selected["total_cost_mean"], errors="coerce")
        selected["reserve_cost_component"] = pd.to_numeric(selected["reserve_cost_mean"], errors="coerce")
        selected["reserve_precont_cost"] = pd.to_numeric(selected.get("reserve_up_cost_mean"), errors="coerce")
        selected["reserve_postcont_cost"] = pd.to_numeric(selected.get("reserve_postcont_cost_mean"), errors="coerce")
        selected["solve_time_sec"] = np.nan
        if formulations:
            selected = selected.loc[selected["formulation_id"].astype(str).isin(formulations)].copy()
    else:
        selected["total_cost_raw"] = pd.to_numeric(selected["total_cost"], errors="coerce")
        selected["reserve_cost_component"] = pd.to_numeric(selected["reserve_cost_component"], errors="coerce")
        selected["reserve_precont_cost"] = pd.to_numeric(selected.get("reserve_up_cost_component"), errors="coerce")
        selected["reserve_postcont_cost"] = pd.to_numeric(selected.get("reserve_postcont_cost_component"), errors="coerce")
    selected["dispatch_only_cost"] = selected["total_cost_raw"] - selected["reserve_cost_component"].fillna(0.0)
    selected["reserve_only_cost"] = selected["reserve_precont_cost"].where(
        selected["reserve_precont_cost"].notna(),
        pd.to_numeric(selected["reserve_cost_component"], errors="coerce"),
    )
    selected["total_cost"] = selected["dispatch_only_cost"] + selected["reserve_only_cost"].fillna(0.0)
    selected["formulation_id"] = selected["formulation_id"].astype(str)
    selected["formulation_label"] = selected["formulation_id"].map(FORMULATION_LABELS).fillna(selected["formulation_name"])
    ordered = _ordered_formulations(formulations)
    selected["formulation_id"] = pd.Categorical(selected["formulation_id"], categories=ordered, ordered=True)
    selected = selected.sort_values("formulation_id").reset_index(drop=True)
    return selected


def plot_cost_impact(
    cost_df: pd.DataFrame,
    *,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = dict(layout_cfg or {})
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    figsize = tuple(style.get("figsize", (11.0, 4.5)))
    component_map = {
        "dispatch_only_cost": "Dispatch-only cost",
        "reserve_only_cost": "Pre-contingency reserve cost",
    }
    components = [str(value) for value in _as_list(layout.get("components"))] or ["dispatch_only_cost", "reserve_only_cost"]
    x = np.arange(cost_df.shape[0], dtype=float)
    width = 0.36 if len(components) <= 2 else 0.24
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for idx, component in enumerate(components):
        vals = pd.to_numeric(cost_df[component], errors="coerce") / 1000.0
        ax.bar(
            x + (idx - (len(components) - 1) / 2.0) * width,
            vals,
            width=width,
            label=component_map.get(component, component),
            color=["#c58b4f", "#6e9f8f", "#7f8db3"][idx % 3],
            alpha=0.9,
        )
    if bool(layout.get("show_total_line", True)):
        total_vals = pd.to_numeric(cost_df["total_cost"], errors="coerce") / 1000.0
        ax.plot(x, total_vals, color="#2b2826", marker="o", linewidth=2.0, label="Total objective")
    ax.set_xticks(x)
    ax.set_xticklabels(cost_df["formulation_label"].astype(str), rotation=float(layout.get("label_rotation", 0)))
    ax.set_ylabel("Cost [$10^3$]")
    title = layout.get("title") or "Formulation Cost Comparison"
    ax.set_title(str(title))
    ax.legend(**_legend_kwargs(style))
    return fig


def summarize_cost_impact(cost_df: pd.DataFrame, baseline_id: str = "ed") -> tuple[pd.DataFrame, list[str]]:
    frame = cost_df.copy()
    baseline = frame.loc[frame["formulation_id"].astype(str) == str(baseline_id)]
    if baseline.empty:
        return pd.DataFrame(), ["Baseline formulation is not present in the selected cost table."]
    base = baseline.iloc[0]
    out_rows: list[dict[str, Any]] = []
    sentences: list[str] = []
    for _, row in frame.iterrows():
        total = float(pd.to_numeric(row.get("total_cost"), errors="coerce"))
        dispatch = float(pd.to_numeric(row.get("dispatch_only_cost"), errors="coerce"))
        reserve = float(pd.to_numeric(row.get("reserve_only_cost"), errors="coerce"))
        total_delta = total - float(base["total_cost"])
        dispatch_delta = dispatch - float(base["dispatch_only_cost"])
        reserve_delta = reserve - float(base["reserve_only_cost"])
        solve_ratio = np.nan
        if pd.notna(row.get("solve_time_sec")) and pd.notna(base.get("solve_time_sec")) and float(base["solve_time_sec"]) > 0:
            solve_ratio = float(row["solve_time_sec"]) / float(base["solve_time_sec"])
        out_rows.append(
            {
                "formulation_id": str(row["formulation_id"]),
                "formulation_label": str(row["formulation_label"]),
                "total_cost_delta": total_delta,
                "dispatch_only_delta": dispatch_delta,
                "reserve_cost_delta": reserve_delta,
                "solve_time_ratio_vs_ed": solve_ratio,
            }
        )
        if str(row["formulation_id"]) != str(baseline_id):
            sentences.append(
                f"{row['formulation_label']} changes the objective by {total_delta / 1000.0:.3f}k, "
                f"with dispatch-only delta {dispatch_delta / 1000.0:.3f}k and reserve delta {reserve_delta / 1000.0:.3f}k versus ED."
            )
    return pd.DataFrame(out_rows), sentences


def prepare_dispatch_vis(
    dispatch_gen: pd.DataFrame,
    dispatch_ibr: pd.DataFrame,
    scenario_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = dict(scenario_cfg or {})
    scenario_id = str(cfg.get("scenario_id", "b0p600_s1p200_t1p000"))
    formulations = _ordered_formulations(cfg.get("formulations") or ["ed", "ed_line", "ed_line_n1", "ed_surrogate", "ed_line_n1_surrogate"])
    gen = dispatch_gen.loc[
        (dispatch_gen["scenario_id"].astype(str) == scenario_id)
        & (dispatch_gen["formulation_id"].astype(str).isin(formulations))
    ].copy()
    ibr = dispatch_ibr.loc[
        (dispatch_ibr["scenario_id"].astype(str) == scenario_id)
        & (dispatch_ibr["formulation_id"].astype(str).isin(formulations))
    ].copy()
    gen["formulation_id"] = pd.Categorical(gen["formulation_id"], categories=formulations, ordered=True)
    ibr["formulation_id"] = pd.Categorical(ibr["formulation_id"], categories=formulations, ordered=True)
    gen = gen.sort_values(["formulation_id", "index"]).reset_index(drop=True)
    ibr = ibr.sort_values(["formulation_id", "index"]).reset_index(drop=True)
    return gen, ibr


def _grouped_bar(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, label_col: str, style_cfg: dict[str, Any]) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    categories = sorted(df[x_col].dropna().astype(int).unique().tolist())
    forms = [str(value) for value in df[label_col].astype(str).unique().tolist()]
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(len(forms), 1)
    for idx, formulation_id in enumerate(forms):
        subset = df.loc[df[label_col].astype(str) == formulation_id].copy()
        subset = subset.set_index(x_col)
        vals = [pd.to_numeric(subset[y_col], errors="coerce").get(cat, np.nan) for cat in categories]
        ax.bar(
            x + (idx - (len(forms) - 1) / 2.0) * width,
            vals,
            width=width,
            label=FORMULATION_LABELS.get(formulation_id, formulation_id),
            color=_palette_color(formulation_id, style_cfg),
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(cat + 1) for cat in categories])


def plot_dispatch_vis(
    gen_df: pd.DataFrame,
    ibr_df: pd.DataFrame,
    *,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    figsize = tuple(style.get("figsize", (14.0, 8.0)))
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_dispatch, ax_headroom, ax_m, ax_d = axes.ravel()
    _grouped_bar(ax_dispatch, gen_df, "index", "pg_delta", "formulation_id", style)
    _grouped_bar(ax_headroom, ibr_df, "index", "headroom_up", "formulation_id", style)
    _grouped_bar(ax_m, ibr_df, "index", "M_opt", "formulation_id", style)
    _grouped_bar(ax_d, ibr_df, "index", "D_opt", "formulation_id", style)
    ax_dispatch.set_title("Generator Dispatch Shift")
    ax_dispatch.set_ylabel("Dispatch delta [MW]")
    ax_dispatch.axhline(0.0, color="#777777", linestyle=":", linewidth=0.9)
    ax_headroom.set_title("IBR Upward Headroom")
    ax_headroom.set_ylabel("Headroom up [MW]")
    ax_m.set_title("Scheduled Inertia")
    ax_m.set_ylabel("M [p.u.]")
    ax_d.set_title("Scheduled Damping")
    ax_d.set_ylabel("D [p.u.]")
    if bool(layout.get("baseline_lines", True)):
        ax_m.axhline(4.0, color="#777777", linestyle=":", linewidth=0.9)
        ax_d.axhline(2.0, color="#777777", linestyle=":", linewidth=0.9)
    ylims = dict(layout.get("ylims", {}) or {})
    for key, ax in {
        "dispatch": ax_dispatch,
        "headroom": ax_headroom,
        "M": ax_m,
        "D": ax_d,
    }.items():
        if key in ylims and len(ylims[key]) == 2:
            ax.set_ylim(tuple(ylims[key]))
    handles, labels = ax_dispatch.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, **_legend_kwargs(style))
    return fig


def summarize_dispatch_vis(gen_df: pd.DataFrame, ibr_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if gen_df.empty or ibr_df.empty:
        return pd.DataFrame(), ["Selected dispatch/VIS slice is empty."]
    ibr_summary = (
        ibr_df.groupby("formulation_id", dropna=False)
        .agg(
            mean_headroom_up=("headroom_up", "mean"),
            mean_M=("M_opt", "mean"),
            mean_D=("D_opt", "mean"),
        )
        .reset_index()
    )
    sentences = []
    for _, row in ibr_summary.iterrows():
        label = FORMULATION_LABELS.get(str(row["formulation_id"]), str(row["formulation_id"]))
        sentences.append(
            f"{label} has mean headroom {row['mean_headroom_up']:.3f} MW, mean M {row['mean_M']:.3f}, and mean D {row['mean_D']:.3f}."
        )
    return ibr_summary, sentences


def prepare_zone_md_allocations(
    unit_allocations: pd.DataFrame,
    *,
    formulation_id: str = "retained_vis",
    step_scale: float | None = None,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    df = unit_allocations.loc[unit_allocations["formulation_id"].astype(str) == str(formulation_id)].copy()
    if step_scale is not None:
        df = df.loc[_close_float_match(df["step_scale"], float(step_scale))].copy()
        if df.empty:
            df = load_zone_md_allocations_from_raw(step_scale=float(step_scale), formulation_id=formulation_id, repo_root=repo_root)
    order = ["all_loads", "owner_1", "owner_2", "owner_3", "owner_4"]
    df["scenario_target_key"] = pd.Categorical(df["scenario_target_key"], categories=order, ordered=True)
    return df.sort_values(["scenario_target_key", "index"]).reset_index(drop=True)


def load_zone_md_allocations_from_raw(
    *,
    step_scale: float,
    formulation_id: str = "retained_vis",
    repo_root: Path | None = None,
) -> pd.DataFrame:
    base = _zone_raw_results_root(step_scale, repo_root)
    rows: list[pd.DataFrame] = []
    if not base.exists():
        return pd.DataFrame()
    for dispatch_path in sorted(base.glob(f"*/{formulation_id}/{formulation_id}__*_dispatch_impact.csv")):
        summary_path = dispatch_path.with_name(dispatch_path.name.replace("_dispatch_impact.csv", "_summary.json"))
        summary_payload = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        scenario_id = str(summary_payload.get("scenario_id", dispatch_path.parent.parent.name))
        scenario_name = str(summary_payload.get("scenario_name", scenario_id))
        scenario = dict(summary_payload.get("scenario", {}) or {})
        frame = pd.read_csv(dispatch_path)
        frame = frame.loc[frame["row_type"].astype(str) == "ibr_summary"].copy()
        if frame.empty:
            continue
        frame["run_id"] = str(summary_payload.get("run_id", f"{formulation_id}__{scenario_id}"))
        frame["formulation_id"] = formulation_id
        frame["formulation_name"] = str(summary_payload.get("formulation_name", FORMULATION_LABELS.get(formulation_id, formulation_id)))
        frame["scenario_id"] = scenario_id
        frame["scenario_name"] = scenario_name
        frame["scenario_target_key"] = "all_loads" if scenario_id == "global_uniform" else scenario_id.replace("zone_", "")
        if frame["scenario_target_key"].iloc[0] == "global_uniform":
            frame["scenario_target_key"] = "all_loads"
        frame["scenario_target_label"] = (
            "Global uniform" if scenario_id == "global_uniform" else scenario_name.replace("Zone-based mismatch in ", "")
        )
        frame["step_scale"] = scenario.get("step_scale", step_scale)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    global_df = (
        df.loc[df["scenario_target_key"].astype(str) == "all_loads", ["index", "M_opt", "D_opt"]]
        .rename(columns={"M_opt": "M_opt_global", "D_opt": "D_opt_global"})
        .drop_duplicates(subset=["index"])
    )
    if not global_df.empty:
        df = df.merge(global_df, on="index", how="left")
        df["M_opt_vs_global"] = pd.to_numeric(df["M_opt"], errors="coerce") - pd.to_numeric(df["M_opt_global"], errors="coerce")
        df["D_opt_vs_global"] = pd.to_numeric(df["D_opt"], errors="coerce") - pd.to_numeric(df["D_opt_global"], errors="coerce")
    return df


def plot_zone_md_allocations(
    zone_df: pd.DataFrame,
    *,
    step_scale: float,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    figsize = tuple(style.get("figsize", (13.0, 5.2)))
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    palette = {
        "all_loads": "#4c6a92",
        "owner_1": "#c46646",
        "owner_2": "#8f6f4b",
        "owner_3": "#2f7f6d",
        "owner_4": "#8b6fb3",
    }
    categories = sorted(zone_df["index"].dropna().astype(int).unique().tolist())
    targets = [str(value) for value in zone_df["scenario_target_key"].dropna().astype(str).unique().tolist()]
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(len(targets), 1)
    for ax, col, title, baseline in [
        (axes[0], "M_opt", f"Inertia allocation at s_step={step_scale:.2f}", 4.0),
        (axes[1], "D_opt", f"Damping allocation at s_step={step_scale:.2f}", 2.0),
    ]:
        for idx, target in enumerate(targets):
            subset = zone_df.loc[zone_df["scenario_target_key"].astype(str) == target].set_index("index")
            vals = [pd.to_numeric(subset[col], errors="coerce").get(cat, np.nan) for cat in categories]
            label = str(subset["scenario_target_label"].dropna().iloc[0]) if not subset.empty else target
            ax.bar(
                x + (idx - (len(targets) - 1) / 2.0) * width,
                vals,
                width=width,
                label=label,
                color=palette.get(target, "#4c6a92"),
                alpha=0.9,
            )
        ax.axhline(baseline, color="#777777", linestyle=":", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"IBR {cat + 1}" for cat in categories], rotation=float(layout.get("label_rotation", 0)))
        ax.set_title(title)
        ax.set_ylabel("M [p.u.]" if col == "M_opt" else "D [p.u.]")
    axes[0].legend(**_legend_kwargs(style))
    return fig


def summarize_zone_md_allocations(zone_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if zone_df.empty:
        return pd.DataFrame(), ["No zone-targeted allocation rows were found for the selected step scale."]
    non_global = zone_df.loc[zone_df["scenario_target_key"].astype(str) != "all_loads"].copy()
    if non_global.empty:
        return pd.DataFrame(), ["Only the global allocation is present in the selected slice."]
    def _largest_shift(frame: pd.DataFrame, col: str) -> tuple[float, float]:
        values = pd.to_numeric(frame[col], errors="coerce").abs()
        valid = frame.loc[values.notna()].copy()
        if valid.empty:
            return np.nan, np.nan
        idx = pd.to_numeric(valid[col], errors="coerce").abs().idxmax()
        shift_value = float(pd.to_numeric(pd.Series([valid.loc[idx, col]]), errors="coerce").abs().iloc[0])
        return float(valid.loc[idx, "index"]) + 1.0, shift_value

    rows: list[dict[str, Any]] = []
    for label, frame in non_global.groupby(["scenario_target_label"], dropna=False):
        m_ibr, m_shift = _largest_shift(frame, "M_opt_vs_global")
        d_ibr, d_shift = _largest_shift(frame, "D_opt_vs_global")
        rows.append(
            {
                "scenario_target_label": label,
                "largest_abs_M_shift_ibr": m_ibr,
                "largest_abs_M_shift": m_shift,
                "largest_abs_D_shift_ibr": d_ibr,
                "largest_abs_D_shift": d_shift,
            }
        )
    shift_df = pd.DataFrame(rows)
    sentences = []
    for _, row in shift_df.iterrows():
        if pd.notna(row["largest_abs_M_shift_ibr"]) and pd.notna(row["largest_abs_D_shift_ibr"]):
            sentences.append(
                f"{row['scenario_target_label']} shifts M most on IBR {int(row['largest_abs_M_shift_ibr'])} by {row['largest_abs_M_shift']:.3f}, "
                f"and D most on IBR {int(row['largest_abs_D_shift_ibr'])} by {row['largest_abs_D_shift']:.3f}."
            )
        else:
            sentences.append(f"{row['scenario_target_label']} has no finite global-reference shift available in the selected slice.")
    return shift_df, sentences


def load_topk_screening(repo_root: Path | None = None) -> pd.DataFrame:
    base = get_paths(repo_root)["results_dir"] / "mtlsh_topk_screening" / "global"
    rows: list[dict[str, Any]] = []
    if not base.exists():
        return pd.DataFrame()
    for path in sorted(base.rglob("*_summary.json")):
        payload = json.loads(path.read_text())
        formulation_id = str(payload.get("formulation_id", ""))
        if "screen_" not in formulation_id:
            continue
        k_value = np.nan
        match = re.search(r"top(\d+)", formulation_id)
        if match:
            k_value = int(match.group(1))
        elif "all10" in formulation_id:
            k_value = 10
        scenario = dict(payload.get("scenario", {}) or {})
        rows.append(
            {
                "summary_json": str(path),
                "scenario_id": str(payload.get("scenario_id", path.parent.parent.name)),
                "formulation_id": formulation_id,
                "k": k_value,
                "status": str(payload.get("status", "")),
                "objective_total": payload.get("objective"),
                "base_scale": scenario.get("base_scale"),
                "step_scale": scenario.get("step_scale"),
            }
        )
    return pd.DataFrame(rows)


def prepare_topk_screening(topk_df: pd.DataFrame, scenario_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = dict(scenario_cfg or {})
    df = topk_df.copy()
    if cfg.get("base_scales"):
        df = df.loc[pd.to_numeric(df["base_scale"], errors="coerce").isin([float(v) for v in _as_list(cfg["base_scales"])])].copy()
    if cfg.get("step_scales"):
        df = df.loc[pd.to_numeric(df["step_scale"], errors="coerce").isin([float(v) for v in _as_list(cfg["step_scales"])])].copy()
    if cfg.get("k_values"):
        df = df.loc[pd.to_numeric(df["k"], errors="coerce").isin([int(v) for v in _as_list(cfg["k_values"])])].copy()
    return df.reset_index(drop=True)


def plot_topk_screening(
    topk_df: pd.DataFrame,
    *,
    feasible_subset_only: bool = True,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    fig, axes = plt.subplots(1, 2, figsize=tuple(style.get("figsize", (12.4, 4.6))), constrained_layout=True)
    feasible = topk_df.assign(is_feasible=topk_df["status"].astype(str).str.startswith("optimal"))
    feasible_counts = (
        feasible.groupby("k", dropna=False)["is_feasible"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("k")
    )
    axes[0].bar(feasible_counts["k"].astype(str), feasible_counts["sum"], color="#4c6a92", alpha=0.9)
    axes[0].set_title("Feasible cases by screening width")
    axes[0].set_ylabel("Number of feasible cases")

    objective_df = feasible.loc[feasible["is_feasible"]].copy()
    objective_df["objective_total"] = pd.to_numeric(objective_df["objective_total"], errors="coerce")
    objective_df = objective_df.loc[np.isfinite(objective_df["objective_total"])].copy()
    if feasible_subset_only and not objective_df.empty:
        ks = sorted(objective_df["k"].dropna().astype(int).unique().tolist())
        common_ids = set(objective_df.loc[objective_df["k"] == ks[0], "scenario_id"].astype(str))
        for k in ks[1:]:
            common_ids &= set(objective_df.loc[objective_df["k"] == k, "scenario_id"].astype(str))
        objective_df = objective_df.loc[objective_df["scenario_id"].astype(str).isin(common_ids)].copy()
    objective_plot = (
        objective_df.groupby("k", dropna=False)["objective_total"]
        .mean()
        .reset_index()
        .sort_values("k")
    )
    axes[1].plot(
        objective_plot["k"].astype(int),
        pd.to_numeric(objective_plot["objective_total"], errors="coerce") / 1000.0,
        marker="o",
        linewidth=2.0,
        color="#c46646",
    )
    axes[1].set_title("Mean objective on common feasible subset")
    axes[1].set_ylabel("Objective [$10^3$]")
    axes[1].set_xlabel("Top-k screening width")
    return fig


def summarize_topk_screening(topk_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if topk_df.empty:
        return pd.DataFrame(), ["No MTLSH top-k screening summaries were found."]
    numeric_objective = pd.to_numeric(topk_df["objective_total"], errors="coerce")
    filtered = topk_df.assign(is_feasible=topk_df["status"].astype(str).str.startswith("optimal"), objective_total=numeric_objective)
    filtered = filtered.loc[np.isfinite(filtered["objective_total"]) | ~filtered["is_feasible"]].copy()
    summary = (
        filtered
        .groupby("k", dropna=False)
        .agg(
            feasible_cases=("is_feasible", "sum"),
            total_cases=("is_feasible", "count"),
            feasible_rate=("is_feasible", "mean"),
            mean_objective=("objective_total", "mean"),
        )
        .reset_index()
        .sort_values("k")
    )
    summary["mean_objective"] = pd.to_numeric(summary["mean_objective"], errors="coerce")
    summary.loc[~np.isfinite(summary["mean_objective"]), "mean_objective"] = np.nan
    sentences = [
        (
            f"Top-{int(row['k'])} is feasible in {int(row['feasible_cases'])}/{int(row['total_cases'])} cases "
            f"({100.0 * float(row['feasible_rate']):.1f}%), with mean objective {float(row['mean_objective']) / 1000.0:.3f}k."
            if pd.notna(row["mean_objective"])
            else f"Top-{int(row['k'])} is feasible in {int(row['feasible_cases'])}/{int(row['total_cases'])} cases "
            f"({100.0 * float(row['feasible_rate']):.1f}%), but no finite common objective is available."
        )
        for _, row in summary.iterrows()
    ]
    return summary, sentences


def prepare_predicted_vs_replayed(
    detail_df: pd.DataFrame,
    scenario_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = dict(scenario_cfg or {})
    df = detail_df.copy()
    formulations = _as_list(cfg.get("formulations"))
    if not formulations and bool(cfg.get("retained_only", True)):
        formulations = ["ed_line_n1_surrogate"]
    if formulations:
        df = df.loc[df["formulation_id"].astype(str).isin([str(v) for v in formulations])].copy()
    metrics = _as_list(cfg.get("metrics"))
    if metrics:
        df = df.loc[df["metric_name"].astype(str).isin([str(v) for v in metrics])].copy()
    return df.reset_index(drop=True)


def _metric_scale(metric_name: str, override_map: dict[str, Any] | None = None) -> float:
    override = dict(override_map or {})
    return float(override.get(metric_name, METRIC_SCALES.get(metric_name, 1.0)))


def _metric_limit(metric_name: str, df: pd.DataFrame) -> float | None:
    relevant = pd.to_numeric(df.get("relevant_limit"), errors="coerce")
    if relevant is not None:
        relevant = relevant.dropna()
        if not relevant.empty:
            return float(relevant.abs().max())
    if metric_name in METRIC_LIMITS:
        return float(METRIC_LIMITS[metric_name])
    return None


def plot_predicted_vs_replayed(
    detail_df: pd.DataFrame,
    *,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
    scale_overrides: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    metrics = detail_df["metric_name"].dropna().astype(str).unique().tolist()
    ncols = min(3, max(len(metrics), 1))
    nrows = int(math.ceil(max(len(metrics), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(style.get("figsize", (13.0, 8.0))), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, metric in zip(axes_arr.ravel(), metrics):
        subset = detail_df.loc[detail_df["metric_name"].astype(str) == metric].copy()
        scale = _metric_scale(metric, scale_overrides)
        x = pd.to_numeric(subset["predicted_value"], errors="coerce") * scale
        y = pd.to_numeric(subset["replayed_value"], errors="coerce") * scale
        for formulation_id, frame in subset.groupby(subset["formulation_id"].astype(str)):
            form_x = pd.to_numeric(frame["predicted_value"], errors="coerce") * scale
            form_y = pd.to_numeric(frame["replayed_value"], errors="coerce") * scale
            ax.scatter(
                form_x,
                form_y,
                s=42,
                color=_palette_color(formulation_id, style),
                alpha=0.88,
                label=FORMULATION_LABELS.get(formulation_id, formulation_id),
                edgecolors="white",
                linewidths=0.5,
            )
        finite = pd.concat([x, y], axis=0).dropna()
        if not finite.empty:
            lo = float(finite.min())
            hi = float(finite.max())
            pad = 0.08 * (hi - lo if hi > lo else max(abs(lo), 1.0))
            lo -= pad
            hi += pad
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="#333333", linewidth=1.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        if bool(layout.get("safe_region_shading", True)):
            limit = _metric_limit(metric, subset)
            if limit is not None and metric in METRIC_LIMITS:
                lim = float(limit) * scale
                ax.axvspan(-lim, lim, color="#dff0df", alpha=0.12)
                ax.axhspan(-lim, lim, color="#dff0df", alpha=0.12)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Replayed")
        if bool(layout.get("equal_axis", True)):
            ax.set_aspect("equal", adjustable="box")
    for ax in axes_arr.ravel()[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes_arr.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, **_legend_kwargs(style))
    return fig


def summarize_predicted_vs_replayed(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if detail_df.empty:
        return pd.DataFrame(), ["No predicted-versus-replayed rows were selected."]
    summary = (
        detail_df.groupby(["formulation_id", "metric_name"], dropna=False)
        .agg(
            n_rows=("metric_name", "size"),
            mae=("abs_prediction_error", "mean"),
            rmse=("prediction_error", lambda s: float(np.sqrt(np.mean(np.square(pd.to_numeric(s, errors="coerce").dropna())))) if len(pd.to_numeric(s, errors="coerce").dropna()) else np.nan),
            max_abs_error=("abs_prediction_error", "max"),
            false_safe_count=("false_safe_flag", "sum"),
            replay_safe_rate=("violated_in_replay", lambda s: 1.0 - float(pd.to_numeric(s, errors="coerce").fillna(0.0).mean()) if len(s) else np.nan),
        )
        .reset_index()
    )
    sentences = [
        f"{FORMULATION_LABELS.get(str(row['formulation_id']), str(row['formulation_id']))} on {row['metric_name']} has MAE {row['mae']:.4f}, "
        f"RMSE {row['rmse']:.4f}, max error {row['max_abs_error']:.4f}, and false-safe count {int(row['false_safe_count'])}."
        for _, row in summary.iterrows()
    ]
    return summary, sentences


def prepare_replay_frequency(
    replay_df: pd.DataFrame,
    scenario_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = dict(scenario_cfg or {})
    df = replay_df.copy()
    formulations = _as_list(cfg.get("formulations"))
    if not formulations and bool(cfg.get("retained_only", False)):
        formulations = ["ed_line_n1_surrogate"]
    if formulations:
        df = df.loc[df["formulation"].astype(str).isin([str(v) for v in formulations])].copy()
    metrics = _as_list(cfg.get("metrics"))
    if metrics:
        df = df.loc[df["metric_name"].astype(str).isin([str(v) for v in metrics])].copy()
    return df.reset_index(drop=True)


def plot_replay_frequency(
    replay_df: pd.DataFrame,
    *,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
    scale_overrides: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    metrics = replay_df["metric_name"].dropna().astype(str).unique().tolist()
    ncols = min(3, max(len(metrics), 1))
    nrows = int(math.ceil(max(len(metrics), 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(style.get("figsize", (13.0, 8.0))), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    order = _ordered_formulations(sorted(replay_df["formulation"].dropna().astype(str).unique().tolist()))
    for ax, metric in zip(axes_arr.ravel(), metrics):
        subset = replay_df.loc[replay_df["metric_name"].astype(str) == metric].copy()
        scale = _metric_scale(metric, scale_overrides)
        labels: list[str] = []
        series_list: list[np.ndarray] = []
        colors: list[str] = []
        for formulation_id in order:
            frame = subset.loc[subset["formulation"].astype(str) == formulation_id].copy()
            vals = pd.to_numeric(frame["abs_replayed_value"], errors="coerce").dropna() * scale
            if vals.empty:
                continue
            labels.append(FORMULATION_LABELS.get(formulation_id, formulation_id))
            series_list.append(vals.to_numpy())
            colors.append(_palette_color(formulation_id, style))
        if not series_list:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(series_list, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("#2b2826")
        for median in bp["medians"]:
            median.set_color("#2b2826")
        ax.set_xticklabels(labels, rotation=float(layout.get("label_rotation", 15)), ha="right")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_ylabel("Absolute replayed value")
        if bool(layout.get("show_limits", True)):
            limit = _metric_limit(metric, subset)
            if limit is not None:
                ax.axhline(float(limit) * scale, color="#c46646", linestyle="--", linewidth=1.0)
    for ax in axes_arr.ravel()[len(metrics):]:
        ax.set_visible(False)
    return fig


def summarize_replay_frequency(replay_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if replay_df.empty:
        return pd.DataFrame(), ["No replay-frequency rows were selected."]
    summary = (
        replay_df.groupby(["formulation", "metric_name"], dropna=False)
        .agg(
            n_rows=("metric_name", "size"),
            mean_abs_replayed=("abs_replayed_value", "mean"),
            max_abs_replayed=("abs_replayed_value", "max"),
            violation_rate=("violated_in_replay", "mean"),
        )
        .reset_index()
    )
    sentences = [
        f"{FORMULATION_LABELS.get(str(row['formulation']), str(row['formulation']))} on {row['metric_name']} has mean |replay| {row['mean_abs_replayed']:.4f}, "
        f"max |replay| {row['max_abs_replayed']:.4f}, and violation rate {100.0 * float(row['violation_rate']):.1f}%."
        for _, row in summary.iterrows()
    ]
    return summary, sentences


def load_trace_family(family_key: str, repo_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    family = dict(TRACE_FAMILY_CATALOG[family_key])
    folder = _resolve_rel(family["folder"], repo_root)
    summary_path = folder / "trace_summary.csv"
    compact_path = folder / "compact_metrics.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    compact_df = pd.read_csv(compact_path) if compact_path.exists() else pd.DataFrame()
    traces: dict[str, pd.DataFrame] = {}
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            trace_path = Path(str(row["trace_csv"]))
            if trace_path.exists():
                traces[str(row["label"])] = pd.read_csv(trace_path)
    return summary_df, compact_df, traces, family


def rerun_trace_family(
    family_key: str,
    *,
    enabled: bool = False,
    python_bin: str | None = None,
    repo_root: Path | None = None,
) -> str:
    if not enabled:
        return "Trace rerun disabled."
    family = TRACE_FAMILY_CATALOG[family_key]
    root = find_repo_root(repo_root)
    script = root / "results" / "thesis_optimization_results" / "scripts" / "export_replay_trace_panel.py"
    config_path = _resolve_rel(family["rerun_config"], repo_root)
    py = python_bin or sys.executable
    completed = subprocess.run(
        [py, str(script), "--config", str(config_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip() or completed.stderr.strip() or "Replay trace export completed."


def _style_lookup_from_summary(summary_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if summary_df.empty:
        return out
    for _, row in summary_df.iterrows():
        raw_color = row.get("color", "#1f77b4")
        color = "#1f77b4" if pd.isna(raw_color) or str(raw_color).lower() == "nan" else str(raw_color)
        raw_linestyle = row.get("linestyle", "-")
        linestyle = "-" if pd.isna(raw_linestyle) or str(raw_linestyle).lower() == "nan" else str(raw_linestyle)
        out[str(row["label"])] = {
            "linestyle": linestyle,
            "linewidth": float(row.get("linewidth", 1.8) or 1.8),
            "alpha": float(row.get("alpha", 1.0) or 1.0),
            "color": color,
        }
    return out


def _trace_label(label: str, label_map: dict[str, str] | None = None) -> str:
    mapped = dict(label_map or {})
    return str(mapped.get(label, label))


def _trace_style(
    label: str,
    style_lookup: dict[str, dict[str, Any]],
    layout_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = dict(style_lookup.get(label, {}))
    layout = dict(layout_cfg or {})
    color_map = dict(layout.get("color_map", {}) or {})
    linestyle_map = dict(layout.get("linestyle_map", {}) or {})
    linewidth_map = dict(layout.get("linewidth_map", {}) or {})
    alpha_map = dict(layout.get("alpha_map", {}) or {})
    if label in color_map:
        spec["color"] = str(color_map[label])
    if label in linestyle_map:
        spec["linestyle"] = str(linestyle_map[label])
    if label in linewidth_map:
        spec["linewidth"] = float(linewidth_map[label])
    if label in alpha_map:
        spec["alpha"] = float(alpha_map[label])
    return spec


def plot_trace_combined(
    summary_df: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    *,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    fig, axes = plt.subplots(1, 3, figsize=tuple(style.get("figsize", (15.0, 4.6))), constrained_layout=True)
    style_lookup = _style_lookup_from_summary(summary_df)
    label_map = dict(layout.get("label_map", {}) or {})
    columns = [
        ("delta_f_coi_hz", r"$\Delta f_{\mathrm{COI}}$ [Hz]"),
        ("rocof_coi_hz_per_s", r"$\mathrm{RoCoF}_{\mathrm{COI}}$ [Hz/s]"),
        ("max_abs_delta_p_ibr", r"Max $|\Delta P_{\mathrm{IBR}}|$ [MW]"),
    ]
    for ax, (column, title) in zip(axes, columns):
        for label, trace_df in traces.items():
            if column not in trace_df.columns:
                continue
            spec = _trace_style(label, style_lookup, layout)
            ax.plot(
                pd.to_numeric(trace_df["time_s"], errors="coerce"),
                pd.to_numeric(trace_df[column], errors="coerce"),
                label=_trace_label(label, label_map),
                color=str(spec.get("color", "#1f77b4")),
                linestyle=str(spec.get("linestyle", "-")),
                linewidth=float(spec.get("linewidth", 1.8)),
                alpha=float(spec.get("alpha", 1.0)),
            )
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        if layout.get("xlim"):
            ax.set_xlim(tuple(layout["xlim"]))
    axes[0].legend(**_legend_kwargs(style))
    return fig


def _trace_columns_with_prefix(trace_df: pd.DataFrame, prefix: str) -> list[str]:
    return [str(col) for col in trace_df.columns if str(col).startswith(prefix)]


def plot_trace_frequency_panel(
    summary_df: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    *,
    columns: list[str] | None = None,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    if not traces:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No traces loaded", ha="center", va="center", transform=ax.transAxes)
        return fig
    first_trace = next(iter(traces.values()))
    selected_columns = columns or ["delta_f_coi_hz", "rocof_coi_hz_per_s"] + _trace_columns_with_prefix(first_trace, "delta_f_bus_")[:2]
    fig, axes = plt.subplots(1, len(selected_columns), figsize=tuple(style.get("figsize", (15.0, 4.6))), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    style_lookup = _style_lookup_from_summary(summary_df)
    label_map = dict(layout.get("label_map", {}) or {})
    for ax, column in zip(axes_arr, selected_columns):
        for label, trace_df in traces.items():
            if column not in trace_df.columns:
                continue
            spec = _trace_style(label, style_lookup, layout)
            ax.plot(
                pd.to_numeric(trace_df["time_s"], errors="coerce"),
                pd.to_numeric(trace_df[column], errors="coerce"),
                label=_trace_label(label, label_map),
                color=str(spec.get("color", "#1f77b4")),
                linestyle=str(spec.get("linestyle", "-")),
                linewidth=float(spec.get("linewidth", 1.8)),
                alpha=float(spec.get("alpha", 1.0)),
            )
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("Time [s]")
        if layout.get("xlim"):
            ax.set_xlim(tuple(layout["xlim"]))
    axes_arr[0].legend(**_legend_kwargs(style))
    return fig


def plot_trace_ibr_panel(
    summary_df: pd.DataFrame,
    traces: dict[str, pd.DataFrame],
    *,
    columns: list[str] | None = None,
    layout_cfg: dict[str, Any] | None = None,
    style_cfg: dict[str, Any] | None = None,
) -> plt.Figure:
    layout = default_layout_cfg()
    layout.update(dict(layout_cfg or {}))
    style = default_style_cfg()
    style.update(dict(style_cfg or {}))
    configure_notebook_style(style)
    if not traces:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No traces loaded", ha="center", va="center", transform=ax.transAxes)
        return fig
    first_trace = next(iter(traces.values()))
    selected_columns = columns or _trace_columns_with_prefix(first_trace, "Delta_P_IBR_")
    fig, axes = plt.subplots(1, len(selected_columns), figsize=tuple(style.get("figsize", (16.0, 4.8))), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    style_lookup = _style_lookup_from_summary(summary_df)
    label_map = dict(layout.get("label_map", {}) or {})
    for ax, column in zip(axes_arr, selected_columns):
        for label, trace_df in traces.items():
            if column not in trace_df.columns:
                continue
            spec = _trace_style(label, style_lookup, layout)
            ax.plot(
                pd.to_numeric(trace_df["time_s"], errors="coerce"),
                100.0 * pd.to_numeric(trace_df[column], errors="coerce"),
                label=_trace_label(label, label_map),
                color=str(spec.get("color", "#1f77b4")),
                linestyle=str(spec.get("linestyle", "-")),
                linewidth=float(spec.get("linewidth", 1.8)),
                alpha=float(spec.get("alpha", 1.0)),
            )
        ax.set_title(column.replace("_", " "))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("MW")
        if layout.get("xlim"):
            ax.set_xlim(tuple(layout["xlim"]))
    axes_arr[0].legend(**_legend_kwargs(style))
    return fig


def summarize_trace_family(summary_df: pd.DataFrame, compact_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not compact_df.empty:
        sentences = [
            f"{row['label']} reaches max |df| {row['max_abs_delta_f_hz_replay']:.3f} Hz, "
            f"max |RoCoF| {row['max_abs_rocof_hz_per_s_replay']:.3f} Hz/s, and max |Delta P_IBR| {row['max_abs_delta_p_ibr_MW_replay']:.3f} MW."
            for _, row in compact_df.iterrows()
        ]
        return compact_df, sentences
    if summary_df.empty:
        return pd.DataFrame(), ["No trace-summary rows were selected."]
    cols = [
        "label",
        "max_abs_dev_replayed",
        "max_abs_rocof_replayed",
        "max_abs_delta_p_ibr",
        "M_1",
        "M_2",
        "M_3",
        "M_4",
        "D_1",
        "D_2",
        "D_3",
        "D_4",
    ]
    available = [col for col in cols if col in summary_df.columns]
    reduced = summary_df[available].copy()
    sentences = [
        f"{row['label']} reaches max |df| {row.get('max_abs_dev_replayed', np.nan):.3f} Hz, "
        f"max |RoCoF| {row.get('max_abs_rocof_replayed', np.nan):.3f} Hz/s, and max |Delta P_IBR| {row.get('max_abs_delta_p_ibr', np.nan):.3f}."
        for _, row in reduced.iterrows()
    ]
    return reduced, sentences
