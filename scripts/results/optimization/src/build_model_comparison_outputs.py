from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

try:
    from .analysis_utils import write_latex_table, write_markdown_table
    from .plot_utils import save_figure, set_thesis_style
except ImportError:
    ROOT = Path(__file__).resolve().parents[3]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from analysis_utils import write_latex_table, write_markdown_table  # type: ignore
    from plot_utils import save_figure, set_thesis_style  # type: ignore


ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = ROOT / "results" / "thesis_optimization_results"
DEFAULT_RESULTS_ROOT = ANALYSIS_ROOT / "results" / "by_model"
DEFAULT_OUTPUT_ROOT = ANALYSIS_ROOT / "outputs" / "model_comparison"

FORMULATION_ORDER = [
    "ed",
    "ed_line",
    "ed_line_n1",
    "ed_surrogate",
    "ed_line_n1_surrogate",
    "ed_line_n1_surrogate_redispatch",
]
FORMULATION_LABELS = {
    "ed": "ED",
    "ed_line": "ED+Line",
    "ed_line_n1": "ED+Line+N-1",
    "ed_surrogate": "ED+NN",
    "ed_line_n1_surrogate": "ED+Line+N-1+NN",
    "ed_line_n1_surrogate_redispatch": "NN+Redispatch",
    "retained_vis": "Retained VIS",
    "retained_vis_area_tied": "Area-Tied VIS",
}
MODEL_LABELS = {
    "mtlsh": "MTLSH",
    "mtlsh_no_dispatch": "MTLSH No Dispatch",
    "mlp": "MLP",
    "picnn": "PICNN",
}
MODEL_COLORS = {
    "mtlsh": "#285c8e",
    "mtlsh_no_dispatch": "#3f7f4c",
    "mlp": "#c06a2b",
    "picnn": "#c06a2b",
}
FORMULATION_HATCHES = {
    "retained_vis": "",
    "retained_vis_area_tied": "///",
}
SCENARIO_ORDER = ["global_uniform", "zone_owner_1", "zone_owner_2", "zone_owner_3", "zone_owner_4"]
SCENARIO_LABELS = {
    "global_uniform": "Global",
    "zone_owner_1": "Owner 1",
    "zone_owner_2": "Owner 2",
    "zone_owner_3": "Owner 3",
    "zone_owner_4": "Owner 4",
}


def _model_label(model_tag: str) -> str:
    return MODEL_LABELS.get(str(model_tag), str(model_tag).upper())


def _formulation_label(formulation_id: str) -> str:
    return FORMULATION_LABELS.get(str(formulation_id), str(formulation_id))


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(num):
        return ""
    if abs(num) >= 100:
        return f"{num:.1f}"
    if abs(num) >= 10:
        return f"{num:.2f}"
    return f"{num:.3f}"


def _write_table_bundle(base_dir: Path, stem: str, df: pd.DataFrame) -> dict[str, str]:
    tables_dir = base_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{stem}.csv"
    md_path = tables_dir / f"{stem}.md"
    tex_path = tables_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    write_markdown_table(md_path, df)
    write_latex_table(tex_path, df)
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path)}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _existing_summary_csv(model_root: Path, stem: str) -> Path | None:
    path = model_root / f"{stem}_summary.csv"
    return path if path.exists() else None


def _load_suite_summary(model_tag: str, results_root: Path, stem: str) -> tuple[pd.DataFrame, list[str]]:
    model_root = results_root / model_tag
    path = _existing_summary_csv(model_root, stem)
    if path is None:
        return pd.DataFrame(), [f"{_model_label(model_tag)}: missing {stem}_summary.csv under {model_root}"]

    df = pd.read_csv(path)
    if df.empty:
        return df, [f"{_model_label(model_tag)}: {path} exists but is empty"]
    df["model_tag"] = str(model_tag)
    df["model_name"] = _model_label(model_tag)
    return df, []


def _enrich_from_summary_json(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "summary_json" not in df.columns:
        return df.copy()

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload: dict[str, Any] = {}
        raw = str(row.get("summary_json", "")).strip()
        if raw:
            path = Path(raw)
            if path.exists():
                payload = _read_json(path)
        security = dict(payload.get("security_checks", {}) or {})
        line_security = dict(security.get("line_security", {}) or {})
        solver_stats = dict(payload.get("solver_stats", {}) or {})
        dispatch_summary = dict(payload.get("dispatch_summary", {}) or {})
        rows.append(
            {
                "objective_reserve_only": payload.get("objective_reserve_only"),
                "objective_reserve_up_only": payload.get("objective_reserve_up_only"),
                "objective_reserve_postcont_only": payload.get("objective_reserve_postcont_only"),
                "predicted_output_violation_count": security.get("predicted_output_violation_count"),
                "predicted_outputs_within_limits": security.get("predicted_outputs_within_limits"),
                "base_max_loading_pct": line_security.get("base_max_loading_pct"),
                "n1_max_loading_pct": line_security.get("n1_max_loading_pct"),
                "n1_total_line_violations": line_security.get("n1_total_line_violations"),
                "solve_wall_time_sec": solver_stats.get("solve_wall_time_sec"),
                "dispatch_m_total": float(np.nansum(np.asarray(dispatch_summary.get("m_opt", []), dtype=float)))
                if dispatch_summary.get("m_opt") is not None
                else np.nan,
                "dispatch_d_total": float(np.nansum(np.asarray(dispatch_summary.get("d_opt", []), dtype=float)))
                if dispatch_summary.get("d_opt") is not None
                else np.nan,
            }
        )
    extra = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


def _build_formulation_tradeoff(models: Iterable[str], results_root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for model_tag in models:
        df, local_warnings = _load_suite_summary(model_tag, results_root, "formulation_comparison")
        warnings.extend(local_warnings)
        if df.empty:
            continue
        df = _enrich_from_summary_json(df)
        df = df.loc[df["formulation_id"].astype(str).isin(FORMULATION_ORDER)].copy()
        df["formulation_order"] = pd.Categorical(df["formulation_id"], categories=FORMULATION_ORDER, ordered=True)
        df = df.sort_values("formulation_order")
        frames.append(df)

    if not frames:
        return pd.DataFrame(), warnings

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.rename(
        columns={
            "objective_dispatch_only": "dispatch_cost_component",
            "objective_reserve_only": "reserve_cost_component",
            "objective": "total_cost",
            "n_variables_binary": "binary_vars",
            "n_constraints_scalar_total": "scalar_constraints",
        }
    )
    cols = [
        "model_name",
        "formulation_id",
        "formulation_name",
        "status",
        "total_cost",
        "dispatch_cost_component",
        "reserve_cost_component",
        "cost_increase_pct_vs_ed",
        "solve_time_sec",
        "solve_wall_time_sec",
        "binary_vars",
        "scalar_constraints",
    ]
    out = out[cols].copy()
    out["formulation_name"] = out["formulation_id"].map(_formulation_label)
    return out, warnings


def _build_security_table(models: Iterable[str], results_root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for model_tag in models:
        df, local_warnings = _load_suite_summary(model_tag, results_root, "security_checks")
        warnings.extend(local_warnings)
        if df.empty:
            continue
        df = _enrich_from_summary_json(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), warnings

    out = pd.concat(frames, ignore_index=True, sort=False).rename(
        columns={
            "objective_dispatch_only": "dispatch_cost_component",
            "objective_reserve_only": "reserve_cost_component",
            "objective": "total_cost",
        }
    )
    cols = [
        "model_name",
        "formulation_id",
        "formulation_name",
        "status",
        "total_cost",
        "dispatch_cost_component",
        "reserve_cost_component",
        "base_max_loading_pct",
        "n1_max_loading_pct",
        "n1_total_line_violations",
        "predicted_output_violation_count",
        "solve_time_sec",
    ]
    out = out[cols].copy()
    out["formulation_name"] = out["formulation_id"].map(_formulation_label)
    return out, warnings


def _build_redispatch_table(models: Iterable[str], results_root: Path) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for model_tag in models:
        df, local_warnings = _load_suite_summary(model_tag, results_root, "redispatch_sensitivity")
        warnings.extend(local_warnings)
        if df.empty:
            continue
        df = _enrich_from_summary_json(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), warnings

    out = pd.concat(frames, ignore_index=True, sort=False).rename(
        columns={
            "objective_dispatch_only": "dispatch_cost_component",
            "objective_reserve_only": "reserve_cost_component",
            "objective": "total_cost",
        }
    )
    cols = [
        "model_name",
        "formulation_id",
        "formulation_name",
        "total_cost",
        "dispatch_cost_component",
        "reserve_cost_component",
        "cost_increase_pct_vs_ed",
        "solve_time_sec",
        "n_variables_total",
        "n_variables_binary",
    ]
    out = out[cols].copy()
    out["formulation_name"] = out["formulation_id"].map(_formulation_label)
    return out, warnings


def _aggregate_zone_dispatch(dispatch_path: Path) -> dict[str, Any]:
    df = pd.read_csv(dispatch_path)
    df.columns = [str(col).strip() for col in df.columns]
    if "row_type" not in df.columns:
        return {}
    ibr_df = df.loc[df["row_type"].astype(str) == "ibr_summary"].copy()
    return {
        "total_M_opt": float(pd.to_numeric(ibr_df["M_opt"], errors="coerce").fillna(0.0).sum()),
        "total_D_opt": float(pd.to_numeric(ibr_df["D_opt"], errors="coerce").fillna(0.0).sum()),
        "total_delta_p_predicted": float(pd.to_numeric(ibr_df["delta_p_predicted"], errors="coerce").fillna(0.0).sum()),
        "total_headroom_up": float(pd.to_numeric(ibr_df["headroom_up"], errors="coerce").fillna(0.0).sum()),
        "total_headroom_down": float(pd.to_numeric(ibr_df["headroom_down"], errors="coerce").fillna(0.0).sum()),
        "total_reserve_postcont_cost": float(
            pd.to_numeric(ibr_df["reserve_postcont_cost_component"], errors="coerce").fillna(0.0).sum()
        ),
        "total_reserve_up_cost": float(
            pd.to_numeric(ibr_df["reserve_up_cost_component"], errors="coerce").fillna(0.0).sum()
        ),
    }


def _build_zone_table(models: Iterable[str], results_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for model_tag in models:
        df, local_warnings = _load_suite_summary(model_tag, results_root, "zone_mismatch_vis_sensitivity")
        warnings.extend(local_warnings)
        if df.empty:
            continue
        df = _enrich_from_summary_json(df)
        for _, row in df.iterrows():
            dispatch_path_raw = str(row.get("dispatch_impact_csv", "")).strip()
            if not dispatch_path_raw:
                continue
            dispatch_path = Path(dispatch_path_raw)
            if not dispatch_path.exists():
                warnings.append(f"{_model_label(model_tag)}: missing dispatch-impact CSV {dispatch_path}")
                continue
            agg = _aggregate_zone_dispatch(dispatch_path)
            scenario_id = str(row.get("scenario_id", ""))
            formulation_id = str(row.get("formulation_id", ""))
            rows.append(
                {
                    "model_name": _model_label(model_tag),
                    "scenario_id": scenario_id,
                    "scenario_name": SCENARIO_LABELS.get(scenario_id, str(row.get("scenario_name", scenario_id))),
                    "formulation_id": formulation_id,
                    "formulation_name": _formulation_label(formulation_id),
                    "total_cost": row.get("objective"),
                    "dispatch_cost_component": row.get("objective_dispatch_only"),
                    "reserve_cost_component": row.get("objective_reserve_only"),
                    **agg,
                }
            )
    if not rows:
        return pd.DataFrame(), warnings
    out = pd.DataFrame(rows)
    out["scenario_order"] = pd.Categorical(out["scenario_id"], categories=SCENARIO_ORDER, ordered=True)
    out["formulation_order"] = pd.Categorical(
        out["formulation_id"], categories=["retained_vis", "retained_vis_area_tied"], ordered=True
    )
    out = out.sort_values(["scenario_order", "model_name", "formulation_order"]).drop(columns=["scenario_order", "formulation_order"])
    return out, warnings


def _plot_formulation_tradeoff(df: pd.DataFrame, base_dir: Path) -> list[str]:
    if df.empty:
        return []
    plot_df = df.loc[df["status"].astype(str).str.startswith("optimal")].copy()
    if plot_df.empty:
        return []

    set_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    metrics = [
        ("cost_increase_pct_vs_ed", "Cost Increase vs ED [%]"),
        ("solve_time_sec", "Solve Time [s]"),
    ]
    model_order = [m for m in MODEL_LABELS.values() if m in set(plot_df["model_name"].astype(str))]
    x = np.arange(len(FORMULATION_ORDER), dtype=float)
    width = 0.8 / max(len(model_order), 1)

    for ax, (metric, ylabel) in zip(axes, metrics):
        for idx, model_name in enumerate(model_order):
            sub = plot_df.loc[plot_df["model_name"].astype(str) == model_name].copy()
            sub["formulation_id"] = pd.Categorical(sub["formulation_id"], categories=FORMULATION_ORDER, ordered=True)
            sub = sub.sort_values("formulation_id")
            vals = (
                pd.to_numeric(sub.set_index("formulation_id").reindex(FORMULATION_ORDER)[metric], errors="coerce")
                .fillna(np.nan)
                .to_numpy()
            )
            offset = (idx - (len(model_order) - 1) / 2.0) * width
            model_tag = next((k for k, v in MODEL_LABELS.items() if v == model_name), model_name.lower())
            ax.bar(x + offset, vals, width=width, color=MODEL_COLORS.get(model_tag, "#4c6a92"), alpha=0.9, label=model_name)
        ax.set_xticks(x)
        ax.set_xticklabels([_formulation_label(fid) for fid in FORMULATION_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        if metric == "solve_time_sec":
            ax.set_yscale("log")
        ax.set_title("Formulation Trade-Off" if metric == "cost_increase_pct_vs_ed" else "Runtime Burden")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("Model-by-Formulation Comparison", y=1.03)
    fig_dir = base_dir / "figures"
    return [str(path) for path in save_figure(fig, "model_formulation_tradeoff", fig_dir)]


def _plot_security_profile(df: pd.DataFrame, base_dir: Path) -> list[str]:
    if df.empty:
        return []
    set_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))
    metrics = [
        ("base_max_loading_pct", "Base Max Loading [%]"),
        ("n1_max_loading_pct", "Post-N-1 Max Loading [%]"),
        ("predicted_output_violation_count", "Predicted Output Violations [-]"),
    ]
    plot_df = df.copy()
    model_order = plot_df["model_name"].astype(str).tolist()
    x = np.arange(len(model_order), dtype=float)

    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy()
        colors = [
            MODEL_COLORS.get(next((k for k, v in MODEL_LABELS.items() if v == model_name), model_name.lower()), "#4c6a92")
            for model_name in model_order
        ]
        ax.bar(x, vals, color=colors, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Retained Security Formulation Profile", y=1.02)
    fig_dir = base_dir / "figures"
    return [str(path) for path in save_figure(fig, "model_security_profile", fig_dir)]


def _plot_redispatch_effect(df: pd.DataFrame, base_dir: Path) -> list[str]:
    if df.empty:
        return []
    set_thesis_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    metrics = [
        ("dispatch_cost_component", "Dispatch Cost Component"),
        ("reserve_cost_component", "Reserve Cost Component"),
        ("total_cost", "Total Objective"),
    ]
    formulation_order = ["ed_line_n1_surrogate", "ed_line_n1_surrogate_redispatch"]
    x = np.arange(len(MODEL_LABELS), dtype=float)
    width = 0.34
    model_order = [label for label in MODEL_LABELS.values() if label in set(df["model_name"].astype(str))]

    for ax, (metric, title) in zip(axes, metrics):
        for idx, formulation_id in enumerate(formulation_order):
            sub = df.loc[df["formulation_id"].astype(str) == formulation_id].copy()
            vals = []
            for model_name in model_order:
                val = pd.to_numeric(
                    sub.loc[sub["model_name"].astype(str) == model_name, metric],
                    errors="coerce",
                )
                vals.append(float(val.iloc[0]) if not val.empty else np.nan)
            offset = (idx - 0.5) * width
            ax.bar(
                np.arange(len(model_order), dtype=float) + offset,
                vals,
                width=width,
                label=_formulation_label(formulation_id),
                color="#5b8c45" if formulation_id.endswith("redispatch") else "#2f7f6d",
                alpha=0.9,
            )
        ax.set_xticks(np.arange(len(model_order), dtype=float))
        ax.set_xticklabels(model_order)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Redispatch Sensitivity by Model", y=1.03)
    fig_dir = base_dir / "figures"
    return [str(path) for path in save_figure(fig, "model_redispatch_effect", fig_dir)]


def _plot_zone_totals(df: pd.DataFrame, base_dir: Path) -> list[str]:
    if df.empty:
        return []
    set_thesis_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.8))
    metric_specs = [
        ("total_M_opt", "Total Scheduled M [-]"),
        ("total_D_opt", "Total Scheduled D [-]"),
        ("total_delta_p_predicted", "Total Predicted Delta P [p.u.]"),
        ("total_headroom_up", "Total Upward Headroom [p.u.]"),
    ]
    scenario_order = [sid for sid in SCENARIO_ORDER if sid in set(df["scenario_id"].astype(str))]
    combo_rows = df[["model_name", "formulation_id"]].drop_duplicates().reset_index(drop=True)
    combo_rows["legend_label"] = combo_rows.apply(
        lambda row: f"{row['model_name']} | {_formulation_label(str(row['formulation_id']))}",
        axis=1,
    )
    x = np.arange(len(scenario_order), dtype=float)
    width = 0.8 / max(len(combo_rows), 1)

    for ax, (metric, title) in zip(axes.ravel(), metric_specs):
        for idx, combo in combo_rows.iterrows():
            sub = df.loc[
                (df["model_name"].astype(str) == str(combo["model_name"]))
                & (df["formulation_id"].astype(str) == str(combo["formulation_id"]))
            ].copy()
            sub["scenario_id"] = pd.Categorical(sub["scenario_id"], categories=scenario_order, ordered=True)
            sub = sub.sort_values("scenario_id")
            vals = (
                pd.to_numeric(sub.set_index("scenario_id").reindex(scenario_order)[metric], errors="coerce")
                .fillna(0.0)
                .to_numpy()
            )
            model_tag = next(
                (k for k, v in MODEL_LABELS.items() if v == str(combo["model_name"])),
                str(combo["model_name"]).lower(),
            )
            ax.bar(
                x + (idx - (len(combo_rows) - 1) / 2.0) * width,
                vals,
                width=width,
                label=str(combo["legend_label"]),
                color=MODEL_COLORS.get(model_tag, "#4c6a92"),
                hatch=FORMULATION_HATCHES.get(str(combo["formulation_id"]), ""),
                alpha=0.9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(sid, sid) for sid in scenario_order], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Zone-Mismatch VIS Allocation Summary", y=1.02)
    fig_dir = base_dir / "figures"
    return [str(path) for path in save_figure(fig, "model_zone_vis_totals", fig_dir)]


def build_outputs(*, models: list[str], results_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    (output_root / "merged_results").mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    outputs: dict[str, Any] = {"tables": {}, "figures": [], "warnings": warnings}

    tradeoff_df, local = _build_formulation_tradeoff(models, results_root)
    warnings.extend(local)
    if not tradeoff_df.empty:
        tradeoff_df.to_csv(output_root / "merged_results" / "model_formulation_tradeoff.csv", index=False)
        outputs["tables"]["model_formulation_tradeoff"] = _write_table_bundle(output_root, "model_formulation_tradeoff", tradeoff_df)
        outputs["figures"].extend(_plot_formulation_tradeoff(tradeoff_df, output_root))

    security_df, local = _build_security_table(models, results_root)
    warnings.extend(local)
    if not security_df.empty:
        security_df.to_csv(output_root / "merged_results" / "model_security_retained.csv", index=False)
        outputs["tables"]["model_security_retained"] = _write_table_bundle(output_root, "model_security_retained", security_df)
        outputs["figures"].extend(_plot_security_profile(security_df, output_root))

    redispatch_df, local = _build_redispatch_table(models, results_root)
    warnings.extend(local)
    if not redispatch_df.empty:
        redispatch_df.to_csv(output_root / "merged_results" / "model_redispatch_effect.csv", index=False)
        outputs["tables"]["model_redispatch_effect"] = _write_table_bundle(output_root, "model_redispatch_effect", redispatch_df)
        outputs["figures"].extend(_plot_redispatch_effect(redispatch_df, output_root))

    zone_df, local = _build_zone_table(models, results_root)
    warnings.extend(local)
    if not zone_df.empty:
        zone_df.to_csv(output_root / "merged_results" / "model_zone_vis_totals.csv", index=False)
        outputs["tables"]["model_zone_vis_totals"] = _write_table_bundle(output_root, "model_zone_vis_totals", zone_df)
        outputs["figures"].extend(_plot_zone_totals(zone_df, output_root))

    with (output_root / "model_comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-facing MTLSH/PICNN comparison tables and figures.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mtlsh", "picnn"],
        help="Model result folders under results/thesis_optimization_results/results/by_model.",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing per-model optimization results.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for the comparison tables and figures.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = (ROOT / results_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()

    outputs = build_outputs(models=[str(v) for v in args.models], results_root=results_root, output_root=output_root)
    print(f"[build_model_comparison_outputs] output_root={output_root}")
    for warning in outputs.get("warnings", []):
        print(f"[build_model_comparison_outputs] warning={warning}")


if __name__ == "__main__":
    main()
