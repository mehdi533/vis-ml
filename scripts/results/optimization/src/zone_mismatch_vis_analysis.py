from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from results.thesis_optimization_results.src.analysis_utils import (
        write_latex_table,
        write_markdown_table,
    )
    from results.thesis_optimization_results.src.area_vis_analysis import _load_area_map
    from results.thesis_optimization_results.src.config_analysis import (
        FIGURES_DIR,
        TABLES_DIR,
        ensure_output_dirs,
    )
    from results.thesis_optimization_results.src.plot_utils import save_figure, set_thesis_style
except ImportError:
    from analysis_utils import write_latex_table, write_markdown_table  # type: ignore
    from area_vis_analysis import _load_area_map  # type: ignore
    from config_analysis import FIGURES_DIR, TABLES_DIR, ensure_output_dirs  # type: ignore
    from plot_utils import save_figure, set_thesis_style  # type: ignore


def _write_table_bundle(stem: str, df: pd.DataFrame) -> dict[str, str]:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    tex_path = TABLES_DIR / f"{stem}.tex"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_markdown_table(md_path, df)
    write_latex_table(tex_path, df)
    return {"csv": str(csv_path), "markdown": str(md_path), "latex": str(tex_path)}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _target_owner_labels(scenario_cfg: dict[str, Any]) -> tuple[str, str]:
    owners = [str(value) for value in list(scenario_cfg.get("load_step_target_owners") or scenario_cfg.get("load_step_owners") or [])]
    if not owners:
        return "all_loads", "Global uniform"
    owner_key = "owner_" + "_".join(owners)
    owner_label = "Owner " + ", ".join(owners)
    return owner_key, owner_label


def _load_ibr_rows(summary_path: Path) -> tuple[pd.DataFrame, str | None]:
    suite_payload = _load_json(summary_path)
    rows: list[pd.DataFrame] = []
    system_case: str | None = None

    for suite_row in list(suite_payload.get("rows") or []):
        summary_json_raw = str(suite_row.get("summary_json", "")).strip()
        if not summary_json_raw:
            continue
        summary_json = Path(summary_json_raw)
        if not summary_json.exists():
            continue

        payload = _load_json(summary_json)
        artifacts = dict(payload.get("artifacts", {}) or {})
        dispatch_csv_raw = str(artifacts.get("dispatch_impact_csv", "")).strip()
        if not dispatch_csv_raw:
            continue
        dispatch_csv = Path(dispatch_csv_raw)
        if not dispatch_csv.exists():
            continue

        if system_case is None:
            system_case = str(payload.get("system_case", "")).strip() or None

        dispatch_df = pd.read_csv(dispatch_csv)
        dispatch_df.columns = [str(col).strip() for col in dispatch_df.columns]
        if "row_type" not in dispatch_df.columns:
            continue
        dispatch_df = dispatch_df.loc[dispatch_df["row_type"].astype(str) == "ibr_summary"].copy()
        if dispatch_df.empty:
            continue

        scenario_cfg = dict(payload.get("scenario", {}) or {})
        owner_key, owner_label = _target_owner_labels(scenario_cfg)
        dispatch_df["run_id"] = str(payload.get("run_id", suite_row.get("run_id", "")))
        dispatch_df["formulation_id"] = str(payload.get("formulation_id", suite_row.get("formulation_id", "")))
        dispatch_df["formulation_name"] = str(payload.get("formulation_name", suite_row.get("formulation_name", "")))
        dispatch_df["scenario_id"] = str(payload.get("scenario_id", suite_row.get("scenario_id", "")))
        dispatch_df["scenario_name"] = str(payload.get("scenario_name", suite_row.get("scenario_name", "")))
        dispatch_df["scenario_target_key"] = owner_key
        dispatch_df["scenario_target_label"] = owner_label
        dispatch_df["step_scale"] = scenario_cfg.get("step_scale")
        dispatch_df["load_step_time"] = scenario_cfg.get("load_step_time")
        dispatch_df["status"] = str(payload.get("status", suite_row.get("status", "")))
        dispatch_df["objective_dispatch_only"] = payload.get("objective_dispatch_only", suite_row.get("objective_dispatch_only"))
        rows.append(dispatch_df)

    return (pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()), system_case


def _unit_delta_summary(unit_df: pd.DataFrame, *, global_scenario_id: str) -> pd.DataFrame:
    if unit_df.empty:
        return unit_df.copy()

    numeric_cols = ["M_opt", "D_opt", "delta_p_dispatch", "delta_p_predicted", "headroom_up", "headroom_down"]
    for col in numeric_cols:
        unit_df[col] = pd.to_numeric(unit_df[col], errors="coerce")

    baseline = (
        unit_df.loc[unit_df["scenario_id"].astype(str) == str(global_scenario_id), ["formulation_id", "index", "M_opt", "D_opt"]]
        .rename(columns={"M_opt": "M_opt_global", "D_opt": "D_opt_global"})
        .copy()
    )
    out = unit_df.merge(baseline, on=["formulation_id", "index"], how="left")
    out["M_opt_vs_global"] = out["M_opt"] - out["M_opt_global"]
    out["D_opt_vs_global"] = out["D_opt"] - out["D_opt_global"]
    return out


def _area_summary(unit_df: pd.DataFrame, *, global_scenario_id: str) -> pd.DataFrame:
    if unit_df.empty:
        return unit_df.copy()

    grouped = (
        unit_df.groupby(
            ["formulation_id", "formulation_name", "scenario_id", "scenario_name", "scenario_target_key", "scenario_target_label", "area_name"],
            dropna=False,
        )
        .agg(
            n_ibr=("index", "count"),
            M_sum=("M_opt", "sum"),
            M_mean=("M_opt", "mean"),
            D_sum=("D_opt", "sum"),
            D_mean=("D_opt", "mean"),
        )
        .reset_index()
    )

    baseline = (
        grouped.loc[grouped["scenario_id"].astype(str) == str(global_scenario_id), ["formulation_id", "area_name", "M_sum", "M_mean", "D_sum", "D_mean"]]
        .rename(
            columns={
                "M_sum": "M_sum_global",
                "M_mean": "M_mean_global",
                "D_sum": "D_sum_global",
                "D_mean": "D_mean_global",
            }
        )
        .copy()
    )
    out = grouped.merge(baseline, on=["formulation_id", "area_name"], how="left")
    out["M_sum_vs_global"] = out["M_sum"] - out["M_sum_global"]
    out["D_sum_vs_global"] = out["D_sum"] - out["D_sum_global"]
    out["M_mean_vs_global"] = out["M_mean"] - out["M_mean_global"]
    out["D_mean_vs_global"] = out["D_mean"] - out["D_mean_global"]
    return out


def _plot_unit_delta_heatmaps(unit_df: pd.DataFrame, *, global_scenario_id: str, stem: str) -> list[str]:
    if unit_df.empty:
        return []

    focus = unit_df.loc[unit_df["scenario_id"].astype(str) != str(global_scenario_id)].copy()
    if focus.empty:
        return []

    set_thesis_style()
    formulations = focus["formulation_id"].dropna().astype(str).unique().tolist()
    formulations = sorted(formulations)
    n_rows = len(formulations)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10.5, max(3.2, 3.2 * n_rows)), squeeze=False)

    for row_idx, formulation_id in enumerate(formulations):
        sub = focus.loc[focus["formulation_id"].astype(str) == formulation_id].copy()
        owner_order = sorted(sub["scenario_target_label"].dropna().astype(str).unique().tolist())
        unit_order = sorted(sub["regcv1_name"].dropna().astype(str).unique().tolist())
        for col_idx, metric in enumerate(["M_opt_vs_global", "D_opt_vs_global"]):
            ax = axes[row_idx, col_idx]
            pivot = (
                sub.pivot_table(index="scenario_target_label", columns="regcv1_name", values=metric, aggfunc="mean")
                .reindex(index=owner_order, columns=unit_order)
            )
            values = np.asarray(pivot.to_numpy(dtype=float), dtype=float)
            vmax = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
            vmax = max(vmax, 1.0e-9)
            im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(len(unit_order)))
            ax.set_xticklabels(unit_order, rotation=20, ha="right")
            ax.set_yticks(np.arange(len(owner_order)))
            ax.set_yticklabels(owner_order)
            ax.set_xlabel("IBR unit")
            if col_idx == 0:
                ax.set_ylabel(f"{formulation_id}\nTargeted zone")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"$\Delta M$ [p.u.]" if metric == "M_opt_vs_global" else r"$\Delta D$ [p.u.]")

    paths = save_figure(fig, stem, FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def _plot_area_totals(area_df: pd.DataFrame, *, global_scenario_id: str, stem: str) -> list[str]:
    if area_df.empty:
        return []

    focus = area_df.loc[
        (area_df["scenario_id"].astype(str) != str(global_scenario_id))
        & (area_df["formulation_id"].astype(str) == "retained_vis")
    ].copy()
    if focus.empty:
        focus = area_df.loc[area_df["scenario_id"].astype(str) != str(global_scenario_id)].copy()
    if focus.empty:
        return []

    set_thesis_style()
    owner_order = sorted(focus["scenario_target_label"].dropna().astype(str).unique().tolist())
    area_order = sorted(focus["area_name"].dropna().astype(str).unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, value_col, ylabel in [
        (axes[0], "M_sum", "Scheduled inertia sum [p.u.]"),
        (axes[1], "D_sum", "Scheduled damping sum [p.u.]"),
    ]:
        x = np.arange(len(owner_order), dtype=float)
        width = 0.8 / max(len(area_order), 1)
        for idx, area_name in enumerate(area_order):
            subset = (
                focus.loc[focus["area_name"].astype(str) == area_name, ["scenario_target_label", value_col]]
                .groupby("scenario_target_label", as_index=True)
                .mean()
                .reindex(owner_order)
            )
            values = pd.to_numeric(subset[value_col], errors="coerce").fillna(0.0).to_numpy()
            offset = (idx - (len(area_order) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, label=area_name, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(owner_order, rotation=20, ha="right")
        ax.set_xlabel("Targeted zone")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    paths = save_figure(fig, stem, FIGURES_DIR)
    plt.close(fig)
    return [str(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze VIS allocation sensitivity to zone-based load mismatches.")
    parser.add_argument(
        "--summary-json",
        default="results/thesis_optimization_results/results/zone_mismatch_vis_sensitivity_summary.json",
        help="Suite summary JSON written by scheduling.run_experiment_suite.",
    )
    parser.add_argument(
        "--global-scenario-id",
        default="global_uniform",
        help="Scenario id used as the global-load baseline for delta columns.",
    )
    parser.add_argument(
        "--case",
        default="",
        help="Optional explicit system case path. If omitted, the first run's system_case is used.",
    )
    parser.add_argument(
        "--stem",
        default="zone_mismatch_vis_sensitivity",
        help="Output stem used for tables and figures.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    summary_path = Path(args.summary_json)
    if not summary_path.is_absolute():
        summary_path = (ROOT / summary_path).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Suite summary JSON not found: {summary_path}")

    unit_df, system_case = _load_ibr_rows(summary_path)
    if unit_df.empty:
        raise RuntimeError(f"No IBR dispatch rows found in suite summary: {summary_path}")

    raw_case = str(args.case).strip() or str(system_case or "").strip()
    if not raw_case:
        raise ValueError("Could not resolve system case path from --case or suite summary.")
    case_path = Path(raw_case)
    if not case_path.is_absolute():
        case_path = (ROOT / case_path).resolve()
    if not case_path.exists():
        raise FileNotFoundError(f"System case not found: {case_path}")

    area_map = _load_area_map(case_path)
    unit_df = unit_df.merge(area_map, on="index", how="left")
    unit_df["area_name"] = unit_df["area_name"].fillna("UNKNOWN")
    unit_df = _unit_delta_summary(unit_df, global_scenario_id=str(args.global_scenario_id))
    area_df = _area_summary(unit_df.copy(), global_scenario_id=str(args.global_scenario_id))

    scenario_df = (
        unit_df.groupby(
            ["formulation_id", "formulation_name", "scenario_id", "scenario_name", "scenario_target_key", "scenario_target_label"],
            dropna=False,
        )
        .agg(
            n_ibr=("index", "count"),
            M_mean=("M_opt", "mean"),
            D_mean=("D_opt", "mean"),
            M_mean_vs_global=("M_opt_vs_global", "mean"),
            D_mean_vs_global=("D_opt_vs_global", "mean"),
        )
        .reset_index()
    )

    stem = str(args.stem)
    outputs = {
        "unit_table": _write_table_bundle(f"{stem}_unit_allocations", unit_df),
        "scenario_table": _write_table_bundle(f"{stem}_by_scenario", scenario_df),
        "area_table": _write_table_bundle(f"{stem}_by_area", area_df),
        "figures": [],
    }
    outputs["figures"].extend(
        _plot_unit_delta_heatmaps(unit_df, global_scenario_id=str(args.global_scenario_id), stem=f"{stem}_unit_deltas")
    )
    outputs["figures"].extend(
        _plot_area_totals(area_df, global_scenario_id=str(args.global_scenario_id), stem=f"{stem}_area_totals")
    )

    summary_out = TABLES_DIR / f"{stem}_summary.json"
    with summary_out.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print(f"[zone_mismatch_vis_analysis] unit_rows={len(unit_df)} area_rows={len(area_df)}")
    print(f"[zone_mismatch_vis_analysis] summary={summary_out}")


if __name__ == "__main__":
    main()
