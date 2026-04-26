from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from .benchmark_utils import DEFAULT_SYSTEM_BASE_MVA, benchmark_roots, load_benchmark_config, write_csv_and_parquet
    from .config_analysis import (
        ANALYSIS_CONFIG,
        BASE_OPTIMIZATION_CONFIG,
        BENCHMARK_CONFIG,
        BENCHMARK_CROSS_RESULTS_DIR,
        BENCHMARK_MAIN_RESULTS_DIR,
        BENCHMARK_REPLAY_RESULTS_DIR,
        FORMULATION_SUITE_CONFIG,
        PLOT_DATA_DIR,
        FORMULATION_SUMMARY_JSON,
        REPLAY_DETAIL_CSV,
        REPLAY_SUMMARY_CSV,
        TABLES_DIR,
        OptimizationResultsNotReadyError,
    )
    from .validation_utils import (
        add_line_security_flags,
        add_replay_feasibility_flags,
        replay_breakdown_by_method,
        replay_breakdown_by_metric,
    )
except ImportError:
    from benchmark_utils import DEFAULT_SYSTEM_BASE_MVA, benchmark_roots, load_benchmark_config, write_csv_and_parquet  # type: ignore
    from config_analysis import (  # type: ignore
        ANALYSIS_CONFIG,
        BASE_OPTIMIZATION_CONFIG,
        BENCHMARK_CONFIG,
        BENCHMARK_CROSS_RESULTS_DIR,
        BENCHMARK_MAIN_RESULTS_DIR,
        BENCHMARK_REPLAY_RESULTS_DIR,
        FORMULATION_SUITE_CONFIG,
        PLOT_DATA_DIR,
        FORMULATION_SUMMARY_JSON,
        REPLAY_DETAIL_CSV,
        REPLAY_SUMMARY_CSV,
        TABLES_DIR,
        OptimizationResultsNotReadyError,
    )
    from validation_utils import (  # type: ignore
        add_line_security_flags,
        add_replay_feasibility_flags,
        replay_breakdown_by_method,
        replay_breakdown_by_metric,
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_analysis_config(path: Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else ANALYSIS_CONFIG
    if not cfg_path.exists():
        return {}
    cfg = load_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(num):
        return ""
    if abs(num) >= 100 or abs(num) == int(abs(num)):
        return f"{num:.0f}"
    if abs(num) >= 10:
        return f"{num:.2f}"
    return f"{num:.3f}"


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("No rows.\n", encoding="utf-8")
        return

    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_float(row.get(col)) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        path.write_text("% No rows.\n", encoding="utf-8")
        return
    latex = df.to_latex(index=False, escape=True, na_rep="", float_format=lambda value: _format_float(value))
    path.write_text(latex, encoding="utf-8")


def require_file(path: Path, *, label: str, rerun_hint: str) -> Path:
    if not path.exists():
        raise OptimizationResultsNotReadyError(
            f"{label} is not available yet.\n"
            f"Expected: {path}\n"
            f"Rerun: {rerun_hint}"
        )
    return path


def formulation_order(analysis_cfg: Dict[str, Any] | None = None) -> list[str]:
    cfg = analysis_cfg or {}
    ordered = list(cfg.get("formulation_order") or [])
    if ordered:
        return [str(value) for value in ordered]
    suite_cfg = load_yaml(FORMULATION_SUITE_CONFIG)
    return [str(run.get("id")) for run in list(suite_cfg.get("runs") or []) if run.get("id")]


def baseline_formulation_id(analysis_cfg: Dict[str, Any] | None = None) -> str:
    cfg = analysis_cfg or {}
    if cfg.get("baseline_formulation_id"):
        return str(cfg["baseline_formulation_id"])
    suite_cfg = load_yaml(FORMULATION_SUITE_CONFIG)
    return str(suite_cfg.get("baseline_id", "ed"))


def retained_formulation_id(analysis_cfg: Dict[str, Any] | None = None) -> str:
    cfg = analysis_cfg or {}
    return str(cfg.get("retained_formulation_id", "ed_line_n1_surrogate"))


def build_formulation_catalog() -> pd.DataFrame:
    suite_cfg = load_yaml(FORMULATION_SUITE_CONFIG)
    base_cfg = load_yaml(BASE_OPTIMIZATION_CONFIG)
    rows = []
    for run in list(suite_cfg.get("runs") or []):
        formulation_id = str(run.get("id", "")).strip()
        if not formulation_id:
            continue
        cfg = dict(base_cfg)
        if run.get("overrides"):
            cfg = _deep_merge(cfg, dict(run.get("overrides") or {}))
        switches = dict(cfg.get("constraints", {}) or {})
        rows.append(
            {
                "formulation_id": formulation_id,
                "formulation_name": str(run.get("name", formulation_id)),
                "description": str(run.get("description", "")),
                "uses_ed": int(bool(switches.get("use_ed", True))),
                "uses_line": int(bool(switches.get("use_line", False))),
                "uses_n1": int(bool(switches.get("use_n1", False))),
                "uses_surrogate": int(bool(switches.get("use_nn", False))),
                "uses_redispatch": int(bool(switches.get("use_n1_redispatch", False))),
                "embedding_mode": str(switches.get("nn_mode", "disabled")) if switches.get("use_nn", False) else "disabled",
                "model_type": str(cfg.get("model", {}).get("type", "")),
            }
        )
    df = pd.DataFrame(rows)
    order = formulation_order()
    if not df.empty and order:
        df["formulation_id"] = pd.Categorical(df["formulation_id"], categories=order, ordered=True)
        df = df.sort_values("formulation_id").reset_index(drop=True)
        df["formulation_id"] = df["formulation_id"].astype(str)
    return df


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def load_run_catalog() -> pd.DataFrame:
    summary_path = require_file(
        FORMULATION_SUMMARY_JSON,
        label="Formulation-comparison summary",
        rerun_hint="results/thesis_optimization_results/scripts/run_formulations.sh",
    )
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for row in list(payload.get("rows") or []):
        summary_json = str(row.get("summary_json", "")).strip()
        summary_payload: Dict[str, Any] = {}
        if summary_json and Path(summary_json).exists():
            with Path(summary_json).open("r", encoding="utf-8") as f:
                summary_payload = json.load(f)

        switches = dict(summary_payload.get("constraint_switches", {}) or {})
        solver_stats = dict(summary_payload.get("solver_stats", {}) or {})
        artifacts = dict(summary_payload.get("artifacts", {}) or {})
        security_checks = dict(summary_payload.get("security_checks", {}) or {})
        line_security = dict(security_checks.get("line_security", {}) or {})
        scenario = dict(summary_payload.get("scenario", {}) or {})

        rows.append(
            {
                "run_id": str(summary_payload.get("run_id", row.get("run_id", ""))),
                "formulation_id": str(summary_payload.get("formulation_id", row.get("formulation_id", row.get("run_id", "")))),
                "formulation_name": str(summary_payload.get("formulation_name", row.get("formulation_name", ""))),
                "scenario_id": str(summary_payload.get("scenario_id", row.get("scenario_id", ""))),
                "scenario_name": str(summary_payload.get("scenario_name", row.get("scenario_name", ""))),
                "base_scale": scenario.get("base_scale"),
                "step_scale": scenario.get("step_scale"),
                "load_step_time": scenario.get("load_step_time"),
                "status": str(summary_payload.get("status", row.get("status", ""))),
                "is_feasible": int(str(summary_payload.get("status", row.get("status", ""))).startswith("optimal")),
                "objective": summary_payload.get("objective", row.get("objective")),
                "objective_total": summary_payload.get("objective", row.get("objective")),
                "objective_dispatch_only": summary_payload.get("objective_dispatch_only", row.get("objective_dispatch_only")),
                "objective_reserve_only": summary_payload.get("objective_reserve_only", row.get("objective_reserve_only")),
                "objective_reserve_up_only": summary_payload.get("objective_reserve_up_only", row.get("objective_reserve_up_only")),
                "objective_reserve_postcont_only": summary_payload.get(
                    "objective_reserve_postcont_only", row.get("objective_reserve_postcont_only")
                ),
                "cost_increase_pct_vs_ed": row.get("cost_increase_pct_vs_ed"),
                "solve_time_sec": solver_stats.get("solve_time_sec", row.get("solve_time_sec")),
                "solver_name": solver_stats.get("solver_name", row.get("solver_name")),
                "solver_modeling_mode": summary_payload.get("solver_modeling_mode", row.get("solver_modeling_mode")),
                "model_type": summary_payload.get("model_type", row.get("model_type")),
                "nn_mode": summary_payload.get("nn_mode", row.get("nn_mode")),
                "use_line": int(bool(switches.get("use_line"))) if switches else row.get("use_line"),
                "use_n1": int(bool(switches.get("use_n1"))) if switches else row.get("use_n1"),
                "use_n1_redispatch": int(bool(switches.get("use_n1_redispatch"))) if switches else row.get("use_n1_redispatch"),
                "use_nn": int(bool(switches.get("use_nn"))) if switches else row.get("use_nn"),
                "base_max_loading_pct": line_security.get("base_max_loading_pct"),
                "base_n_violations": line_security.get("base_n_violations"),
                "base_max_violation_pct": line_security.get("base_max_violation_pct"),
                "n1_max_loading_pct": line_security.get("n1_max_loading_pct"),
                "n1_total_line_violations": line_security.get("n1_total_line_violations"),
                "n1_max_violation_pct": line_security.get("n1_max_violation_pct"),
                "predicted_outputs_within_limits": security_checks.get("predicted_outputs_within_limits"),
                "predicted_output_violation_count": security_checks.get("predicted_output_violation_count"),
                "summary_json": summary_json,
                "results_csv": str(artifacts.get("results_csv", row.get("results_csv", ""))),
                "dispatch_impact_csv": str(artifacts.get("dispatch_impact_csv", "")),
                "predicted_metrics_csv": str(artifacts.get("predicted_metrics_csv", "")),
                "constraint_blocks_csv": str(artifacts.get("constraint_blocks_csv", "")),
                "config_path": str(summary_payload.get("config_path", row.get("config_path", ""))),
            }
        )

    df = pd.DataFrame(rows)
    order = formulation_order()
    if not df.empty and order:
        df["formulation_id"] = pd.Categorical(df["formulation_id"], categories=order, ordered=True)
        df = df.sort_values(["scenario_id", "formulation_id"]).reset_index(drop=True)
        df["formulation_id"] = df["formulation_id"].astype(str)
    return add_line_security_flags(df)


def cost_summary_table(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df.copy()

    if "objective_total" not in run_df.columns and "objective" in run_df.columns:
        run_df = run_df.copy()
        run_df["objective_total"] = run_df["objective"]

    cols = [
        "scenario_id",
        "scenario_name",
        "formulation_id",
        "formulation_name",
        "status",
        "is_feasible",
        "objective_total",
        "objective_dispatch_only",
        "objective_reserve_only",
        "objective_reserve_up_only",
        "objective_reserve_postcont_only",
        "cost_increase_pct_vs_ed",
        "solve_time_sec",
        "solver_modeling_mode",
        "base_max_loading_pct",
        "n1_max_loading_pct",
        "predicted_output_violation_count",
    ]
    out = run_df[cols].copy()
    out = out.rename(
        columns={
            "objective_total": "total_cost",
            "objective_dispatch_only": "dispatch_cost_component",
            "objective_reserve_only": "reserve_cost_component",
            "objective_reserve_up_only": "reserve_up_cost_component",
            "objective_reserve_postcont_only": "reserve_postcont_cost_component",
        }
    )
    out["cost_increase_pct_vs_ed"] = np.nan
    for scenario_id, sub_idx in out.groupby("scenario_id", dropna=False).groups.items():
        sub = out.loc[sub_idx]
        base = pd.to_numeric(
            sub.loc[sub["formulation_id"].astype(str) == str(baseline_formulation_id()), "total_cost"],
            errors="coerce",
        ).dropna()
        if base.empty:
            continue
        base_val = float(base.iloc[0])
        if not np.isfinite(base_val) or abs(base_val) < 1.0e-12:
            continue
        vals = pd.to_numeric(out.loc[sub_idx, "total_cost"], errors="coerce")
        out.loc[sub_idx, "cost_increase_pct_vs_ed"] = 100.0 * (vals - base_val) / base_val
    return out


def load_dispatch_tables(run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    generator_rows: list[pd.DataFrame] = []
    ibr_rows: list[pd.DataFrame] = []
    for _, run in run_df.iterrows():
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
        for key in ("run_id", "formulation_id", "formulation_name", "scenario_id", "scenario_name"):
            df[key] = run.get(key)
        generator_rows.append(df.loc[df["row_type"] == "generator_dispatch"].copy())
        ibr_rows.append(df.loc[df["row_type"] == "ibr_summary"].copy())

    gen_df = pd.concat(generator_rows, ignore_index=True, sort=False) if generator_rows else pd.DataFrame()
    ibr_df = pd.concat(ibr_rows, ignore_index=True, sort=False) if ibr_rows else pd.DataFrame()
    return gen_df, ibr_df


def build_dispatch_comparison_tables(run_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gen_df, ibr_df = load_dispatch_tables(run_df)
    baseline_id = baseline_formulation_id()

    if not gen_df.empty:
        base_gen = (
            gen_df.loc[gen_df["formulation_id"] == baseline_id, ["scenario_id", "index", "pg_opt", "pg_delta"]]
            .rename(columns={"pg_opt": "baseline_pg_opt", "pg_delta": "baseline_pg_delta"})
            .copy()
        )
        gen_df = gen_df.merge(base_gen, on=["scenario_id", "index"], how="left")
        gen_df["pg_delta_vs_baseline"] = pd.to_numeric(gen_df["pg_opt"], errors="coerce") - pd.to_numeric(gen_df["baseline_pg_opt"], errors="coerce")

    if not ibr_df.empty:
        base_ibr = (
            ibr_df.loc[
                ibr_df["formulation_id"] == baseline_id,
                ["scenario_id", "index", "M_opt", "D_opt", "delta_p_dispatch", "headroom_up", "headroom_down"],
            ]
            .rename(
                columns={
                    "M_opt": "baseline_M_opt",
                    "D_opt": "baseline_D_opt",
                    "delta_p_dispatch": "baseline_delta_p_dispatch",
                    "headroom_up": "baseline_headroom_up",
                    "headroom_down": "baseline_headroom_down",
                }
            )
            .copy()
        )
        ibr_df = ibr_df.merge(base_ibr, on=["scenario_id", "index"], how="left")
        for col in ("M_opt", "D_opt", "delta_p_dispatch", "headroom_up", "headroom_down"):
            ibr_df[f"{col}_vs_baseline"] = pd.to_numeric(ibr_df[col], errors="coerce") - pd.to_numeric(
                ibr_df.get(f"baseline_{col}"), errors="coerce"
            )

    return gen_df, ibr_df


def load_replay_outputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail_path = require_file(
        REPLAY_DETAIL_CSV,
        label="Replay-validation detail table",
        rerun_hint="results/thesis_optimization_results/scripts/run_replay_validation.sh",
    )
    summary_path = require_file(
        REPLAY_SUMMARY_CSV,
        label="Replay-validation summary table",
        rerun_hint="results/thesis_optimization_results/scripts/run_replay_validation.sh",
    )
    detail_df = pd.read_csv(detail_path)
    summary_df = pd.read_csv(summary_path)
    return add_replay_feasibility_flags(detail_df), summary_df


def replay_metric_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return detail_df.copy()
    grouped = (
        detail_df.groupby(["formulation_id", "formulation_name", "metric_name"], dropna=False)
        .agg(
            n_rows=("metric_name", "count"),
            mean_abs_error=("abs_error", "mean"),
            rmse=("squared_error", lambda s: float(np.sqrt(np.nanmean(np.asarray(s, dtype=float))))),
            max_abs_error=("abs_error", "max"),
            false_safe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "false_safe"))),
            false_unsafe_count=("feasibility_case", lambda s: int(np.sum(pd.Series(s) == "false_unsafe"))),
            replay_safe_rate=("replayed_within_limits", lambda s: float(np.nanmean(np.asarray(s, dtype=float) >= 0.5))),
        )
        .reset_index()
    )
    order = formulation_order()
    if order:
        grouped["formulation_id"] = pd.Categorical(grouped["formulation_id"], categories=order, ordered=True)
        grouped = grouped.sort_values(["metric_name", "formulation_id"]).reset_index(drop=True)
        grouped["formulation_id"] = grouped["formulation_id"].astype(str)
    return grouped


def merge_dynamic_and_line_security(run_df: pd.DataFrame, replay_summary_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df.copy()
    if replay_summary_df.empty:
        return add_line_security_flags(run_df)

    cols = [
        "run_id",
        "n_predicted_within_limits",
        "n_replayed_within_limits",
        "n_false_safe",
        "n_false_unsafe",
        "replay_violation_count",
    ]
    replay_cols = [col for col in cols if col in replay_summary_df.columns]
    merged = run_df.merge(replay_summary_df[replay_cols], on="run_id", how="left")
    return add_line_security_flags(merged)


def iter_existing_paths(paths: Iterable[str]) -> list[Path]:
    out = []
    for raw in paths:
        path = Path(str(raw))
        if path.exists():
            out.append(path)
    return out


def _summary_row_from_payload(
    payload: Dict[str, Any],
    *,
    summary_path: Path,
    source_group: str,
) -> dict[str, Any]:
    solver_stats = dict(payload.get("solver_stats", {}) or {})
    problem_size = dict(payload.get("problem_size", {}) or {})
    security_checks = dict(payload.get("security_checks", {}) or {})
    line_security = dict(security_checks.get("line_security", {}) or {})
    switches = dict(payload.get("constraint_switches", {}) or {})
    artifacts = dict(payload.get("artifacts", {}) or {})
    scenario = dict(payload.get("scenario", {}) or {})
    return {
        "run_id": str(payload.get("run_id", "")),
        "formulation_id": str(payload.get("formulation_id", payload.get("run_id", ""))),
        "formulation_name": str(payload.get("formulation_name", payload.get("formulation_id", ""))),
        "scenario_id": str(payload.get("scenario_id", "")),
        "scenario_name": str(payload.get("scenario_name", payload.get("scenario_id", ""))),
        "source_group": source_group,
        "base_scale": scenario.get("base_scale"),
        "step_scale": scenario.get("step_scale"),
        "load_step_time": scenario.get("load_step_time"),
        "outage_id": scenario.get("contingency_line_uid"),
        "status": str(payload.get("status", "")),
        "solver_status": str(solver_stats.get("status", payload.get("status", ""))),
        "objective_total": payload.get("objective"),
        "objective_dispatch_only": payload.get("objective_dispatch_only"),
        "objective_reserve_only": payload.get("objective_reserve_only"),
        "objective_reserve_up_only": payload.get("objective_reserve_up_only"),
        "objective_reserve_postcont_only": payload.get("objective_reserve_postcont_only"),
        "solve_time_sec": solver_stats.get("solve_time_sec"),
        "binaries": problem_size.get("n_variables_binary"),
        "continuous_variables": problem_size.get("n_variables_continuous"),
        "constraints": problem_size.get("n_constraints_scalar_total", problem_size.get("n_constraints_total")),
        "nnz": problem_size.get("nnz_total"),
        "line_security_ok": int((float(line_security.get("base_n_violations", 0) or 0) <= 0)),
        "contingency_security_ok": (
            int((float(line_security.get("n1_total_line_violations", 0) or 0) <= 0))
            if bool(switches.get("use_n1", False))
            else 1
        ),
        "base_max_loading_pct": line_security.get("base_max_loading_pct"),
        "n1_max_loading_pct": line_security.get("n1_max_loading_pct"),
        "base_n_violations": line_security.get("base_n_violations"),
        "n1_total_line_violations": line_security.get("n1_total_line_violations"),
        "predicted_output_violation_count": security_checks.get("predicted_output_violation_count"),
        "predicted_outputs_within_limits": security_checks.get("predicted_outputs_within_limits"),
        "path_to_raw_summary_json": str(summary_path),
        "dispatch_impact_csv": str(artifacts.get("dispatch_impact_csv", "")),
        "predicted_metrics_csv": str(artifacts.get("predicted_metrics_csv", "")),
        "constraint_blocks_csv": str(artifacts.get("constraint_blocks_csv", "")),
        "config_path": str(payload.get("config_path", "")),
    }


def _load_summary_rows_from_paths(paths: Iterable[Path], *, source_group: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        rows.append(_summary_row_from_payload(payload, summary_path=path, source_group=source_group))
    df = pd.DataFrame(rows)
    if not df.empty:
        order = formulation_order()
        if order:
            df["formulation_id"] = pd.Categorical(df["formulation_id"], categories=order, ordered=True)
            df = df.sort_values(["scenario_id", "formulation_id", "run_id"]).reset_index(drop=True)
            df["formulation_id"] = df["formulation_id"].astype(str)
    return df


def _benchmark_summary_paths(benchmark_cfg: Dict[str, Any] | None = None) -> dict[str, list[Path]]:
    cfg = benchmark_cfg or load_benchmark_config(BENCHMARK_CONFIG)
    roots = benchmark_roots(cfg)
    out: dict[str, list[Path]] = {}
    for key in ("main", "cross_method_subset"):
        root = Path(roots[key])
        paths = sorted(
            [
                path
                for path in root.rglob("*_summary.json")
                if path.name != "benchmark_manifest_summary.json" and "_replay_" not in path.name
            ]
        )
        out[key] = paths
    return out


def _dispatch_impact_aggregates(path: Path, *, system_base_mva: float = DEFAULT_SYSTEM_BASE_MVA) -> dict[str, Any]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    if "row_type" not in df.columns:
        return {}
    ibr_df = df.loc[df["row_type"].astype(str) == "ibr_summary"].copy()
    if ibr_df.empty:
        return {}
    numeric_cols = [
        "reserve_up",
        "headroom_up",
        "headroom_down",
        "delta_p_predicted",
        "delta_p_dispatch",
        "reserve_up_cost_component",
        "reserve_postcont_cost_component",
        "reserve_total_cost_component",
        "M_opt",
        "D_opt",
    ]
    for col in numeric_cols:
        if col in ibr_df.columns:
            ibr_df[col] = pd.to_numeric(ibr_df[col], errors="coerce")

    def _sum(col: str) -> float:
        return float(ibr_df[col].fillna(0.0).sum()) if col in ibr_df.columns else np.nan

    def _mean(col: str) -> float:
        return float(ibr_df[col].mean()) if col in ibr_df.columns else np.nan

    headroom_up_pu = _sum("headroom_up")
    headroom_down_pu = _sum("headroom_down")
    reserve_up_pu = _sum("reserve_up")
    return {
        "reserve_up_total_pu": reserve_up_pu,
        "reserve_up_total_mw": reserve_up_pu * system_base_mva if np.isfinite(reserve_up_pu) else np.nan,
        "headroom_up_total_pu": headroom_up_pu,
        "headroom_up_total_mw": headroom_up_pu * system_base_mva if np.isfinite(headroom_up_pu) else np.nan,
        "headroom_down_total_pu": headroom_down_pu,
        "headroom_down_total_mw": headroom_down_pu * system_base_mva if np.isfinite(headroom_down_pu) else np.nan,
        "delta_p_predicted_total_pu": _sum("delta_p_predicted"),
        "delta_p_dispatch_total_pu": _sum("delta_p_dispatch"),
        "reserve_up_cost_component_total": _sum("reserve_up_cost_component"),
        "reserve_postcont_cost_component_total": _sum("reserve_postcont_cost_component"),
        "reserve_total_cost_component_total": _sum("reserve_total_cost_component"),
        "M_opt_mean": _mean("M_opt"),
        "M_opt_sum": _sum("M_opt"),
        "D_opt_mean": _mean("D_opt"),
        "D_opt_sum": _sum("D_opt"),
        "n_ibr_rows": int(ibr_df.shape[0]),
    }


def load_optimization_runs_master(benchmark_cfg_path: Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    try:
        legacy_df = load_run_catalog().copy()
        if not legacy_df.empty:
            legacy_df["source_group"] = "legacy_formulation_comparison"
            legacy_df = legacy_df.rename(columns={"status": "solver_status"})
            if "objective_total" not in legacy_df.columns and "objective" in legacy_df.columns:
                legacy_df = legacy_df.rename(columns={"objective": "objective_total"})
            frames.append(legacy_df)
    except OptimizationResultsNotReadyError:
        pass

    benchmark_cfg = load_benchmark_config(benchmark_cfg_path or BENCHMARK_CONFIG)
    summary_paths = _benchmark_summary_paths(benchmark_cfg)
    for group, paths in summary_paths.items():
        if paths:
            frames.append(_load_summary_rows_from_paths(paths, source_group=f"benchmark_{group}"))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    if "path_to_raw_summary_json" not in out.columns and "summary_json" in out.columns:
        out["path_to_raw_summary_json"] = out["summary_json"]
    if "solver_status" not in out.columns and "status" in out.columns:
        out["solver_status"] = out["status"]
    if "objective_total" not in out.columns and "objective" in out.columns:
        out["objective_total"] = out["objective"]

    dispatch_rows = []
    for _, row in out.iterrows():
        path_raw = str(row.get("dispatch_impact_csv", "")).strip()
        dispatch_rows.append(_dispatch_impact_aggregates(Path(path_raw)) if path_raw else {})
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(dispatch_rows)], axis=1)

    out["run_id"] = out["run_id"].fillna(out["formulation_id"].astype(str) + "__" + out["scenario_id"].astype(str))
    out["line_security_ok"] = pd.to_numeric(out.get("line_security_ok"), errors="coerce").fillna(0).astype(int)
    out["contingency_security_ok"] = pd.to_numeric(out.get("contingency_security_ok"), errors="coerce").fillna(0).astype(int)
    out["formulation"] = out["formulation_id"]
    keep_cols = [
        "run_id",
        "scenario_id",
        "scenario_name",
        "formulation",
        "formulation_id",
        "formulation_name",
        "source_group",
        "base_scale",
        "step_scale",
        "outage_id",
        "solver_status",
        "objective_total",
        "objective_dispatch_only",
        "objective_reserve_only",
        "objective_reserve_up_only",
        "objective_reserve_postcont_only",
        "solve_time_sec",
        "binaries",
        "continuous_variables",
        "constraints",
        "nnz",
        "line_security_ok",
        "contingency_security_ok",
        "reserve_up_total_pu",
        "reserve_up_total_mw",
        "headroom_up_total_pu",
        "headroom_up_total_mw",
        "headroom_down_total_pu",
        "headroom_down_total_mw",
        "M_opt_mean",
        "M_opt_sum",
        "D_opt_mean",
        "D_opt_sum",
        "reserve_up_cost_component_total",
        "reserve_postcont_cost_component_total",
        "reserve_total_cost_component_total",
        "base_max_loading_pct",
        "n1_max_loading_pct",
        "predicted_output_violation_count",
        "path_to_raw_summary_json",
        "dispatch_impact_csv",
        "config_path",
    ]
    keep_cols = [col for col in keep_cols if col in out.columns]
    return out[keep_cols].copy()


def load_replay_runs_master(benchmark_cfg_path: Path | None = None) -> pd.DataFrame:
    detail_frames: list[pd.DataFrame] = []
    try:
        legacy_detail, _ = load_replay_outputs()
        if not legacy_detail.empty:
            legacy_detail["path_to_raw_replay_detail"] = str(REPLAY_DETAIL_CSV)
            legacy_detail["source_group"] = "legacy_replay"
            detail_frames.append(legacy_detail)
    except OptimizationResultsNotReadyError:
        pass

    benchmark_cfg = load_benchmark_config(benchmark_cfg_path or BENCHMARK_CONFIG)
    roots = benchmark_roots(benchmark_cfg)
    for key in ("replay_main", "replay_cross_method_subset"):
        root = Path(roots[key])
        if not root.exists():
            continue
        for path in sorted(root.rglob("*_replay_detail.csv")):
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["path_to_raw_replay_detail"] = str(path)
            df["source_group"] = f"benchmark_{key}"
            detail_frames.append(df)

    if not detail_frames:
        return pd.DataFrame()

    df = pd.concat(detail_frames, ignore_index=True, sort=False)
    df = add_replay_feasibility_flags(df)
    feasibility = df["feasibility_case"] if "feasibility_case" in df.columns else pd.Series("", index=df.index, dtype=object)
    out = pd.DataFrame(
        {
            "replay_id": (
                df["run_id"].astype(str)
                + "__"
                + df["metric_name"].astype(str)
                + "__"
                + df["source_group"].astype(str)
            ),
            "run_id": df["run_id"],
            "scenario_id": df["scenario_id"],
            "formulation": df["formulation_id"],
            "formulation_name": df.get("formulation_name"),
            "metric_name": df["metric_name"],
            "metric_category": df.get("metric_category"),
            "predicted_value": pd.to_numeric(df.get("predicted_value"), errors="coerce"),
            "replayed_value": pd.to_numeric(df.get("replayed_value"), errors="coerce"),
            "prediction_error": pd.to_numeric(df.get("signed_error"), errors="coerce"),
            "abs_prediction_error": pd.to_numeric(df.get("abs_error"), errors="coerce"),
            "violated_in_replay": (pd.to_numeric(df.get("replayed_within_limits"), errors="coerce").fillna(-1) < 0.5).astype(int),
            "false_safe_flag": (feasibility.astype(str) == "false_safe").astype(int),
            "violation_magnitude": pd.to_numeric(df.get("replay_violation_magnitude"), errors="coerce"),
            "relevant_limit": pd.to_numeric(df.get("relevant_limit"), errors="coerce"),
            "scheduled_headroom_violation_flag": pd.to_numeric(df.get("scheduled_headroom_violation_flag"), errors="coerce"),
            "scheduled_headroom_violation_magnitude": pd.to_numeric(df.get("scheduled_headroom_violation_magnitude"), errors="coerce"),
            "physical_limit_violation_flag": pd.to_numeric(df.get("physical_limit_violation_flag"), errors="coerce"),
            "physical_limit_violation_magnitude": pd.to_numeric(df.get("physical_limit_violation_magnitude"), errors="coerce"),
            "scheduled_headroom_up": pd.to_numeric(df.get("scheduled_headroom_up"), errors="coerce"),
            "scheduled_headroom_down": pd.to_numeric(df.get("scheduled_headroom_down"), errors="coerce"),
            "physical_min": pd.to_numeric(df.get("physical_min"), errors="coerce"),
            "physical_max": pd.to_numeric(df.get("physical_max"), errors="coerce"),
            "system_base_mva": pd.to_numeric(df.get("system_base_mva"), errors="coerce"),
            "path_to_raw_replay_detail": df["path_to_raw_replay_detail"],
            "source_group": df["source_group"],
        }
    )
    return out


def save_plot_data(stem: str, df: pd.DataFrame) -> dict[str, str]:
    return write_csv_and_parquet(df, PLOT_DATA_DIR / f"{stem}.csv", PLOT_DATA_DIR / f"{stem}.parquet")


def method_cost_breakdown(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    optimal = run_df.loc[run_df["solver_status"].astype(str).str.startswith("optimal")].copy()
    if optimal.empty:
        return pd.DataFrame()
    return (
        optimal.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            dispatch_cost_mean=("objective_dispatch_only", "mean"),
            reserve_cost_mean=("objective_reserve_only", "mean"),
            reserve_up_cost_mean=("objective_reserve_up_only", "mean"),
            reserve_postcont_cost_mean=("objective_reserve_postcont_only", "mean"),
            total_cost_mean=("objective_total", "mean"),
        )
        .reset_index()
    )


def reserve_summary_by_method(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    optimal = run_df.loc[run_df["solver_status"].astype(str).str.startswith("optimal")].copy()
    if optimal.empty:
        return pd.DataFrame()
    return (
        optimal.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            reserve_up_total_pu_mean=("reserve_up_total_pu", "mean"),
            reserve_up_total_mw_mean=("reserve_up_total_mw", "mean"),
            headroom_up_total_pu_mean=("headroom_up_total_pu", "mean"),
            headroom_up_total_mw_mean=("headroom_up_total_mw", "mean"),
            headroom_down_total_pu_mean=("headroom_down_total_pu", "mean"),
            headroom_down_total_mw_mean=("headroom_down_total_mw", "mean"),
        )
        .reset_index()
    )


def solver_complexity_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    return (
        run_df.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            solve_time_sec_mean=("solve_time_sec", "mean"),
            solve_time_sec_max=("solve_time_sec", "max"),
            binaries_mean=("binaries", "mean"),
            continuous_variables_mean=("continuous_variables", "mean"),
            constraints_mean=("constraints", "mean"),
            nnz_mean=("nnz", "mean"),
        )
        .reset_index()
    )


def line_security_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    return (
        run_df.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            line_security_ok_rate=("line_security_ok", "mean"),
            contingency_security_ok_rate=("contingency_security_ok", "mean"),
            base_max_loading_pct_max=("base_max_loading_pct", "max"),
            n1_max_loading_pct_max=("n1_max_loading_pct", "max"),
            predicted_output_violation_count_sum=("predicted_output_violation_count", "sum"),
        )
        .reset_index()
    )


def vis_allocation_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    optimal = run_df.loc[run_df["solver_status"].astype(str).str.startswith("optimal")].copy()
    if optimal.empty:
        return pd.DataFrame()
    return (
        optimal.groupby(["formulation_id", "formulation_name"], dropna=False)
        .agg(
            n_runs=("run_id", "count"),
            M_opt_mean=("M_opt_mean", "mean"),
            M_opt_sum_mean=("M_opt_sum", "mean"),
            D_opt_mean=("D_opt_mean", "mean"),
            D_opt_sum_mean=("D_opt_sum", "mean"),
            headroom_up_total_mw_mean=("headroom_up_total_mw", "mean"),
            reserve_up_total_mw_mean=("reserve_up_total_mw", "mean"),
        )
        .reset_index()
    )


def scenario_coverage_summary(run_df: pd.DataFrame, replay_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    if manifest_df.empty:
        return pd.DataFrame()
    opt_counts = run_df.groupby("scenario_id", dropna=False)["run_id"].nunique().rename("optimization_runs").reset_index() if not run_df.empty else pd.DataFrame(columns=["scenario_id", "optimization_runs"])
    replay_counts = replay_df.groupby("scenario_id", dropna=False)["run_id"].nunique().rename("replay_runs").reset_index() if not replay_df.empty else pd.DataFrame(columns=["scenario_id", "replay_runs"])
    out = manifest_df.merge(opt_counts, on="scenario_id", how="left").merge(replay_counts, on="scenario_id", how="left")
    out["optimization_runs"] = pd.to_numeric(out["optimization_runs"], errors="coerce").fillna(0).astype(int)
    out["replay_runs"] = pd.to_numeric(out["replay_runs"], errors="coerce").fillna(0).astype(int)
    return (
        out.groupby(["scenario_family", "disturbance_type"], dropna=False)
        .agg(
            planned_scenarios=("scenario_id", "count"),
            optimization_runs=("optimization_runs", "sum"),
            replay_runs=("replay_runs", "sum"),
            cross_method_selected=("selected_for_cross_method_subset", "sum"),
        )
        .reset_index()
    )


def load_manifest_tables(benchmark_cfg_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_path = TABLES_DIR / "scenario_manifest.csv"
    subset_path = TABLES_DIR / "cross_method_subset_manifest.csv"
    if scenario_path.exists() and subset_path.exists():
        return pd.read_csv(scenario_path), pd.read_csv(subset_path)
    benchmark_cfg = load_benchmark_config(benchmark_cfg_path or BENCHMARK_CONFIG)
    try:
        from .benchmark_utils import generate_scenario_manifest, save_manifest_bundle, load_formulation_suite  # type: ignore
    except ImportError:
        from benchmark_utils import generate_scenario_manifest, save_manifest_bundle, load_formulation_suite  # type: ignore

    suite_cfg = load_formulation_suite(FORMULATION_SUITE_CONFIG)
    manifest_df, subset_df, severity_df = generate_scenario_manifest(benchmark_cfg)
    save_manifest_bundle(
        manifest_df=manifest_df,
        subset_df=subset_df,
        severity_df=severity_df,
        benchmark_cfg=benchmark_cfg,
        suite_cfg=suite_cfg,
    )
    return manifest_df, subset_df
