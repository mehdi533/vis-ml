from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from .config_analysis import (
        ANALYSIS_CONFIG,
        BASE_OPTIMIZATION_CONFIG,
        FORMULATION_SUITE_CONFIG,
        FORMULATION_SUMMARY_JSON,
        REPLAY_DETAIL_CSV,
        REPLAY_SUMMARY_CSV,
        OptimizationResultsNotReadyError,
    )
    from .validation_utils import add_line_security_flags, add_replay_feasibility_flags
except ImportError:
    from config_analysis import (  # type: ignore
        ANALYSIS_CONFIG,
        BASE_OPTIMIZATION_CONFIG,
        FORMULATION_SUITE_CONFIG,
        FORMULATION_SUMMARY_JSON,
        REPLAY_DETAIL_CSV,
        REPLAY_SUMMARY_CSV,
        OptimizationResultsNotReadyError,
    )
    from validation_utils import add_line_security_flags, add_replay_feasibility_flags  # type: ignore


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
                "objective_dispatch_only": summary_payload.get("objective_dispatch_only", row.get("objective_dispatch_only")),
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

    cols = [
        "scenario_id",
        "scenario_name",
        "formulation_id",
        "formulation_name",
        "status",
        "is_feasible",
        "objective_dispatch_only",
        "cost_increase_pct_vs_ed",
        "solve_time_sec",
        "solver_modeling_mode",
        "base_max_loading_pct",
        "n1_max_loading_pct",
        "predicted_output_violation_count",
    ]
    out = run_df[cols].copy()
    out = out.rename(columns={"objective_dispatch_only": "dispatch_cost"})
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
