from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve(path_like: str, base: Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base / p).resolve()


def _sanitize_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return safe or "default"


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _collapse_ws(text: str, *, max_len: int = 400) -> str:
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def _read_text_tail(path: Path | None, *, max_lines: int = 80, max_chars: int = 6000) -> str:
    if path is None or not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def _classify_issue(status: str, error_text: str) -> tuple[str, str, str]:
    status_l = str(status or "").strip().lower()
    msg = str(error_text or "")
    msg_l = msg.lower()

    if status_l.startswith("optimal"):
        return ("ok", "none", "No action needed.")

    if "non-finite values detected in resolved optimization feature vector" in msg_l:
        return (
            "feature_nonfinite",
            "feature_builder",
            "Inspect unresolved/non-finite x features in x_seed. Check division features like P_REGCV1_SHARE and denominator assumptions.",
        )
    if "p_regcv1_share is in the feature contract but constraints.use_ed=false" in msg_l:
        return (
            "config_switch_mismatch",
            "constraints.use_ed",
            "Enable ED for this feature contract or remove P_REGCV1_SHARE from the model input contract.",
        )
    if "scaler/model feature mismatch" in msg_l:
        return (
            "scaler_feature_mismatch",
            "model_artifacts",
            "Ensure x_features ordering/length matches x_scaler and model run_config feature contract.",
        )
    if "model feature contract is not fully covered" in msg_l:
        return (
            "feature_contract_not_buildable",
            "feature_registry",
            "Compare model feature contract vs registry-derived buildable features. Add missing extractors or change retained surrogate.",
        )
    if "missing schedulable feature in x_features" in msg_l:
        return (
            "missing_sched_feature",
            "x_features_contract",
            "Ensure every REGCV1 has M_i and D_i in x_features.",
        )
    if "missing dispatch-linked features in x_features" in msg_l:
        return (
            "missing_dispatch_feature",
            "x_features_contract",
            "Ensure all P_GENROU_i/P_REGCV1_i dispatch channels required by pg linkage are present.",
        )
    if "some scheduling features are not decision-linked" in msg_l:
        return (
            "sched_not_linked",
            "input_constraints",
            "Wire all x_sched features as free or derived constraints; do not leave sched channels fixed.",
        )
    if "strict mode" in msg_l or "extract from system only" in msg_l:
        return (
            "strict_policy_violation",
            "features_policy",
            "Disable fallback feature injection options or relax strict_from_system if intentionally needed.",
        )
    if "invalid branch limits" in msg_l:
        return (
            "network_limits_invalid",
            "line_constraint_data",
            "Check line RATE_A/Sn fields and sanitize invalid or sentinel limits in the case data.",
        )
    if "module not found" in msg_l or "modulenotfounderror" in msg_l:
        return (
            "environment_dependency",
            "python_environment",
            "Install missing Python dependencies in the selected virtual environment.",
        )
    if status_l in {"infeasible", "infeasible_inaccurate", "unbounded", "infeasible_or_unbounded"}:
        return (
            "nonoptimal_solver_status",
            "optimization_model",
            "Inspect active constraint blocks, bounds, and feasibility-check logs for the conflicting block.",
        )
    return (
        "unknown",
        "undetermined",
        "Inspect diagnostics_json, resolved_config, and run log tail for the first failing stage.",
    )


def _preflight_notes(cfg: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    model_cfg = dict(cfg.get("model", {}) or {})
    scalers_cfg = dict(cfg.get("scalers", {}) or {})
    feature_cfg = dict(cfg.get("features", {}) or {})
    scenario_cfg = dict(cfg.get("scenario", {}) or {})

    model_raw = str(model_cfg.get("model_dir", model_cfg.get("state_dict", ""))).strip()
    model_dir: Path | None = None
    if not model_raw:
        notes.append("model.model_dir/state_dict is empty")
    else:
        model_dir = _resolve(model_raw, ROOT)
        if model_dir.suffix:
            model_dir = model_dir.parent
        if not model_dir.exists():
            notes.append(f"model_dir_missing:{model_dir}")
        else:
            run_cfg = model_dir / "run_config.yaml"
            if not run_cfg.exists():
                notes.append(f"model_run_config_missing:{run_cfg}")

    x_scaler_raw = str(scalers_cfg.get("x_scaler_path", "")).strip()
    y_scaler_raw = str(scalers_cfg.get("y_scaler_path", "")).strip()
    if x_scaler_raw:
        x_scaler = _resolve(x_scaler_raw, ROOT)
    elif model_dir is not None:
        x_scaler = model_dir / "x_scaler.pkl"
    else:
        x_scaler = None
    if y_scaler_raw:
        y_scaler = _resolve(y_scaler_raw, ROOT)
    elif model_dir is not None:
        y_scaler = model_dir / "y_scaler.pkl"
    else:
        y_scaler = None
    if x_scaler is None or not x_scaler.exists():
        notes.append(f"x_scaler_missing:{x_scaler}")
    if y_scaler is None or not y_scaler.exists():
        notes.append(f"y_scaler_missing:{y_scaler}")

    if bool(feature_cfg.get("x_features_from_registry", False)):
        registry_raw = str(feature_cfg.get("feature_names_path", "configs/data_generation_feature_names.yaml")).strip()
        registry_path = _resolve(registry_raw, ROOT)
        if not registry_path.exists():
            notes.append(f"feature_registry_missing:{registry_path}")

    step_scale = scenario_cfg.get("step_scale")
    try:
        if step_scale is not None and abs(float(step_scale)) <= 1e-12:
            notes.append("scenario.step_scale_is_zero")
    except Exception:
        notes.append(f"scenario.step_scale_invalid:{step_scale}")

    strict = bool(feature_cfg.get("strict_from_system", True))
    if strict:
        if str(feature_cfg.get("missing_feature_source", "none")).strip().lower() not in {"", "none"}:
            notes.append("strict_mode_with_missing_feature_source")
        if bool(feature_cfg.get("allow_missing_features", False)):
            notes.append("strict_mode_with_allow_missing_features")
        if bool(feature_cfg.get("fixed_source_override_non_sched", False)):
            notes.append("strict_mode_with_fixed_source_override_non_sched")

    return notes


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _scenario_id_from_cfg(cfg: dict[str, Any], fallback: str = "default") -> str:
    scenario_cfg = dict(cfg.get("scenario", {}) or {})
    raw = str(scenario_cfg.get("id", "")).strip()
    if raw:
        return _sanitize_token(raw)

    base_scale = scenario_cfg.get("base_scale")
    step_scale = scenario_cfg.get("step_scale")
    load_step_time = scenario_cfg.get("load_step_time")
    if base_scale is None and step_scale is None and load_step_time is None:
        return _sanitize_token(fallback)

    parts = []
    if base_scale is not None:
        parts.append(f"b{float(base_scale):.3f}".replace(".", "p"))
    if step_scale is not None:
        parts.append(f"s{float(step_scale):.3f}".replace(".", "p"))
    if load_step_time is not None:
        parts.append(f"t{float(load_step_time):.3f}".replace(".", "p"))
    return _sanitize_token("_".join(parts) or fallback)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return

    headers: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                headers.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _suite_scenarios(suite: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    raw = list(suite.get("scenarios") or [])
    if not raw:
        return ([{"id": "", "name": "", "description": "", "overrides": {}}], False)

    scenarios: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        item = dict(entry or {})
        overrides = dict(item.get("overrides") or {})
        scenario_id = str(item.get("id", "")).strip()
        if not scenario_id:
            scenario_id = _scenario_id_from_cfg(overrides, fallback=f"scenario_{idx + 1}")
        scenarios.append(
            {
                "id": _sanitize_token(scenario_id),
                "name": str(item.get("name", scenario_id)),
                "description": str(item.get("description", "")),
                "overrides": overrides,
            }
        )
    return scenarios, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a suite of optimization formulations and aggregate KPIs.")
    parser.add_argument("--suite", required=True, help="Path to suite YAML.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one run fails. Default is continue.",
    )
    parser.add_argument(
        "--log-tail-lines",
        type=int,
        default=80,
        help="Number of trailing log lines to include in diagnostics.",
    )
    args = parser.parse_args()

    suite_path = Path(args.suite).resolve()
    suite = _load_yaml(suite_path)
    try:
        from scheduling.problem import run_optimization
    except ModuleNotFoundError:
        from final_optimization_folder.problem import run_optimization

    suite_dir = suite_path.parent
    runs = list(suite.get("runs") or [])
    if not runs:
        raise ValueError(f"No runs configured in suite file: {suite_path}")
    scenarios, has_explicit_scenarios = _suite_scenarios(suite)

    baseline_id = str(suite.get("baseline_id", "ed"))
    results_root_raw = suite.get("results_root")
    results_root = _resolve(str(results_root_raw), suite_dir) if results_root_raw else None
    output_cfg = suite.get("output", {}) or {}
    summary_csv = _resolve(str(output_cfg.get("summary_csv", "results/suite_summary.csv")), suite_dir)
    summary_md = _resolve(str(output_cfg.get("summary_markdown", "results/suite_summary.md")), suite_dir)
    summary_json = _resolve(str(output_cfg.get("summary_json", "results/suite_summary.json")), suite_dir)
    failures_csv = summary_csv.with_name(summary_csv.stem + "_failures.csv")
    failures_md = summary_md.with_name(summary_md.stem + "_failures.md")
    diagnostics_json = summary_json.with_name(summary_json.stem + "_diagnostics.json")

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_overrides = dict(scenario.get("overrides") or {})
        scenario_id = str(scenario.get("id", "")).strip()
        scenario_name = str(scenario.get("name", "")).strip()
        scenario_description = str(scenario.get("description", "")).strip()

        for run in runs:
            formulation_id = str(run.get("id", "")).strip()
            if not formulation_id:
                raise ValueError("Each run entry must define a non-empty 'id'.")

            config_path: Path | None = None
            resolved_cfg_path: Path | None = None
            run_dir: Path | None = None
            log_path: Path | None = None
            diagnostics_path: Path | None = None
            run_key = formulation_id
            failure_stage = "init"
            preflight = []

            row: dict[str, Any] = {
                "run_id": "",
                "formulation_id": formulation_id,
                "formulation_name": str(run.get("name", formulation_id)),
                "paper_method": str(run.get("paper_method", "")),
                "comparison_family": str(run.get("comparison_family", "")),
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "config_path": "",
                "resolved_config": "",
                "run_dir": "",
                "log_file": "",
                "summary_json": "",
                "summary_csv_path": "",
                "results_csv": "",
                "constraint_blocks_csv": "",
                "predicted_metrics_csv": "",
                "dispatch_impact_csv": "",
                "diagnostics_json": "",
                "preflight_ok": None,
                "preflight_notes": "",
                "failure_stage": "",
                "error_type": "",
                "error_category": "",
                "suspected_source": "",
                "suggested_next_step": "",
                "log_tail": "",
                "error": "",
            }
            try:
                failure_stage = "load_config"
                if "config" in run:
                    config_path = _resolve(str(run["config"]), suite_dir)
                    cfg = _load_yaml(config_path)
                    if scenario_overrides:
                        cfg = _deep_merge(cfg, scenario_overrides)
                    if run.get("overrides"):
                        cfg = _deep_merge(cfg, dict(run.get("overrides") or {}))
                elif "base_config" in run:
                    base_path = _resolve(str(run["base_config"]), suite_dir)
                    base_cfg = _load_yaml(base_path)
                    cfg = _deep_merge(base_cfg, scenario_overrides)
                    overrides = dict(run.get("overrides") or {})
                    cfg = _deep_merge(cfg, overrides)
                    config_path = base_path
                else:
                    raise ValueError(f"Run '{formulation_id}' must define either 'config' or 'base_config'.")
                row["config_path"] = str(config_path)

                failure_stage = "inject_formulation_metadata"
                cfg.setdefault("formulation", {})
                cfg["formulation"]["id"] = formulation_id
                if run.get("name"):
                    cfg["formulation"]["name"] = str(run["name"])
                if run.get("description"):
                    cfg["formulation"]["description"] = str(run["description"])
                if run.get("equation_map"):
                    cfg["formulation"]["equation_map"] = run["equation_map"]

                failure_stage = "inject_scenario_metadata"
                cfg.setdefault("scenario", {})
                if scenario_id:
                    cfg["scenario"]["id"] = scenario_id
                if scenario_name:
                    cfg["scenario"]["name"] = scenario_name
                if scenario_description:
                    cfg["scenario"]["description"] = scenario_description
                row["model_dir"] = str(cfg.get("model", {}).get("model_dir", cfg.get("model", {}).get("state_dict", "")))

                resolved_scenario_id = _scenario_id_from_cfg(cfg)
                row["scenario_id"] = resolved_scenario_id
                row["scenario_name"] = str(cfg.get("scenario", {}).get("name", scenario_name or resolved_scenario_id))

                run_key = formulation_id if not has_explicit_scenarios else f"{formulation_id}__{resolved_scenario_id}"
                row["run_id"] = run_key

                failure_stage = "prepare_output_paths"
                cfg.setdefault("output", {})
                cfg["output"]["run_tag"] = run_key
                if results_root is not None:
                    if has_explicit_scenarios:
                        run_dir = (results_root / resolved_scenario_id / formulation_id).resolve()
                    else:
                        run_dir = (results_root / formulation_id).resolve()
                    cfg["output"]["results_dir"] = str(run_dir)
                    cfg["output"]["log_file"] = str((run_dir / f"{run_key}.log").resolve())
                    resolved_cfg_path = (run_dir / f"{run_key}_resolved_config.yaml").resolve()
                    resolved_cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    with resolved_cfg_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)
                    config_path = resolved_cfg_path
                    row["config_path"] = str(config_path)
                    row["resolved_config"] = str(resolved_cfg_path)

                if run_dir is None:
                    raw_results_dir = str(cfg.get("output", {}).get("results_dir", "")).strip()
                    if raw_results_dir:
                        run_dir = _resolve(raw_results_dir, config_path.parent if config_path is not None else suite_dir)
                log_raw = str(cfg.get("output", {}).get("log_file", "")).strip()
                if log_raw:
                    log_path = _resolve(log_raw, config_path.parent if config_path is not None else suite_dir)
                elif run_dir is not None:
                    log_path = (run_dir / f"{run_key}.log").resolve()
                if run_dir is None and log_path is not None:
                    run_dir = log_path.parent

                row["run_dir"] = str(run_dir) if run_dir is not None else ""
                row["log_file"] = str(log_path) if log_path is not None else ""

                failure_stage = "preflight"
                preflight = _preflight_notes(cfg)
                row["preflight_ok"] = int(len(preflight) == 0)
                row["preflight_notes"] = "; ".join(preflight)

                failure_stage = "run_optimization"
                res = run_optimization(cfg, config_path=str(config_path))

                failure_stage = "load_summary_payload"
                summary_path = Path(res["summary_json"])
                with summary_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)

                solver_stats = payload.get("solver_stats") or {}
                problem_size = payload.get("problem_size") or {}
                switches = payload.get("constraint_switches") or {}
                artifacts = payload.get("artifacts") or {}
                active_nn_mode = payload.get("nn_mode") if bool(switches.get("use_nn")) else "disabled"

                row.update(
                    {
                        "run_id": str(payload.get("run_id", run_key)),
                        "formulation_id": str(payload.get("formulation_id", formulation_id)),
                        "formulation_name": payload.get("formulation_name", ""),
                        "scenario_id": str(payload.get("scenario_id", resolved_scenario_id)),
                        "scenario_name": str(payload.get("scenario_name", row["scenario_name"])),
                        "status": payload.get("status", ""),
                        "objective": payload.get("objective"),
                        "objective_dispatch_only": payload.get("objective_dispatch_only"),
                        "solve_time_sec": solver_stats.get("solve_time_sec"),
                        "solver_name": solver_stats.get("solver_name"),
                        "nn_mode": active_nn_mode,
                        "model_type": payload.get("model_type"),
                        "model_dir": payload.get("model_dir"),
                        "use_line": int(bool(switches.get("use_line"))),
                        "use_n1": int(bool(switches.get("use_n1"))),
                        "use_n1_redispatch": int(bool(switches.get("use_n1_redispatch"))),
                        "use_nn": int(bool(switches.get("use_nn"))),
                        "n_variables_total": problem_size.get("n_variables_total"),
                        "n_variables_binary": problem_size.get("n_variables_binary"),
                        "n_constraints_total": problem_size.get("n_constraints_total"),
                        "n_constraints_scalar_total": problem_size.get("n_constraints_scalar_total"),
                        "summary_json": str(summary_path),
                        "summary_csv_path": str(artifacts.get("summary_csv", "")),
                        "results_csv": str(res.get("results_csv", "")),
                        "constraint_blocks_csv": str(artifacts.get("constraint_blocks_csv", "")),
                        "predicted_metrics_csv": str(artifacts.get("predicted_metrics_csv", "")),
                        "dispatch_impact_csv": str(artifacts.get("dispatch_impact_csv", "")),
                        "wall_runtime_sec": payload.get("wall_runtime_sec"),
                    }
                )
                row["solver_modeling_mode"] = (
                    f"{row.get('solver_name', '')}:{active_nn_mode}"
                    if row.get("solver_name")
                    else str(active_nn_mode)
                )

                status_text = str(row.get("status", ""))
                category, source, hint = _classify_issue(status_text, "")
                row["error_category"] = category
                row["suspected_source"] = source
                row["suggested_next_step"] = hint
                if not status_text.startswith("optimal"):
                    row["failure_stage"] = "solve_status"
                    log_tail = _read_text_tail(log_path, max_lines=max(1, int(args.log_tail_lines)))
                    row["log_tail"] = _collapse_ws(log_tail, max_len=800)
                    diag_dir = run_dir or summary_json.parent
                    diagnostics_path = (diag_dir / f"{run_key}_diagnostics.json").resolve()
                    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                    with diagnostics_path.open("w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "run_id": row.get("run_id"),
                                "status": status_text,
                                "failure_stage": row.get("failure_stage"),
                                "error_category": category,
                                "suspected_source": source,
                                "suggested_next_step": hint,
                                "preflight_notes": preflight,
                                "config_path": row.get("config_path"),
                                "resolved_config": row.get("resolved_config"),
                                "log_file": row.get("log_file"),
                                "summary_json": row.get("summary_json"),
                                "log_tail": log_tail,
                            },
                            f,
                            indent=2,
                        )
                    row["diagnostics_json"] = str(diagnostics_path)

            except Exception as exc:
                error_text = str(exc)
                error_type = type(exc).__name__
                trace_text = traceback.format_exc()
                category, source, hint = _classify_issue("error", error_text)

                if log_path is None and run_dir is not None:
                    candidate_log = (run_dir / f"{run_key}.log").resolve()
                    if candidate_log.exists():
                        log_path = candidate_log
                log_tail = _read_text_tail(log_path, max_lines=max(1, int(args.log_tail_lines)))

                row.update(
                    {
                        "status": "error",
                        "objective": None,
                        "objective_dispatch_only": None,
                        "solve_time_sec": None,
                        "solver_name": None,
                        "nn_mode": None,
                        "model_type": None,
                        "use_line": None,
                        "use_n1": None,
                        "use_n1_redispatch": None,
                        "use_nn": None,
                        "n_variables_total": None,
                        "n_variables_binary": None,
                        "n_constraints_total": None,
                        "n_constraints_scalar_total": None,
                        "summary_json": "",
                        "summary_csv_path": "",
                        "results_csv": "",
                        "constraint_blocks_csv": "",
                        "predicted_metrics_csv": "",
                        "dispatch_impact_csv": "",
                        "solver_modeling_mode": "",
                        "failure_stage": failure_stage,
                        "error_type": error_type,
                        "error_category": category,
                        "suspected_source": source,
                        "suggested_next_step": hint,
                        "error": error_text,
                        "log_tail": _collapse_ws(log_tail, max_len=800),
                    }
                )
                if row.get("preflight_ok") is None:
                    row["preflight_ok"] = int(len(preflight) == 0)
                if not row.get("preflight_notes"):
                    row["preflight_notes"] = "; ".join(preflight)
                row["log_file"] = str(log_path) if log_path is not None else str(row.get("log_file", ""))
                row["run_dir"] = str(run_dir) if run_dir is not None else str(row.get("run_dir", ""))
                if resolved_cfg_path is not None and not row.get("resolved_config"):
                    row["resolved_config"] = str(resolved_cfg_path)

                diag_dir = run_dir or (config_path.parent if config_path is not None else summary_json.parent)
                diagnostics_path = (diag_dir / f"{run_key}_diagnostics.json").resolve()
                diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
                with diagnostics_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "run_id": row.get("run_id") or run_key,
                            "formulation_id": formulation_id,
                            "scenario_id": row.get("scenario_id", scenario_id),
                            "status": "error",
                            "failure_stage": failure_stage,
                            "error_type": error_type,
                            "error": error_text,
                            "error_category": category,
                            "suspected_source": source,
                            "suggested_next_step": hint,
                            "traceback": trace_text,
                            "preflight_notes": preflight,
                            "config_path": row.get("config_path"),
                            "resolved_config": row.get("resolved_config"),
                            "log_file": row.get("log_file"),
                            "log_tail": log_tail,
                        },
                        f,
                        indent=2,
                    )
                row["diagnostics_json"] = str(diagnostics_path)

                rows.append(row)
                if args.stop_on_error:
                    raise
                continue

            rows.append(row)

    baseline_obj_by_scenario: dict[str, float] = {}
    for row in rows:
        if str(row.get("formulation_id")) != baseline_id:
            continue
        if not str(row.get("status", "")).startswith("optimal"):
            continue
        scenario_key = str(row.get("scenario_id", ""))
        baseline_obj_by_scenario[scenario_key] = _as_float(row.get("objective_dispatch_only"))

    for row in rows:
        scenario_key = str(row.get("scenario_id", ""))
        baseline_obj = baseline_obj_by_scenario.get(scenario_key, float("nan"))
        obj = _as_float(row.get("objective_dispatch_only"))
        row["baseline_formulation_id"] = baseline_id
        row["baseline_objective_dispatch_only"] = baseline_obj if baseline_obj == baseline_obj else None
        if baseline_obj == baseline_obj and obj == obj and abs(baseline_obj) > 1e-12:
            row["cost_increase_pct_vs_ed"] = 100.0 * (obj - baseline_obj) / baseline_obj
        else:
            row["cost_increase_pct_vs_ed"] = float("nan")

    for row in rows:
        if str(row.get("error_category", "")).strip():
            continue
        category, source, hint = _classify_issue(str(row.get("status", "")), str(row.get("error", "")))
        row["error_category"] = category
        row["suspected_source"] = source
        row["suggested_next_step"] = hint

    failure_rows = [row for row in rows if not str(row.get("status", "")).startswith("optimal")]
    failure_rows.sort(key=lambda r: (str(r.get("scenario_id", "")), str(r.get("formulation_id", ""))))

    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for row in failure_rows:
        category = str(row.get("error_category", "unknown")) or "unknown"
        source = str(row.get("suspected_source", "undetermined")) or "undetermined"
        category_counts[category] = int(category_counts.get(category, 0) + 1)
        source_counts[source] = int(source_counts.get(source, 0) + 1)

    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, rows)
    _write_csv(failures_csv, failure_rows)
    _write_markdown(failures_md, failure_rows)

    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": str(suite.get("name", "")),
                "suite_path": str(suite_path),
                "n_rows_total": int(len(rows)),
                "n_rows_failed": int(len(failure_rows)),
                "failure_counts_by_category": category_counts,
                "failure_counts_by_suspected_source": source_counts,
                "failure_rows": failure_rows,
            },
            f,
            indent=2,
        )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "suite": str(suite.get("name", "")),
                "suite_path": str(suite_path),
                "baseline_id": baseline_id,
                "baseline_objective_dispatch_only_by_scenario": {
                    key: value for key, value in baseline_obj_by_scenario.items() if value == value
                },
                "explicit_scenarios": has_explicit_scenarios,
                "failures_csv": str(failures_csv),
                "failures_markdown": str(failures_md),
                "diagnostics_json": str(diagnostics_json),
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"[run_experiment_suite] Wrote: {summary_csv}")
    print(f"[run_experiment_suite] Wrote: {summary_md}")
    print(f"[run_experiment_suite] Wrote: {failures_csv}")
    print(f"[run_experiment_suite] Wrote: {failures_md}")
    print(f"[run_experiment_suite] Wrote: {diagnostics_json}")
    print(f"[run_experiment_suite] Wrote: {summary_json}")


if __name__ == "__main__":
    main()
