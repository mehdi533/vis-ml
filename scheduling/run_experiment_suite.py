from __future__ import annotations

import argparse
import csv
import json
import re
import sys
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

            row: dict[str, Any] = {
                "run_id": "",
                "formulation_id": formulation_id,
                "formulation_name": str(run.get("name", formulation_id)),
                "paper_method": str(run.get("paper_method", "")),
                "comparison_family": str(run.get("comparison_family", "")),
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "config_path": "",
            }
            try:
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

                cfg.setdefault("formulation", {})
                cfg["formulation"]["id"] = formulation_id
                if run.get("name"):
                    cfg["formulation"]["name"] = str(run["name"])
                if run.get("description"):
                    cfg["formulation"]["description"] = str(run["description"])
                if run.get("equation_map"):
                    cfg["formulation"]["equation_map"] = run["equation_map"]

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

                res = run_optimization(cfg, config_path=str(config_path))
                summary_path = Path(res["summary_json"])
                with summary_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)

                solver_stats = payload.get("solver_stats") or {}
                problem_size = payload.get("problem_size") or {}
                switches = payload.get("constraint_switches") or {}
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
                        "results_csv": str(res.get("results_csv", "")),
                    }
                )
                row["solver_modeling_mode"] = (
                    f"{row.get('solver_name', '')}:{active_nn_mode}"
                    if row.get("solver_name")
                    else str(active_nn_mode)
                )
            except Exception as exc:
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
                        "results_csv": "",
                        "solver_modeling_mode": "",
                        "error": str(exc),
                    }
                )
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

    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, rows)
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
                "rows": rows,
            },
            f,
            indent=2,
        )

    print(f"[run_experiment_suite] Wrote: {summary_csv}")
    print(f"[run_experiment_suite] Wrote: {summary_md}")
    print(f"[run_experiment_suite] Wrote: {summary_json}")


if __name__ == "__main__":
    main()
