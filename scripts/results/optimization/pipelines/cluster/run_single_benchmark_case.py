#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[4]
PYTHON_BIN = str(ROOT / ".." / "venv" / "bin" / "python")
BENCHMARK_MANIFESTS_DIR = ROOT / "results" / "thesis_optimization_results" / "results" / "benchmark" / "manifests"
TABLES_DIR = ROOT / "results" / "thesis_optimization_results" / "outputs" / "tables"
REPLAY_CONFIG = ROOT / "results" / "thesis_optimization_results" / "configs" / "replay" / "replay_validation.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _resolve(path_like: str | Path, base: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base / path).resolve()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _task_manifest_path(mode: str, group: str) -> Path:
    if mode == "optimize" and group == "main":
        return BENCHMARK_MANIFESTS_DIR / "main_benchmark_tasks.csv"
    if mode == "optimize" and group == "cross_method_subset":
        return BENCHMARK_MANIFESTS_DIR / "cross_method_benchmark_tasks.csv"
    if mode == "replay" and group == "main":
        return BENCHMARK_MANIFESTS_DIR / "replay_main_benchmark_tasks.csv"
    if mode == "replay" and group == "cross_method_subset":
        return BENCHMARK_MANIFESTS_DIR / "replay_cross_method_benchmark_tasks.csv"
    raise ValueError(f"Unsupported mode/group combination: mode={mode}, group={group}")


def _scenario_manifest_path(group: str) -> Path:
    if group == "main":
        return TABLES_DIR / "scenario_manifest.csv"
    if group == "cross_method_subset":
        return TABLES_DIR / "cross_method_subset_manifest.csv"
    raise ValueError(f"Unsupported benchmark group: {group}")


def _select_row(rows: list[dict[str, str]], *, task_index: int | None, scenario_id: str | None, formulation_id: str | None) -> dict[str, str]:
    matches = []
    for row in rows:
        if task_index is not None and int(row.get("task_index", "-1")) != int(task_index):
            continue
        if scenario_id is not None and str(row.get("scenario_id", "")) != str(scenario_id):
            continue
        if formulation_id is not None and str(row.get("formulation_id", "")) != str(formulation_id):
            continue
        matches.append(row)
    if not matches:
        raise ValueError("No task matched the requested selector.")
    if len(matches) > 1:
        raise ValueError("Task selector is ambiguous; provide --task-index or both --scenario-id and --formulation-id.")
    return matches[0]


def _execv(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["KMP_USE_SHM"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["MKL_THREADING_LAYER"] = "GNU"
    env["KMP_AFFINITY"] = "disabled"
    env["KMP_INIT_AT_FORK"] = "FALSE"
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    os.chdir(cwd)
    os.execvpe(cmd[0], cmd, env)


def _scenario_overrides(row: dict[str, str]) -> dict[str, Any]:
    disturbance_type = str(row.get("disturbance_type", "load_step"))
    scenario_cfg: dict[str, Any] = {
        "id": str(row["scenario_id"]),
        "name": str(row["scenario_id"]),
        "description": str(row.get("notes", "")),
        "base_scale": float(row["base_scale"]),
    }
    features_cfg: dict[str, Any] = {}
    if disturbance_type == "load_step":
        scenario_cfg["step_scale"] = float(row["step_scale"])
        zone_owner = str(row.get("zone_owner", "")).strip()
        if zone_owner and zone_owner.lower() != "nan":
            scenario_cfg["load_step_target_owners"] = [zone_owner]
        features_cfg["contingency_mode"] = "load_mismatch"
    elif disturbance_type == "line_trip":
        scenario_cfg["step_scale"] = 1.0
        scenario_cfg["contingency_line_uid"] = int(float(row["outage_id"]))
        features_cfg["contingency_mode"] = "line"
        features_cfg["contingency_line_uid"] = int(float(row["outage_id"]))
    else:
        raise ValueError(f"Unsupported disturbance_type: {disturbance_type}")
    return {"scenario": scenario_cfg, "features": features_cfg}


def _build_optimization_config(
    *,
    formulation_suite: Path,
    scenario_row: dict[str, str],
    task_row: dict[str, str],
) -> tuple[dict[str, Any], Path]:
    suite_cfg = _load_yaml(formulation_suite)
    suite_dir = formulation_suite.resolve().parent
    run_map = {str(run.get("id", "")).strip(): dict(run) for run in list(suite_cfg.get("runs") or []) if str(run.get("id", "")).strip()}
    formulation_id = str(task_row["formulation_id"])
    if formulation_id not in run_map:
        raise KeyError(f"Unknown formulation_id in suite: {formulation_id}")
    run_cfg = run_map[formulation_id]
    base_cfg_path = _resolve(str(run_cfg.get("base_config", "../base_optimization.yaml")), suite_dir)
    base_cfg = _load_yaml(base_cfg_path)
    cfg = _deep_merge(base_cfg, _scenario_overrides(scenario_row))
    cfg = _deep_merge(cfg, dict(run_cfg.get("overrides") or {}))

    cfg.setdefault("formulation", {})
    cfg["formulation"]["id"] = formulation_id
    cfg["formulation"]["name"] = str(run_cfg.get("name", formulation_id))
    cfg["formulation"]["description"] = str(run_cfg.get("description", ""))

    run_dir = Path(task_row["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("output", {})
    cfg["output"]["run_tag"] = formulation_id
    cfg["output"]["results_dir"] = str(run_dir)
    cfg["output"]["log_file"] = str(Path(task_row["log_file"]).resolve())
    resolved_cfg_path = Path(task_row["resolved_config"])
    return cfg, resolved_cfg_path


def _build_replay_config(task_row: dict[str, str]) -> tuple[dict[str, Any], Path]:
    base_replay_cfg = _load_yaml(REPLAY_CONFIG)
    replay_dir = Path(task_row["replay_dir"])
    replay_dir.mkdir(parents=True, exist_ok=True)
    opt_cfg = _load_yaml(Path(task_row["resolved_config"]))
    scenario_cfg = dict(opt_cfg.get("scenario", {}) or {})
    contingency_type = str(task_row.get("contingency_type", "load_step"))
    contingency: dict[str, Any] = {"type": contingency_type}
    if contingency_type == "line_trip":
        contingency["time"] = float(scenario_cfg.get("load_step_time", 1.0))
        contingency["line_uid"] = int(float(task_row["outage_id"]))

    cfg = dict(base_replay_cfg)
    cfg["contingency"] = contingency
    cfg["runs"] = [
        {
            "summary_json": str(task_row["summary_json"]),
            "optimization_config": str(task_row["resolved_config"]),
            "run_id": str(task_row["task_id"]),
        }
    ]
    cfg["output"] = {
        "summary_csv": str(task_row["replay_summary_csv"]),
        "detail_csv": str(task_row["replay_detail_csv"]),
        "summary_json": str(task_row["replay_summary_json"]),
    }
    replay_cfg_path = replay_dir / f"{task_row['formulation_id']}_replay_config.yaml"
    return cfg, replay_cfg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one resumable thesis benchmark optimization or replay case.")
    parser.add_argument("--benchmark-config", default="configs/scheduling/thesis_optimization_benchmark.yaml")
    parser.add_argument("--formulation-suite", default="configs/scheduling/suites/01_formulation_comparison.yaml")
    parser.add_argument("--mode", choices=["optimize", "replay"], default="optimize")
    parser.add_argument("--group", choices=["main", "cross_method_subset"], default="main")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--formulation-id", default=None)
    parser.add_argument("--count-tasks", action="store_true")
    parser.add_argument("--show-task", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    task_rows = _load_csv_rows(_task_manifest_path(args.mode, args.group))
    if args.count_tasks:
        print(len(task_rows))
        return

    task_row = _select_row(
        task_rows,
        task_index=args.task_index,
        scenario_id=args.scenario_id,
        formulation_id=args.formulation_id,
    )
    if args.show_task:
        print(json.dumps(task_row, indent=2))
        return

    if args.mode == "optimize":
        scenario_rows = _load_csv_rows(_scenario_manifest_path(args.group))
        scenario_row = _select_row(
            scenario_rows,
            task_index=None,
            scenario_id=task_row["scenario_id"],
            formulation_id=None,
        )
        cfg, resolved_cfg_path = _build_optimization_config(
            formulation_suite=Path(args.formulation_suite),
            scenario_row=scenario_row,
            task_row=task_row,
        )
        summary_json = Path(task_row["summary_json"])
        if summary_json.exists() and not args.force:
            print(f"[benchmark_case] skip existing summary_json={summary_json}")
            return
        if args.dry_run:
            print(json.dumps({"resolved_config": str(resolved_cfg_path), "summary_json": str(summary_json)}, indent=2))
            return
        _write_yaml(resolved_cfg_path, cfg)
        if args.prepare_only:
            print(str(resolved_cfg_path))
            return
        cmd = [PYTHON_BIN, "scheduling/problem.py", "--config", str(resolved_cfg_path)]
        print(f"[benchmark_case] optimize task_id={task_row['task_id']}")
        _execv(cmd, cwd=ROOT)
        return

    replay_summary_json = Path(task_row["replay_summary_json"])
    replay_summary_csv = Path(task_row["replay_summary_csv"])
    replay_detail_csv = Path(task_row["replay_detail_csv"])
    if replay_summary_json.exists() and replay_summary_csv.exists() and replay_detail_csv.exists() and not args.force:
        print(f"[benchmark_case] skip existing replay summary_json={replay_summary_json}")
        return
    replay_cfg, replay_cfg_path = _build_replay_config(task_row)
    if args.dry_run:
        print(json.dumps({"replay_config": str(replay_cfg_path), "replay_summary_json": str(replay_summary_json)}, indent=2))
        return
    _write_yaml(replay_cfg_path, replay_cfg)
    if args.prepare_only:
        print(str(replay_cfg_path))
        return
    cmd = [PYTHON_BIN, "scheduling/replay_validation.py", "--config", str(replay_cfg_path)]
    print(f"[benchmark_case] replay task_id={task_row['task_id']}")
    _execv(cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
