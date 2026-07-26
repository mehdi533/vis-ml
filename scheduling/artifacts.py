"""Artifact I/O for the scheduler: JSON/CSV writers and output-path resolution.

Extracted verbatim from problem.py to shrink that module (see README refactor note).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _save_results_csv(
    path: Path,
    *,
    status: str,
    objective: float | None,
    pg: np.ndarray,
    m: np.ndarray,
    d: np.ndarray,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["status", "objective"]
        + [f"pg_{i + 1}" for i in range(pg.size)]
        + [f"M_{i + 1}" for i in range(m.size)]
        + [f"D_{i + 1}" for i in range(d.size)]
    )
    row = [status, "" if objective is None else float(objective)] + pg.tolist() + m.tolist() + d.tolist()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

def _default_output_paths(output_cfg: Mapping[str, Any], formulation_id: str) -> dict[str, Path]:
    results_csv = Path(str(output_cfg.get("results_csv", "results/thesis_optimization_results/results/optimization_results.csv")))
    run_tag = str(output_cfg.get("run_tag", formulation_id)).strip() or formulation_id
    results_dir = Path(str(output_cfg.get("results_dir", results_csv.parent)))
    results_dir.mkdir(parents=True, exist_ok=True)

    def _p(name: str, default: Path) -> Path:
        raw = output_cfg.get(name)
        return Path(str(raw)) if raw else default

    return {
        "results_csv": _p("results_csv", results_dir / f"{run_tag}_decisions.csv"),
        "summary_json": _p("summary_json", results_dir / f"{run_tag}_summary.json"),
        "summary_csv": _p("summary_csv", results_dir / f"{run_tag}_summary.csv"),
        "predicted_metrics_csv": _p("predicted_metrics_csv", results_dir / f"{run_tag}_predicted_metrics.csv"),
        "dispatch_impact_csv": _p("dispatch_impact_csv", results_dir / f"{run_tag}_dispatch_impact.csv"),
        "constraint_blocks_csv": _p("constraint_blocks_csv", results_dir / f"{run_tag}_constraint_blocks.csv"),
    }

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, ensure_ascii=False)

def _write_dict_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
                headers.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _summary_row_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    solver_stats = payload.get("solver_stats", {})
    problem_size = payload.get("problem_size", {})
    return {
        "run_id": payload.get("run_id"),
        "formulation_id": payload.get("formulation_id"),
        "formulation_name": payload.get("formulation_name"),
        "scenario_id": payload.get("scenario_id"),
        "scenario_name": payload.get("scenario_name"),
        "status": payload.get("status"),
        "objective": payload.get("objective"),
        "objective_dispatch_only": payload.get("objective_dispatch_only"),
        "objective_reserve_only": payload.get("objective_reserve_only"),
        "objective_reserve_up_only": payload.get("objective_reserve_up_only"),
        "solve_time_sec": solver_stats.get("solve_time_sec"),
        "num_iters": solver_stats.get("num_iters"),
        "n_variables_total": problem_size.get("n_variables_total"),
        "n_variables_binary": problem_size.get("n_variables_binary"),
        "n_constraints_total": problem_size.get("n_constraints_total"),
        "n_constraints_scalar_total": problem_size.get("n_constraints_scalar_total"),
        "nnz_total": problem_size.get("nnz_total"),
        "system_case": payload.get("system_case"),
        "model_type": payload.get("model_type"),
        "nn_mode": payload.get("nn_mode"),
        "solver_modeling_mode": payload.get("solver_modeling_mode"),
    }
