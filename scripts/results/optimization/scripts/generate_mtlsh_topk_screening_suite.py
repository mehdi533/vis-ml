#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from results.thesis_optimization_results.src.benchmark_utils import (  # noqa: E402
    formulation_run_map,
    generate_scenario_manifest,
    load_benchmark_config,
    load_formulation_suite,
    resolve_path,
    select_screened_outages,
)
from results.thesis_optimization_results.src.config_analysis import (  # noqa: E402
    GENERATED_CONFIGS_DIR,
    RESULTS_DIR,
)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(dict(out[key]), value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _relpath(path: Path, start: Path) -> str:
    return str(path.resolve().relative_to(start.resolve())) if path.resolve().is_relative_to(start.resolve()) else str(path.resolve())


def _parse_top_k_values(spec: str, *, n_available: int) -> list[str | int]:
    values: list[str | int] = []
    seen: set[str | int] = set()
    for raw in str(spec).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        value: str | int
        if token == "all":
            value = "all"
        else:
            value = int(token)
            if value <= 0:
                raise ValueError(f"top-k value must be positive: {raw}")
            if value > n_available:
                raise ValueError(
                    f"top-k value {value} exceeds the number of screened outages ({n_available})."
                )
        if value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        raise ValueError("No top-k values were provided.")
    return values


def _scenario_family_filter(name: str) -> set[str]:
    key = str(name).strip().lower()
    mapping = {
        "global": {"global_load_mismatch"},
        "zone": {"zone_load_mismatch"},
        "line": {"line_outage"},
        "all": {"global_load_mismatch", "zone_load_mismatch", "line_outage"},
    }
    if key not in mapping:
        raise ValueError(f"Unsupported scenario family '{name}'. Choose from global, zone, line, all.")
    return mapping[key]


def _scenario_entry(row: pd.Series) -> dict[str, Any]:
    scenario_id = str(row["scenario_id"])
    disturbance_type = str(row["disturbance_type"])
    scenario_cfg: dict[str, Any] = {
        "id": scenario_id,
        "name": scenario_id,
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
        outage_id = int(float(row["outage_id"]))
        scenario_cfg["step_scale"] = 1.0
        scenario_cfg["contingency_line_uid"] = outage_id
        features_cfg["contingency_mode"] = "line"
        features_cfg["contingency_line_uid"] = outage_id
    else:
        raise ValueError(f"Unsupported disturbance_type '{disturbance_type}' for scenario {scenario_id}.")
    return {
        "id": scenario_id,
        "name": scenario_id,
        "description": str(row.get("notes", "")),
        "overrides": {"scenario": scenario_cfg, "features": features_cfg},
    }


def _label_for_top_k(value: str | int, *, n_available: int) -> tuple[str, str]:
    if value == "all":
        return (f"all{n_available:02d}", f"all {n_available}")
    top_k = int(value)
    return (f"top{top_k:02d}", f"top {top_k}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an MTLSH top-k screened contingency suite for run_experiment_suite.py."
    )
    parser.add_argument(
        "--benchmark-config",
        default="configs/scheduling/thesis_optimization_benchmark.yaml",
        help="Benchmark config used to source ranked screened outages and benchmark scenarios.",
    )
    parser.add_argument(
        "--formulation-suite",
        default="configs/scheduling/suites/01_formulation_comparison.yaml",
        help="Formulation suite containing the retained MTLSH baseline formulation.",
    )
    parser.add_argument(
        "--retained-formulation-id",
        default="",
        help="Override the retained formulation id. Defaults to benchmark.retained_formulation_id.",
    )
    parser.add_argument(
        "--scenario-family",
        choices=["global", "zone", "line", "all"],
        default="global",
        help="Which benchmark scenario family to include in the generated suite.",
    )
    parser.add_argument(
        "--scenario-ids",
        default="",
        help="Optional comma-separated scenario ids to restrict the generated suite.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=0,
        help="Optional cap on the number of scenarios after filtering. Useful for quick local tests.",
    )
    parser.add_argument(
        "--top-k-values",
        default="1,3,5,all",
        help="Comma-separated list of top-k screening sets to generate, e.g. '1,3,5,all'.",
    )
    parser.add_argument(
        "--out-suite",
        default="",
        help=(
            "Path to the generated suite YAML. "
            "Defaults to results/thesis_optimization_results/configs/generated/"
            "mtlsh_topk_screening_<family>.yaml."
        ),
    )
    parser.add_argument(
        "--results-root",
        default="",
        help="Override the results root used by the generated suite.",
    )
    parser.add_argument(
        "--metadata-json",
        default="",
        help="Optional metadata JSON path. Defaults beside the generated results root.",
    )
    args = parser.parse_args()

    benchmark_path = resolve_path(args.benchmark_config, ROOT)
    formulation_suite_path = resolve_path(args.formulation_suite, ROOT)
    benchmark_cfg = load_benchmark_config(benchmark_path)
    suite_cfg = load_formulation_suite(formulation_suite_path)
    retained_formulation_id = str(args.retained_formulation_id).strip() or str(
        benchmark_cfg.get("retained_formulation_id", "ed_line_n1_surrogate")
    ).strip()

    run_map = formulation_run_map(suite_cfg)
    if retained_formulation_id not in run_map:
        raise KeyError(
            f"Retained formulation '{retained_formulation_id}' was not found in {formulation_suite_path}."
        )
    retained_run = deepcopy(run_map[retained_formulation_id])

    scenario_df, _, severity_df = generate_scenario_manifest(benchmark_cfg)
    family_filter = _scenario_family_filter(args.scenario_family)
    scenarios = scenario_df.loc[scenario_df["scenario_family"].astype(str).isin(family_filter)].copy()
    requested_scenario_ids = [item.strip() for item in str(args.scenario_ids).split(",") if item.strip()]
    if requested_scenario_ids:
        scenarios = scenarios.loc[scenarios["scenario_id"].astype(str).isin(requested_scenario_ids)].copy()
    scenarios = scenarios.sort_values(["scenario_family", "scenario_id"]).reset_index(drop=True)
    if int(args.max_scenarios) > 0:
        scenarios = scenarios.head(int(args.max_scenarios)).copy()
    if scenarios.empty:
        raise ValueError("No scenarios matched the requested filters.")

    screened_outages = select_screened_outages(severity_df, benchmark_cfg).copy()
    screened_outages = screened_outages.sort_values(["selection_rank", "line_uid"], ascending=[True, True]).reset_index(drop=True)
    n_available = int(screened_outages.shape[0])
    if n_available <= 0:
        raise ValueError("No screened outages are available after applying benchmark exclusions.")
    top_k_values = _parse_top_k_values(args.top_k_values, n_available=n_available)

    generated_runs: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for value in top_k_values:
        run_suffix, label = _label_for_top_k(value, n_available=n_available)
        selected_outages = screened_outages if value == "all" else screened_outages.head(int(value))
        line_uids = [int(v) for v in selected_outages["line_uid"].tolist()]
        ranks = [int(v) for v in selected_outages["selection_rank"].tolist()]
        scores = [float(v) for v in selected_outages["severity_score"].tolist()]

        run = deepcopy(retained_run)
        formulation_id = f"{retained_formulation_id}_screen_{run_suffix}"
        base_name = str(retained_run.get("name", retained_formulation_id)).strip() or retained_formulation_id
        base_desc = str(retained_run.get("description", "")).strip()
        run["id"] = formulation_id
        run["name"] = f"{base_name} ({label} screened outages)"
        run["description"] = (
            f"{base_desc} Preventive N-1 screening restricted to the {label} ranked line outages: {line_uids}."
        ).strip()
        run["comparison_family"] = "mtlsh_topk_screening"
        run["overrides"] = _deep_merge(
            dict(retained_run.get("overrides") or {}),
            {
                "constraints": {
                    "include_n1_line_uids": line_uids,
                }
            },
        )
        generated_runs.append(run)
        metadata_rows.append(
            {
                "formulation_id": formulation_id,
                "screening_label": label,
                "n_screened_outages": len(line_uids),
                "line_uids": line_uids,
                "selection_ranks": ranks,
                "severity_scores": scores,
            }
        )

    default_suite_path = GENERATED_CONFIGS_DIR / f"mtlsh_topk_screening_{args.scenario_family}.yaml"
    out_suite = resolve_path(args.out_suite, ROOT) if str(args.out_suite).strip() else default_suite_path
    suite_dir = out_suite.resolve().parent
    default_results_root = RESULTS_DIR / "mtlsh_topk_screening" / args.scenario_family
    results_root = resolve_path(args.results_root, ROOT) if str(args.results_root).strip() else default_results_root
    results_root.mkdir(parents=True, exist_ok=True)

    metadata_json = resolve_path(args.metadata_json, ROOT) if str(args.metadata_json).strip() else (results_root / "screening_sets.json")
    metadata_csv = metadata_json.with_suffix(".csv")

    summary_csv = results_root / "suite_summary.csv"
    summary_md = results_root / "suite_summary.md"
    summary_json = results_root / "suite_summary.json"

    payload = {
        "name": f"chapter5_mtlsh_topk_screening_{args.scenario_family}",
        "baseline_id": generated_runs[-1]["id"],
        "results_root": _relpath(results_root, suite_dir),
        "output": {
            "summary_csv": _relpath(summary_csv, suite_dir),
            "summary_markdown": _relpath(summary_md, suite_dir),
            "summary_json": _relpath(summary_json, suite_dir),
        },
        "metadata": {
            "generator": "generate_mtlsh_topk_screening_suite.py",
            "benchmark_config": str(benchmark_path),
            "formulation_suite": str(formulation_suite_path),
            "retained_formulation_id": retained_formulation_id,
            "scenario_family": args.scenario_family,
            "top_k_values": [value if value == "all" else int(value) for value in top_k_values],
            "screening_metadata_json": str(metadata_json),
            "screening_metadata_csv": str(metadata_csv),
        },
        "scenarios": [_scenario_entry(row) for _, row in scenarios.iterrows()],
        "runs": generated_runs,
    }

    out_suite.parent.mkdir(parents=True, exist_ok=True)
    with out_suite.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "retained_formulation_id": retained_formulation_id,
                "scenario_family": args.scenario_family,
                "available_screened_outages": [
                    {
                        "line_uid": int(row["line_uid"]),
                        "selection_rank": int(row["selection_rank"]),
                        "severity_score": float(row["severity_score"]),
                    }
                    for _, row in screened_outages.iterrows()
                ],
                "generated_runs": metadata_rows,
            },
            f,
            indent=2,
        )

    with metadata_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "formulation_id",
                "screening_label",
                "n_screened_outages",
                "line_uids",
                "selection_ranks",
                "severity_scores",
            ],
        )
        writer.writeheader()
        for row in metadata_rows:
            writer.writerow(
                {
                    "formulation_id": row["formulation_id"],
                    "screening_label": row["screening_label"],
                    "n_screened_outages": row["n_screened_outages"],
                    "line_uids": json.dumps(row["line_uids"]),
                    "selection_ranks": json.dumps(row["selection_ranks"]),
                    "severity_scores": json.dumps(row["severity_scores"]),
                }
            )

    print(json.dumps({"suite": str(out_suite), "results_root": str(results_root), "metadata_json": str(metadata_json)}, indent=2))


if __name__ == "__main__":
    main()
