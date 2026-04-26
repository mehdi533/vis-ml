from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from .config_analysis import (
        BENCHMARK_CONFIG,
        BENCHMARK_CROSS_RESULTS_DIR,
        BENCHMARK_GENERATED_CONFIGS_DIR,
        BENCHMARK_MAIN_RESULTS_DIR,
        BENCHMARK_MANIFESTS_DIR,
        BENCHMARK_REPLAY_RESULTS_DIR,
        CONFIGS_DIR,
        FORMULATION_SUITE_CONFIG,
        OUTPUTS_DIR,
        REPO_ROOT,
        TABLES_DIR,
    )
except ImportError:
    from config_analysis import (  # type: ignore
        BENCHMARK_CONFIG,
        BENCHMARK_CROSS_RESULTS_DIR,
        BENCHMARK_GENERATED_CONFIGS_DIR,
        BENCHMARK_MAIN_RESULTS_DIR,
        BENCHMARK_MANIFESTS_DIR,
        BENCHMARK_REPLAY_RESULTS_DIR,
        CONFIGS_DIR,
        FORMULATION_SUITE_CONFIG,
        OUTPUTS_DIR,
        REPO_ROOT,
        TABLES_DIR,
    )


DEFAULT_SYSTEM_BASE_MVA = 100.0


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def resolve_path(path_like: str | Path, base: Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (base / path).resolve()


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(dict(out[key]), value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def float_token(value: Any) -> str:
    return f"{float(value):.3f}".replace(".", "p")


def write_csv_and_parquet(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> dict[str, str]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    out = {"csv": str(csv_path)}
    if parquet_path is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        write_parquet_compat(df, parquet_path)
        out["parquet"] = str(parquet_path)
    return out


def write_parquet_compat(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
        return
    except Exception:
        pass

    python3 = shutil.which("python3")
    if not python3:
        raise RuntimeError(
            f"Could not write parquet to {path}: current interpreter has no parquet engine and python3 is unavailable."
        )

    with tempfile.TemporaryDirectory(prefix="thesis_opt_parquet_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        csv_path = tmp_dir_path / "frame.csv"
        df.to_csv(csv_path, index=False)
        code = (
            "import pandas as pd, sys; "
            "csv_path, parquet_path = sys.argv[1], sys.argv[2]; "
            "pd.read_csv(csv_path).to_parquet(parquet_path, index=False)"
        )
        proc = subprocess.run(
            [python3, "-c", code, str(csv_path), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Could not write parquet to {path}: {proc.stderr.strip() or proc.stdout.strip() or 'unknown error'}"
            )


def load_benchmark_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else BENCHMARK_CONFIG
    cfg = load_yaml(cfg_path)
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def load_formulation_suite(path: Path | None = None) -> dict[str, Any]:
    suite_path = Path(path) if path is not None else FORMULATION_SUITE_CONFIG
    suite = load_yaml(suite_path)
    suite["_suite_path"] = str(suite_path.resolve())
    return suite


def formulation_run_map(suite_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = {}
    for run in list(suite_cfg.get("runs") or []):
        run_id = str(run.get("id", "")).strip()
        if run_id:
            runs[run_id] = dict(run)
    return runs


def formulation_ids_from_suite(suite_cfg: dict[str, Any]) -> list[str]:
    return list(formulation_run_map(suite_cfg).keys())


def discover_zone_targets(benchmark_cfg: dict[str, Any]) -> list[dict[str, str]]:
    zone_cfg = dict(benchmark_cfg.get("zone_mismatch", {}) or {})
    explicit = list(zone_cfg.get("zones") or [])
    if explicit:
        out = []
        for row in explicit:
            zone_id = str(row.get("zone_id", "")).strip()
            if not zone_id:
                continue
            out.append(
                {
                    "zone_id": zone_id,
                    "zone_name": str(row.get("zone_name", zone_id)),
                    "owner": str(row.get("owner", zone_id)),
                }
            )
        if out:
            return out

    suite_path = resolve_path(
        str(zone_cfg.get("zone_suite_config", "suites/04_zone_mismatch_vis_sensitivity.yaml")),
        CONFIGS_DIR,
    )
    suite = load_yaml(suite_path)
    out: list[dict[str, str]] = []
    for scenario in list(suite.get("scenarios") or []):
        overrides = dict(scenario.get("overrides", {}) or {})
        sc_cfg = dict(overrides.get("scenario", {}) or {})
        owners = [str(v) for v in list(sc_cfg.get("load_step_target_owners") or []) if str(v).strip()]
        if not owners:
            continue
        zone_id = str(scenario.get("id", "")).strip() or f"zone_owner_{owners[0]}"
        out.append(
            {
                "zone_id": zone_id,
                "zone_name": str(scenario.get("name", zone_id)),
                "owner": owners[0],
            }
        )
    return out


def _rank_normalized(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranked = numeric.rank(method="average", pct=True)
    return ranked.astype(float)


def compute_outage_severity_table(benchmark_cfg: dict[str, Any]) -> pd.DataFrame:
    line_cfg = dict(benchmark_cfg.get("line_outage", {}) or {})
    severity_cfg = dict(line_cfg.get("severity_proxy", {}) or {})
    csv_path = resolve_path(
        str(
            severity_cfg.get(
                "source_csv",
                "results/thesis_data_generation_results/results/line_outages_only/simulation_results.csv",
            )
        ),
        REPO_ROOT,
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"Line-outage severity source is missing: {csv_path}")

    columns = {"line_uid"}
    weights = dict(severity_cfg.get("weights") or {})
    if not weights:
        weights = {
            "pre_fault_loading": 0.40,
            "ptdf_l1_norm_outaged_line": 0.35,
            "max_abs_lodf_row": 0.25,
        }
    columns.update(weights.keys())
    df = pd.read_csv(csv_path, usecols=[col for col in columns if col in pd.read_csv(csv_path, nrows=0).columns])
    if "line_uid" not in df.columns:
        raise ValueError(f"Severity source does not include line_uid: {csv_path}")

    grouped = df.groupby("line_uid", dropna=False).median(numeric_only=True).reset_index()
    grouped["line_uid"] = pd.to_numeric(grouped["line_uid"], errors="coerce").astype("Int64")
    grouped = grouped.loc[grouped["line_uid"].notna()].copy()
    grouped["line_uid"] = grouped["line_uid"].astype(int)

    missing = [col for col in weights if col not in grouped.columns]
    if missing:
        raise ValueError(f"Severity source is missing required proxy columns: {missing}")

    grouped["severity_score"] = 0.0
    for col, weight in weights.items():
        grouped[f"{col}_rank"] = _rank_normalized(grouped[col])
        grouped["severity_score"] += float(weight) * grouped[f"{col}_rank"].fillna(0.0)

    grouped = grouped.sort_values(["severity_score", "line_uid"], ascending=[False, True]).reset_index(drop=True)

    n_bins = int(line_cfg.get("severity_bins", 5))
    n_bins = max(1, min(n_bins, max(1, grouped.shape[0])))
    labels = [f"bin_{idx + 1}" for idx in range(n_bins)]
    grouped["severity_bin"] = pd.qcut(
        grouped["severity_score"].rank(method="first"),
        q=n_bins,
        labels=labels,
        duplicates="drop",
    ).astype(str)
    grouped["selection_rank"] = np.arange(1, grouped.shape[0] + 1, dtype=int)
    return grouped


def select_screened_outages(severity_df: pd.DataFrame, benchmark_cfg: dict[str, Any]) -> pd.DataFrame:
    line_cfg = dict(benchmark_cfg.get("line_outage", {}) or {})
    target_count = int(line_cfg.get("n_screened_outages", 10))
    if severity_df.empty or target_count <= 0:
        return severity_df.iloc[0:0].copy()
    excluded_line_uids = {
        int(value)
        for value in list(line_cfg.get("exclude_line_uids") or [])
        if pd.notna(value)
    }
    candidate_df = severity_df.loc[~severity_df["line_uid"].isin(excluded_line_uids)].copy()
    if candidate_df.empty:
        return candidate_df
    severity_df = candidate_df

    bins = [str(value) for value in severity_df["severity_bin"].dropna().astype(str).unique().tolist()]
    bins = sorted(bins)
    per_bin_target = max(1, math.ceil(target_count / max(len(bins), 1)))
    picks: list[pd.DataFrame] = []
    for severity_bin in bins:
        subset = severity_df.loc[severity_df["severity_bin"].astype(str) == severity_bin].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["severity_score", "line_uid"], ascending=[False, True])
        picks.append(subset.head(per_bin_target))

    selected = pd.concat(picks, ignore_index=True, sort=False) if picks else severity_df.iloc[0:0].copy()
    selected = selected.drop_duplicates(subset=["line_uid"]).copy()
    if selected.shape[0] < target_count:
        remaining = severity_df.loc[~severity_df["line_uid"].isin(selected["line_uid"])].copy()
        remaining = remaining.sort_values(["severity_score", "line_uid"], ascending=[False, True])
        selected = pd.concat([selected, remaining.head(target_count - selected.shape[0])], ignore_index=True, sort=False)
    selected = selected.sort_values(["severity_bin", "severity_score", "line_uid"], ascending=[True, False, True]).head(target_count)
    selected["selection_reason"] = "severity_stratified_outage_screen"
    return selected.reset_index(drop=True)


def _global_scenarios(benchmark_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    global_cfg = dict(benchmark_cfg.get("global_load_mismatch", {}) or {})
    rows: list[dict[str, Any]] = []
    for base_scale in list(global_cfg.get("base_scales") or []):
        for step_scale in list(global_cfg.get("step_scales") or []):
            rows.append(
                {
                    "scenario_id": f"global_b{float_token(base_scale)}_s{float_token(step_scale)}",
                    "scenario_family": "global_load_mismatch",
                    "disturbance_type": "load_step",
                    "base_scale": float(base_scale),
                    "step_scale": float(step_scale),
                    "zone": "",
                    "zone_owner": "",
                    "outage_id": pd.NA,
                    "severity_bin": "",
                    "severity_score": np.nan,
                    "notes": "Uniform active/reactive load step across all PQ loads.",
                    "selection_reason": "global_grid",
                    "selected_for_main_benchmark": True,
                    "selected_for_cross_method_subset": False,
                }
            )
    return rows


def _zone_scenarios(benchmark_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    zone_cfg = dict(benchmark_cfg.get("zone_mismatch", {}) or {})
    zones = discover_zone_targets(benchmark_cfg)
    rows: list[dict[str, Any]] = []
    for zone in zones:
        for base_scale in list(zone_cfg.get("base_scales") or []):
            for step_scale in list(zone_cfg.get("step_scales") or []):
                rows.append(
                    {
                        "scenario_id": f"{zone['zone_id']}_b{float_token(base_scale)}_s{float_token(step_scale)}",
                        "scenario_family": "zone_load_mismatch",
                        "disturbance_type": "load_step",
                        "base_scale": float(base_scale),
                        "step_scale": float(step_scale),
                        "zone": str(zone["zone_id"]),
                        "zone_owner": str(zone["owner"]),
                        "outage_id": pd.NA,
                        "severity_bin": "",
                        "severity_score": np.nan,
                        "notes": f"Load step restricted to owner bucket {zone['owner']}.",
                        "selection_reason": "zone_grid",
                        "selected_for_main_benchmark": True,
                        "selected_for_cross_method_subset": False,
                    }
                )
    return rows


def _line_scenarios(benchmark_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    line_cfg = dict(benchmark_cfg.get("line_outage", {}) or {})
    severity_df = compute_outage_severity_table(benchmark_cfg)
    selected = select_screened_outages(severity_df, benchmark_cfg)
    rows: list[dict[str, Any]] = []
    for _, outage in selected.iterrows():
        for base_scale in list(line_cfg.get("base_scales") or []):
            rows.append(
                {
                    "scenario_id": f"line_{int(outage['line_uid']):03d}_b{float_token(base_scale)}",
                    "scenario_family": "line_outage",
                    "disturbance_type": "line_trip",
                    "base_scale": float(base_scale),
                    "step_scale": np.nan,
                    "zone": "",
                    "zone_owner": "",
                    "outage_id": int(outage["line_uid"]),
                    "severity_bin": str(outage["severity_bin"]),
                    "severity_score": float(outage["severity_score"]),
                    "notes": "Post-contingency benchmark using native line-contingency surrogate features.",
                    "selection_reason": str(outage.get("selection_reason", "severity_stratified_outage_screen")),
                    "selected_for_main_benchmark": True,
                    "selected_for_cross_method_subset": False,
                }
            )
    return rows


def apply_cross_method_subset_flags(manifest_df: pd.DataFrame, benchmark_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if manifest_df.empty:
        return manifest_df.copy(), manifest_df.copy()

    df = manifest_df.copy()
    df["stress_score"] = np.where(
        df["disturbance_type"].astype(str) == "load_step",
        pd.to_numeric(df["base_scale"], errors="coerce").fillna(0.0)
        * pd.to_numeric(df["step_scale"], errors="coerce").fillna(1.0),
        pd.to_numeric(df["base_scale"], errors="coerce").fillna(0.0)
        * (1.0 + pd.to_numeric(df["severity_score"], errors="coerce").fillna(0.0)),
    )

    mismatch = df.loc[df["scenario_family"].astype(str).isin(["global_load_mismatch", "zone_load_mismatch"])].copy()
    mismatch = mismatch.sort_values(["stress_score", "scenario_id"], ascending=[True, True])

    picks: list[pd.DataFrame] = []
    easy = mismatch.head(2).copy()
    easy["cross_method_bucket"] = "easy"
    easy["selected_for_cross_method_subset"] = True
    easy["selection_reason"] = "cross_method_easy_by_lowest_stress"
    picks.append(easy)

    remaining = mismatch.loc[~mismatch["scenario_id"].isin(easy["scenario_id"])].copy()
    stressed = remaining.sort_values(["stress_score", "scenario_id"], ascending=[False, True]).head(2).copy()
    stressed["cross_method_bucket"] = "stressed_mismatch"
    stressed["selected_for_cross_method_subset"] = True
    stressed["selection_reason"] = "cross_method_stressed_by_highest_stress"
    picks.append(stressed)

    remaining = remaining.loc[~remaining["scenario_id"].isin(stressed["scenario_id"])].copy()
    if not remaining.empty:
        median_stress = float(pd.to_numeric(mismatch["stress_score"], errors="coerce").median())
        remaining["median_gap"] = (pd.to_numeric(remaining["stress_score"], errors="coerce") - median_stress).abs()
        medium = remaining.sort_values(["median_gap", "scenario_id"], ascending=[True, True]).head(2).copy()
    else:
        medium = remaining.head(0).copy()
    medium["cross_method_bucket"] = "medium"
    medium["selected_for_cross_method_subset"] = True
    medium["selection_reason"] = "cross_method_medium_by_median_stress"
    picks.append(medium)

    line_cases = df.loc[df["scenario_family"].astype(str) == "line_outage"].copy()
    line_cases = line_cases.sort_values(
        ["severity_score", "base_scale", "scenario_id"],
        ascending=[False, False, True],
    )
    severe = line_cases.head(2).copy()
    severe["cross_method_bucket"] = "severe_line_outage"
    severe["selected_for_cross_method_subset"] = True
    severe["selection_reason"] = "cross_method_severe_line_outage"
    picks.append(severe)

    subset = pd.concat(picks, ignore_index=True, sort=False).drop_duplicates(subset=["scenario_id"]).copy()
    subset = subset.sort_values(["cross_method_bucket", "scenario_id"]).reset_index(drop=True)
    df.loc[df["scenario_id"].isin(subset["scenario_id"]), "selected_for_cross_method_subset"] = True
    subset_cols = [col for col in df.columns if col in subset.columns]
    df = df.drop(columns=["stress_score"], errors="ignore")
    subset = subset[subset_cols].drop(columns=["stress_score"], errors="ignore")
    return df, subset


def generate_scenario_manifest(benchmark_cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    severity_df = compute_outage_severity_table(benchmark_cfg)
    rows = _global_scenarios(benchmark_cfg) + _zone_scenarios(benchmark_cfg) + _line_scenarios(benchmark_cfg)
    manifest_df = pd.DataFrame(rows)
    manifest_df = manifest_df.sort_values(["scenario_family", "scenario_id"]).reset_index(drop=True)
    manifest_df, subset_df = apply_cross_method_subset_flags(manifest_df, benchmark_cfg)
    return manifest_df, subset_df, severity_df


def task_root_for_group(group: str) -> Path:
    group_key = str(group).strip().lower()
    if group_key == "main":
        return BENCHMARK_MAIN_RESULTS_DIR
    if group_key == "cross_method_subset":
        return BENCHMARK_CROSS_RESULTS_DIR
    if group_key == "replay_main":
        return BENCHMARK_REPLAY_RESULTS_DIR / "main"
    if group_key == "replay_cross_method_subset":
        return BENCHMARK_REPLAY_RESULTS_DIR / "cross_method_subset"
    raise ValueError(f"Unsupported benchmark group: {group}")


def build_task_manifest(
    scenario_df: pd.DataFrame,
    formulation_ids: list[str],
    *,
    group: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = task_root_for_group(group)
    for scenario_id in scenario_df["scenario_id"].astype(str).tolist():
        for formulation_id in formulation_ids:
            run_dir = root / scenario_id / formulation_id
            rows.append(
                {
                    "task_id": f"{group}__{scenario_id}__{formulation_id}",
                    "benchmark_group": group,
                    "scenario_id": scenario_id,
                    "formulation_id": formulation_id,
                    "run_dir": str(run_dir),
                    "summary_json": str(run_dir / f"{formulation_id}_summary.json"),
                    "resolved_config": str(run_dir / f"{formulation_id}_resolved_config.yaml"),
                    "log_file": str(run_dir / f"{formulation_id}.log"),
                }
            )
    task_df = pd.DataFrame(rows)
    if not task_df.empty:
        task_df.insert(0, "task_index", np.arange(task_df.shape[0], dtype=int))
    return task_df


def build_replay_task_manifest(task_df: pd.DataFrame, scenario_df: pd.DataFrame, *, group: str) -> pd.DataFrame:
    if task_df.empty:
        return task_df.copy()
    scenario_lookup = scenario_df.set_index("scenario_id")
    replay_group = f"replay_{group}"
    replay_root = task_root_for_group(replay_group)
    rows: list[dict[str, Any]] = []
    for _, row in task_df.iterrows():
        scenario_id = str(row["scenario_id"])
        formulation_id = str(row["formulation_id"])
        scenario = scenario_lookup.loc[scenario_id]
        replay_dir = replay_root / scenario_id / formulation_id
        rows.append(
            {
                "task_index": int(row["task_index"]),
                "task_id": f"{replay_group}__{scenario_id}__{formulation_id}",
                "benchmark_group": replay_group,
                "scenario_id": scenario_id,
                "formulation_id": formulation_id,
                "summary_json": str(row["summary_json"]),
                "resolved_config": str(row["resolved_config"]),
                "replay_dir": str(replay_dir),
                "replay_summary_json": str(replay_dir / f"{formulation_id}_replay_summary.json"),
                "replay_summary_csv": str(replay_dir / f"{formulation_id}_replay_summary.csv"),
                "replay_detail_csv": str(replay_dir / f"{formulation_id}_replay_detail.csv"),
                "contingency_type": str(scenario["disturbance_type"]),
                "outage_id": scenario.get("outage_id", pd.NA),
            }
        )
    return pd.DataFrame(rows)


def scenario_overrides_from_row(row: pd.Series) -> dict[str, Any]:
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
        zone_owner = row.get("zone_owner", "")
        if not pd.isna(zone_owner) and str(zone_owner).strip():
            scenario_cfg["load_step_target_owners"] = [str(row["zone_owner"])]
        features_cfg["contingency_mode"] = "load_mismatch"
    elif disturbance_type == "line_trip":
        scenario_cfg["step_scale"] = 1.0
        scenario_cfg["contingency_line_uid"] = int(row["outage_id"])
        features_cfg["contingency_mode"] = "line"
        features_cfg["contingency_line_uid"] = int(row["outage_id"])
    else:
        raise ValueError(f"Unsupported disturbance_type: {disturbance_type}")
    return {"scenario": scenario_cfg, "features": features_cfg}


def build_optimization_config(
    *,
    benchmark_cfg: dict[str, Any],
    suite_cfg: dict[str, Any],
    scenario_row: pd.Series,
    formulation_id: str,
    group: str,
) -> tuple[dict[str, Any], Path]:
    run_map = formulation_run_map(suite_cfg)
    if formulation_id not in run_map:
        raise KeyError(f"Unknown formulation_id in suite: {formulation_id}")
    run_cfg = run_map[formulation_id]
    suite_base = Path(suite_cfg.get("_suite_path", FORMULATION_SUITE_CONFIG)).resolve().parent
    base_cfg_path = resolve_path(str(run_cfg.get("base_config", "../base_optimization.yaml")), suite_base)
    base_cfg = load_yaml(base_cfg_path)

    cfg = deep_merge(base_cfg, scenario_overrides_from_row(scenario_row))
    cfg = deep_merge(cfg, dict(run_cfg.get("overrides") or {}))

    cfg.setdefault("formulation", {})
    cfg["formulation"]["id"] = formulation_id
    cfg["formulation"]["name"] = str(run_cfg.get("name", formulation_id))
    cfg["formulation"]["description"] = str(run_cfg.get("description", ""))

    run_dir = task_root_for_group(group) / str(scenario_row["scenario_id"]) / formulation_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("output", {})
    cfg["output"]["run_tag"] = formulation_id
    cfg["output"]["results_dir"] = str(run_dir)
    cfg["output"]["log_file"] = str((run_dir / f"{formulation_id}.log").resolve())
    resolved_cfg_path = run_dir / f"{formulation_id}_resolved_config.yaml"
    return cfg, resolved_cfg_path


def benchmark_roots(benchmark_cfg: dict[str, Any]) -> dict[str, Path]:
    results_cfg = dict(benchmark_cfg.get("results", {}) or {})
    return {
        "main": resolve_path(str(results_cfg.get("main_results_root", BENCHMARK_MAIN_RESULTS_DIR)), REPO_ROOT),
        "cross_method_subset": resolve_path(
            str(results_cfg.get("cross_results_root", BENCHMARK_CROSS_RESULTS_DIR)),
            REPO_ROOT,
        ),
        "replay_main": resolve_path(
            str(results_cfg.get("replay_main_results_root", BENCHMARK_REPLAY_RESULTS_DIR / "main")),
            REPO_ROOT,
        ),
        "replay_cross_method_subset": resolve_path(
            str(results_cfg.get("replay_cross_results_root", BENCHMARK_REPLAY_RESULTS_DIR / "cross_method_subset")),
            REPO_ROOT,
        ),
    }


def save_manifest_bundle(
    *,
    manifest_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    severity_df: pd.DataFrame,
    benchmark_cfg: dict[str, Any],
    suite_cfg: dict[str, Any],
) -> dict[str, Any]:
    output_tables_dir = TABLES_DIR
    output_tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv_and_parquet(
        manifest_df,
        output_tables_dir / "scenario_manifest.csv",
        output_tables_dir / "scenario_manifest.parquet",
    )
    write_csv_and_parquet(
        subset_df,
        output_tables_dir / "cross_method_subset_manifest.csv",
        output_tables_dir / "cross_method_subset_manifest.parquet",
    )

    BENCHMARK_MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_and_parquet(
        severity_df,
        BENCHMARK_MANIFESTS_DIR / "line_outage_severity_scores.csv",
        BENCHMARK_MANIFESTS_DIR / "line_outage_severity_scores.parquet",
    )

    main_formulations = [str(v) for v in list(benchmark_cfg.get("main_formulation_ids") or [])]
    if not main_formulations:
        main_formulations = [str(benchmark_cfg.get("retained_formulation_id", "ed_line_n1_surrogate"))]
    cross_formulations = [str(v) for v in list(benchmark_cfg.get("cross_method_formulation_ids") or [])]
    if not cross_formulations:
        cross_formulations = formulation_ids_from_suite(suite_cfg)

    main_task_df = build_task_manifest(
        manifest_df.loc[manifest_df["selected_for_main_benchmark"] == True].copy(),
        main_formulations,
        group="main",
    )
    cross_task_df = build_task_manifest(subset_df, cross_formulations, group="cross_method_subset")
    replay_main_df = build_replay_task_manifest(
        main_task_df,
        manifest_df.set_index("scenario_id", drop=False).reset_index(drop=True),
        group="main",
    )
    replay_cross_df = build_replay_task_manifest(
        cross_task_df,
        subset_df.set_index("scenario_id", drop=False).reset_index(drop=True),
        group="cross_method_subset",
    )

    for stem, df in [
        ("main_benchmark_tasks", main_task_df),
        ("cross_method_benchmark_tasks", cross_task_df),
        ("replay_main_benchmark_tasks", replay_main_df),
        ("replay_cross_method_benchmark_tasks", replay_cross_df),
    ]:
        write_csv_and_parquet(
            df,
            BENCHMARK_MANIFESTS_DIR / f"{stem}.csv",
            BENCHMARK_MANIFESTS_DIR / f"{stem}.parquet",
        )

    manifest_summary = {
        "n_main_scenarios": int(manifest_df["selected_for_main_benchmark"].fillna(False).sum()),
        "n_cross_method_scenarios": int(subset_df.shape[0]),
        "n_main_jobs": int(main_task_df.shape[0]),
        "n_cross_method_jobs": int(cross_task_df.shape[0]),
        "selected_outages": [
            {
                "line_uid": int(row["line_uid"]),
                "severity_bin": str(row["severity_bin"]),
                "severity_score": float(row["severity_score"]),
            }
            for _, row in select_screened_outages(severity_df, benchmark_cfg).iterrows()
        ],
    }
    summary_path = BENCHMARK_MANIFESTS_DIR / "benchmark_manifest_summary.json"
    summary_path.write_text(json.dumps(manifest_summary, indent=2), encoding="utf-8")
    return manifest_summary
