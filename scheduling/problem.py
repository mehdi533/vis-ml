from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import andes
import cvxpy as cp
import numpy as np
import yaml

# Support both:
# - python -m scheduling.problem
# - python scheduling/problem.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scheduling.constraints import (
        build_basecase_line_constraints,
        build_ed_constraints,
        build_input_feature_constraints,
        build_n1_constraints,
        build_n1_redispatch_constraints,
        build_output_feature_constraints,
        evaluate_line_security_metrics,
    )
    from scheduling.constraints_nn import build_nn_constraints, compute_nn_output_interval
except ModuleNotFoundError:
    from final_optimization_folder.constraints import (
        build_basecase_line_constraints,
        build_ed_constraints,
        build_input_feature_constraints,
        build_n1_constraints,
        build_n1_redispatch_constraints,
        build_output_feature_constraints,
        evaluate_line_security_metrics,
    )
    from final_optimization_folder.constraints_nn import build_nn_constraints, compute_nn_output_interval
from scheduling.utils import (
    add_measurement_devices,
    build_features,
    build_post_step_p_vector,
    build_model_from_cfg,
    derive_sched_dispatch_vectors,
    load_table_3_1_dispatch_cost_arrays,
    load_optimization_config,
    load_scalers,
    parse_constraint_switches,
    parse_plot_options,
    parse_solver_options,
    setup_logger,
)


def _sanitize_token(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return safe or "default"


def _scenario_id_from_cfg(cfg: Mapping[str, Any]) -> str:
    scenario_cfg = dict(cfg.get("scenario", {}) or {})
    raw = str(scenario_cfg.get("id", "")).strip()
    if raw:
        return _sanitize_token(raw)

    base_scale = scenario_cfg.get("base_scale")
    step_scale = scenario_cfg.get("step_scale")
    load_step_time = scenario_cfg.get("load_step_time")
    parts = []
    if base_scale is not None:
        parts.append(f"b{float(base_scale):.3f}".replace(".", "p"))
    if step_scale is not None:
        parts.append(f"s{float(step_scale):.3f}".replace(".", "p"))
    if load_step_time is not None:
        parts.append(f"t{float(load_step_time):.3f}".replace(".", "p"))
    return _sanitize_token("_".join(parts) or "default")


def _resolve_feature_contingency(
    ss: andes.System,
    *,
    feature_cfg: Mapping[str, Any],
    scenario_cfg: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw_mode = feature_cfg.get("contingency_mode", scenario_cfg.get("contingency_mode", "load_mismatch"))
    mode = str(raw_mode).strip().lower()
    if mode in {"", "none", "load", "load_mismatch"}:
        return None

    if mode in {"line", "line_outage", "line_plus_load"}:
        raw_uid = feature_cfg.get("contingency_line_uid", scenario_cfg.get("contingency_line_uid"))
        if raw_uid is None:
            raise ValueError(
                "features.contingency_mode=line requires features.contingency_line_uid "
                "(or scenario.contingency_line_uid)."
            )
        uid = int(raw_uid)
        n_line = int(getattr(getattr(ss, "Line", None), "n", 0))
        if uid < 0 or uid >= n_line:
            raise ValueError(
                f"features.contingency_line_uid out of range: {uid}. Valid line uid range is [0, {max(n_line - 1, 0)}]."
            )
        contingency: dict[str, Any] = {"uid": int(uid)}
        line = getattr(ss, "Line", None)
        if line is not None:
            bus1 = np.asarray(getattr(getattr(line, "bus1", None), "v", []), dtype=int).reshape(-1)
            bus2 = np.asarray(getattr(getattr(line, "bus2", None), "v", []), dtype=int).reshape(-1)
            if uid < int(bus1.size):
                contingency["bus1"] = int(bus1[uid])
            if uid < int(bus2.size):
                contingency["bus2"] = int(bus2[uid])
        return contingency

    raise ValueError(
        f"Unsupported features.contingency_mode='{mode}'. Supported: load_mismatch, none, line."
    )


def _enforce_strict_feature_policy(feature_cfg: Mapping[str, Any]) -> None:
    """
    Enforce "extract from system only" feature construction.
    """
    if bool(feature_cfg.get("registry_include_unbuildable_model_features", False)):
        raise ValueError(
            "features.registry_include_unbuildable_model_features=true is not allowed in strict mode."
        )
    source = str(feature_cfg.get("missing_feature_source", "none")).strip().lower()
    if source not in {"", "none"}:
        raise ValueError(
            "features.missing_feature_source must be 'none' in strict mode (extract from system only)."
        )
    if bool(feature_cfg.get("allow_missing_features", False)):
        raise ValueError("features.allow_missing_features=true is not allowed in strict mode.")
    if bool(feature_cfg.get("fixed_source_override_non_sched", False)):
        raise ValueError(
            "features.fixed_source_override_non_sched=true is not allowed in strict mode."
        )
    if bool(feature_cfg.get("allow_constant_fill_for_unresolved_missing", False)):
        raise ValueError(
            "features.allow_constant_fill_for_unresolved_missing=true is not allowed in strict mode."
        )


def _resolve_model_dir(model_cfg: Mapping[str, Any]) -> Path:
    raw = str(model_cfg.get("model_dir", model_cfg.get("state_dict", ""))).strip()
    if not raw:
        raise ValueError("Missing model.model_dir or model.state_dict in optimization config.")
    path = Path(raw)
    return path.parent if path.suffix else path


def _load_feature_contract_from_model_dir(model_cfg: Mapping[str, Any]) -> list[str]:
    model_dir = _resolve_model_dir(model_cfg)
    run_cfg_path = model_dir / "run_config.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(
            f"features.x_features_from_model_dir=true but run_config.yaml was not found at {run_cfg_path}"
        )
    with run_cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    feature_cols = list((payload.get("resolved", {}) or {}).get("feature_cols") or [])
    feature_cols = [str(name) for name in feature_cols if str(name).strip()]
    if not feature_cols:
        raise ValueError(
            f"run_config.yaml does not contain resolved.feature_cols at {run_cfg_path}"
        )
    return feature_cols


def _load_missing_feature_reference_values(
    *,
    cfg: Mapping[str, Any],
    x_features: list[str],
) -> dict[str, float]:
    """
    Load fixed values for non-buildable features from a reference dataset row.
    """
    feature_cfg = dict(cfg.get("features", {}) or {})
    source = str(feature_cfg.get("missing_feature_source", "")).strip().lower()
    if not source or source == "none":
        return {}

    csv_path: Path | None = None
    if source == "model_data_first_row":
        model_dir = _resolve_model_dir(cfg.get("model", {}) or {})
        run_cfg_path = model_dir / "run_config.yaml"
        if not run_cfg_path.exists():
            raise FileNotFoundError(
                f"Missing feature source '{source}' requires run_config.yaml in model_dir: {run_cfg_path}"
            )
        with run_cfg_path.open("r", encoding="utf-8") as f:
            model_run_cfg = yaml.safe_load(f) or {}
        raw_csv = str((model_run_cfg.get("data", {}) or {}).get("csv_path", "")).strip()
        if not raw_csv:
            raise ValueError(
                f"Missing feature source '{source}' requires data.csv_path in {run_cfg_path}"
            )
        csv_path = Path(raw_csv)
    elif source == "csv_first_row":
        raw_csv = str(feature_cfg.get("missing_feature_csv_path", "")).strip()
        if not raw_csv:
            raise ValueError(
                "features.missing_feature_source=csv_first_row requires features.missing_feature_csv_path"
            )
        csv_path = Path(raw_csv)
    else:
        raise ValueError(
            f"Unsupported features.missing_feature_source='{source}'. "
            "Supported: none, model_data_first_row, csv_first_row."
        )

    assert csv_path is not None
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing feature reference CSV not found: {csv_path}")

    requested = set(str(name) for name in x_features)
    fixed: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        first = next(reader, None)
    if first is None:
        raise ValueError(f"Missing feature reference CSV is empty: {csv_path}")

    for key, raw in first.items():
        if key is None:
            continue
        name = str(key).strip()
        if not name or name not in requested:
            continue
        text = str(raw).strip()
        if not text:
            continue
        try:
            val = float(text)
        except Exception:
            continue
        if np.isfinite(val):
            fixed[name] = float(val)
    return fixed


def _collect_buildable_feature_names(
    ss: andes.System,
    *,
    base_scale: float,
    step_scale: float,
    load_step_time: float,
    m_seed: np.ndarray,
    d_seed: np.ndarray,
    contingency: Mapping[str, Any] | None = None,
    feature_names_path: str | None = None,
    load_step_target_pq_names: Sequence[str] | None = None,
    load_step_target_owners: Sequence[str] | None = None,
) -> set[str]:
    feat = build_features(
        ss,
        base_scale=base_scale,
        step_scale=step_scale,
        load_step_time=load_step_time,
        M_vec=m_seed,
        D_vec=d_seed,
        contingency=contingency,
        feature_names_path=feature_names_path,
        load_step_target_pq_names=load_step_target_pq_names,
        load_step_target_owners=load_step_target_owners,
    )
    genrou_pg, regcv1_pg = derive_sched_dispatch_vectors(ss)
    for i, val in enumerate(genrou_pg, start=1):
        feat[f"P_GENROU_{i}"] = float(val)
    for i, val in enumerate(regcv1_pg, start=1):
        feat[f"P_REGCV1_{i}"] = float(val)
    gen_total = float(np.sum(np.asarray(genrou_pg, dtype=float))) if genrou_pg.size else 0.0
    reg_total = float(np.sum(np.asarray(regcv1_pg, dtype=float))) if regcv1_pg.size else 0.0
    dispatch_total = float(gen_total + reg_total)
    feat["P_GENROU_TOTAL"] = float(gen_total)
    feat["P_REGCV1_TOTAL"] = float(reg_total)
    feat["P_DISPATCH_TOTAL"] = float(dispatch_total)
    feat["P_REGCV1_SHARE"] = float(reg_total / dispatch_total) if abs(dispatch_total) > 1e-12 else 0.0
    return {str(name) for name in feat.keys()}


def _load_x_features_from_registry(
    *,
    feature_cfg: Mapping[str, Any],
    ss: andes.System,
    base_scale: float,
    step_scale: float,
    load_step_time: float,
    m_seed: np.ndarray,
    d_seed: np.ndarray,
    contingency: Mapping[str, Any] | None = None,
    model_feature_order: list[str] | None = None,
    load_step_target_pq_names: Sequence[str] | None = None,
    load_step_target_owners: Sequence[str] | None = None,
) -> list[str]:
    raw_path = str(feature_cfg.get("feature_names_path", "configs/data_generation_feature_names.yaml")).strip()
    reg_path = Path(raw_path)
    if not reg_path.is_absolute():
        reg_path = (ROOT / reg_path).resolve()
    if not reg_path.exists():
        raise FileNotFoundError(f"Feature registry YAML not found: {reg_path}")

    with reg_path.open("r", encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    groups = [str(v) for v in list(feature_cfg.get("registry_feature_groups") or ["x_op", "x_cont", "x_sched"])]
    exclude_prefixes = [str(v) for v in list(feature_cfg.get("exclude_prefixes") or ["x0__"])]
    group_overrides = dict(feature_cfg.get("registry_group_overrides") or {})

    pq_names = [str(v) for v in list(getattr(ss.PQ, "name", {}).v)] if getattr(ss, "PQ", None) is not None else []
    pq_owners = (
        sorted({_sanitize_token(v) for v in list(getattr(ss.PQ, "owner", {}).v)})
        if getattr(ss, "PQ", None) is not None
        else []
    )
    n_genrou = int(getattr(getattr(ss, "GENROU", None), "n", 0))
    n_regcv1 = int(getattr(getattr(ss, "REGCV1", None), "n", 0))
    n_line = int(getattr(getattr(ss, "Line", None), "n", 0))

    candidates: list[str] = []
    for group in groups:
        group_cfg = dict(registry.get(group, {}) or {})
        override_cfg = dict(group_overrides.get(group, {}) or {})
        keep_fields = {str(v) for v in list(override_cfg.get("keep_fields") or [])}
        keep_prefix_keys = {str(v) for v in list(override_cfg.get("keep_prefix_keys") or [])}
        for key, value in group_cfg.items():
            if key == "prefixes":
                continue
            if isinstance(value, list):
                for name in value:
                    s = str(name).strip()
                    if keep_fields and s not in keep_fields:
                        continue
                    if s:
                        candidates.append(s)

        prefixes = dict(group_cfg.get("prefixes", {}) or {})
        for pref_key, pref_val in prefixes.items():
            if keep_prefix_keys and str(pref_key) not in keep_prefix_keys:
                continue
            pref = str(pref_val)
            if pref_key in {"pq_p_base", "pq_q_base", "pq_delta_p"}:
                candidates.extend(f"{pref}{name}" for name in pq_names)
            elif pref_key == "owner_delta_p":
                candidates.extend(f"{pref}{owner}" for owner in pq_owners)
            elif pref_key in {"M", "D", "p_regcv1", "p_regcv1_reserve"}:
                candidates.extend(f"{pref}{i}" for i in range(1, n_regcv1 + 1))
            elif pref_key in {"p_genrou", "p_genrou_reserve"}:
                candidates.extend(f"{pref}{i}" for i in range(1, n_genrou + 1))
            elif pref_key == "line_one_hot":
                candidates.extend(f"{pref}{i}" for i in range(0, n_line))

    def _keep(name: str) -> bool:
        return bool(name) and not any(name.startswith(pfx) for pfx in exclude_prefixes)

    deduped: list[str] = []
    seen = set()
    for name in candidates:
        if not _keep(name):
            continue
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)

    if bool(feature_cfg.get("registry_filter_to_buildable", True)):
        buildable = _collect_buildable_feature_names(
            ss,
            base_scale=base_scale,
            step_scale=step_scale,
            load_step_time=load_step_time,
            m_seed=m_seed,
            d_seed=d_seed,
            contingency=contingency,
            feature_names_path=str(feature_cfg.get("feature_names_path", "configs/data_generation_feature_names.yaml")),
            load_step_target_pq_names=load_step_target_pq_names,
            load_step_target_owners=load_step_target_owners,
        )
        deduped = [name for name in deduped if name in buildable]

    if model_feature_order:
        registry_set = set(deduped)
        missing_in_registry = [name for name in model_feature_order if name not in registry_set]
        if missing_in_registry:
            if bool(feature_cfg.get("registry_include_unbuildable_model_features", False)):
                # Preserve full model/scaler contract even when some features are not buildable
                # by the scheduler; those can be fixed later via missing-fill logic.
                deduped = [name for name in model_feature_order if _keep(name)]
            else:
                preview = ", ".join(missing_in_registry[:12])
                if len(missing_in_registry) > 12:
                    preview += ", ..."
                raise ValueError(
                    "Model feature contract is not fully covered by registry-derived features. "
                    f"{len(missing_in_registry)} missing, examples: {preview}"
                )
        else:
            # Keep model/scaler feature order for strict compatibility.
            deduped = [name for name in model_feature_order if name in registry_set]

    if not deduped:
        raise ValueError(
            f"No optimization x_features resolved from feature registry {reg_path} "
            f"for groups={groups}. Check registry settings and filters."
        )
    return deduped


def _scale_subset(values: np.ndarray, scaler, idx: list[int]) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=int)
    return values * scaler.scale_[idx_arr] + scaler.min_[idx_arr]


def _unscale_subset(values: np.ndarray, scaler, idx: list[int]) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=int)
    return (values - scaler.min_[idx_arr]) / scaler.scale_[idx_arr]


def _set_feature_scaled_bounds(
    *,
    x_min_sc: np.ndarray,
    x_max_sc: np.ndarray,
    x_scaler: Any,
    name_to_idx: Mapping[str, int],
    feature_name: str,
    lo_raw: float,
    hi_raw: float,
) -> None:
    idx = name_to_idx.get(feature_name)
    if idx is None:
        return
    lo_sc = (float(lo_raw) * float(x_scaler.scale_[idx])) + float(x_scaler.min_[idx])
    hi_sc = (float(hi_raw) * float(x_scaler.scale_[idx])) + float(x_scaler.min_[idx])
    x_min_sc[idx] = min(lo_sc, hi_sc)
    x_max_sc[idx] = max(lo_sc, hi_sc)


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


def _problem_stats(prob: cp.Problem, solver_name: str) -> tuple[int, int | None]:
    metrics = prob.size_metrics
    n_constraints = int(metrics.num_scalar_eq_constr + metrics.num_scalar_leq_constr)

    nnz_total: int | None = None
    try:
        data, _, _ = prob.get_problem_data(solver_name)
        nnz = 0
        for value in data.values():
            if hasattr(value, "nnz"):
                nnz += int(value.nnz)
            elif isinstance(value, np.ndarray):
                nnz += int(np.count_nonzero(value))
        nnz_total = nnz
    except Exception:
        nnz_total = None
    return n_constraints, nnz_total


def _scalar_constraint_count(constraints: list[cp.Constraint]) -> int:
    total = 0
    for cons in constraints:
        size = getattr(cons, "size", None)
        if size is None:
            shape = getattr(cons, "shape", ())
            size = int(np.prod(shape)) if shape else 1
        total += int(size)
    return int(total)


def _variable_type_counts(prob: cp.Problem) -> dict[str, int]:
    n_binary = 0
    n_integer = 0
    n_total = 0
    for var in prob.variables():
        n = int(np.prod(var.shape))
        n_total += n
        attrs = getattr(var, "attributes", {})
        if bool(attrs.get("boolean", False)):
            n_binary += n
        elif bool(attrs.get("integer", False)):
            n_integer += n
    n_cont = n_total - n_binary - n_integer
    return {
        "n_variables_total": int(n_total),
        "n_variables_continuous": int(n_cont),
        "n_variables_binary": int(n_binary),
        "n_variables_integer_nonbinary": int(n_integer),
    }


def _build_solver_kwargs(cfg: Mapping[str, Any], solver_name: str, verbose: bool, reoptimize: bool) -> dict[str, Any]:
    solver_cfg = cfg.get("solver", {}) if isinstance(cfg, Mapping) else {}
    solver_name_u = str(solver_name).upper()
    kwargs: dict[str, Any] = {
        "solver": solver_name,
        "verbose": bool(verbose),
    }
    if solver_name_u == "GUROBI":
        kwargs["reoptimize"] = bool(reoptimize)

    extra = solver_cfg.get("extra_kwargs", {})
    if isinstance(extra, Mapping):
        kwargs.update(dict(extra))

    alias_map = {
        "time_limit": "TimeLimit",
        "mip_gap": "MIPGap",
        "threads": "Threads",
    }
    for key, target in alias_map.items():
        if key in solver_cfg and target not in kwargs:
            kwargs[target] = solver_cfg[key]

    # Keep GUROBI-specific knobs out of non-GUROBI solver calls (e.g. OSQP).
    if solver_name_u != "GUROBI":
        for bad_key in ("reoptimize", "TimeLimit", "MIPGap", "Threads"):
            kwargs.pop(bad_key, None)
    return kwargs


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


def _build_x_seed(
    ss: andes.System,
    *,
    x_features: list[str],
    base_scale: float,
    step_scale: float,
    load_step_time: float,
    m_seed: np.ndarray,
    d_seed: np.ndarray,
    contingency: Mapping[str, Any] | None = None,
    feature_names_path: str | None = None,
    load_step_target_pq_names: Sequence[str] | None = None,
    load_step_target_owners: Sequence[str] | None = None,
    fixed_feature_values: Mapping[str, float] | None = None,
    missing_fill_value: float = 0.0,
    allow_missing_features: bool = False,
    allow_constant_fill_for_unresolved_missing: bool = False,
    fixed_source_override_non_sched: bool = False,
) -> np.ndarray:
    feat = build_features(
        ss,
        base_scale=base_scale,
        step_scale=step_scale,
        load_step_time=load_step_time,
        M_vec=m_seed,
        D_vec=d_seed,
        contingency=contingency,
        feature_names_path=feature_names_path,
        load_step_target_pq_names=load_step_target_pq_names,
        load_step_target_owners=load_step_target_owners,
    )
    fixed_map = {
        str(k): float(v)
        for k, v in dict(fixed_feature_values or {}).items()
        if np.isfinite(float(v))
    }

    genrou = getattr(ss, "GENROU", None)
    regcv1 = getattr(ss, "REGCV1", None)
    genrou_pg, regcv1_pg = derive_sched_dispatch_vectors(ss)

    for i, val in enumerate(genrou_pg, start=1):
        feat[f"P_GENROU_{i}"] = float(val)
    for i, val in enumerate(regcv1_pg, start=1):
        feat[f"P_REGCV1_{i}"] = float(val)

    if fixed_source_override_non_sched and fixed_map:
        n_genrou = int(getattr(genrou, "n", 0)) if genrou is not None else int(genrou_pg.size)
        n_regcv1 = int(getattr(regcv1, "n", 0)) if regcv1 is not None else int(regcv1_pg.size)
        mutable = set()
        mutable.update({f"M_{i + 1}" for i in range(n_regcv1)})
        mutable.update({f"D_{i + 1}" for i in range(n_regcv1)})
        mutable.update({f"P_GENROU_{i + 1}" for i in range(n_genrou)})
        mutable.update({f"P_REGCV1_{i + 1}" for i in range(n_regcv1)})
        for name in x_features:
            if name in mutable:
                continue
            if name in fixed_map:
                feat[name] = float(fixed_map[name])

    # If a buildable feature is present but non-finite, prefer the configured fixed reference value.
    for name in x_features:
        if name not in feat:
            continue
        try:
            val = float(feat[name])
        except Exception:
            val = float("nan")
        if np.isfinite(val):
            continue
        if name in fixed_map:
            feat[name] = float(fixed_map[name])

    missing = [name for name in x_features if name not in feat]
    if missing:
        unresolved: list[str] = []
        for name in missing:
            if name in fixed_map and np.isfinite(float(fixed_map[name])):
                feat[name] = float(fixed_map[name])
            else:
                unresolved.append(name)

        if unresolved:
            if not allow_missing_features:
                preview = ", ".join(unresolved[:12])
                if len(unresolved) > 12:
                    preview += ", ..."
                raise ValueError(
                    "Optimization feature contract mismatch: the configured surrogate expects input columns "
                    "that the current optimizer does not build and that were not available in the configured "
                    f"fixed-feature source ({len(unresolved)} unresolved). "
                    f"Examples: {preview}. "
                    "Use an optimization-ready surrogate trained on buildable scheduling features, or extend "
                    "the optimization feature builder / fixed-feature source before embedding this model."
                )
            if allow_constant_fill_for_unresolved_missing:
                for name in unresolved:
                    feat[name] = float(missing_fill_value)
            else:
                preview = ", ".join(unresolved[:12])
                if len(unresolved) > 12:
                    preview += ", ..."
                raise ValueError(
                    "Missing features remained unresolved after applying fixed feature source. "
                    f"{len(unresolved)} unresolved, examples: {preview}. "
                    "Provide a reference source that contains these columns, or enable "
                    "allow_constant_fill_for_unresolved_missing."
                )

    nonfinite = [name for name in x_features if not np.isfinite(float(feat[name]))]
    if nonfinite:
        if allow_missing_features and allow_constant_fill_for_unresolved_missing:
            for name in nonfinite:
                feat[name] = float(missing_fill_value)
            nonfinite = []
        if nonfinite:
            preview = ", ".join(nonfinite[:12])
            if len(nonfinite) > 12:
                preview += ", ..."
            raise ValueError(
                "Non-finite values detected in resolved optimization feature vector after "
                "build + fixed-feature injection. "
                f"{len(nonfinite)} features are non-finite; examples: {preview}."
            )
    return np.asarray([feat[name] for name in x_features], dtype=float)


def _resolve_ed_cost_arrays(cfg: Mapping[str, Any], ss: andes.System) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ed_costs_cfg = dict(cfg.get("ed_costs", {}) or {})
    cost_table_path = str(ed_costs_cfg.get("cost_table_path", "")).strip()
    if cost_table_path:
        return load_table_3_1_dispatch_cost_arrays(ss, cost_table_path=cost_table_path)

    a_raw = ed_costs_cfg.get("a")
    b_raw = ed_costs_cfg.get("b")
    c_raw = ed_costs_cfg.get("c")
    if a_raw is None or b_raw is None or c_raw is None:
        raise ValueError(
            "Missing ED cost definition. Provide ed_costs.cost_table_path or explicit ed_costs.a/b/c arrays."
        )
    return (
        np.asarray(a_raw, dtype=float),
        np.asarray(b_raw, dtype=float),
        np.asarray(c_raw, dtype=float),
    )


def run_optimization(cfg: dict[str, Any], *, config_path: str | None = None):
    wall_start = time.perf_counter()

    formulation_cfg = cfg.get("formulation", {})
    output_cfg = cfg.get("output", {})
    formulation_id = str(formulation_cfg.get("id", output_cfg.get("run_tag", "run"))).strip() or "run"
    run_id = str(output_cfg.get("run_tag", formulation_id)).strip() or formulation_id
    formulation_name = str(formulation_cfg.get("name", formulation_id))
    equation_map = formulation_cfg.get("equation_map", {})
    scenario_cfg = dict(cfg.get("scenario", {}) or {})
    scenario_id = _scenario_id_from_cfg(cfg)
    scenario_name = str(scenario_cfg.get("name", scenario_id))
    scenario_description = str(scenario_cfg.get("description", ""))

    output_paths = _default_output_paths(output_cfg, run_id)
    logger = setup_logger(Path(output_cfg.get("log_file", output_paths["summary_json"].with_suffix(".log"))))

    switches = parse_constraint_switches(cfg)
    solver = parse_solver_options(cfg)
    plot_options = parse_plot_options(cfg)

    ss = andes.load(cfg["system"]["case"], setup=False)
    ss.config.freq = float(cfg.get("system", {}).get("frequency_hz", 50.0))
    add_measurement_devices(ss)

    base_scale = float(cfg["scenario"]["base_scale"])
    step_scale = float(cfg["scenario"]["step_scale"])
    load_step_time = float(cfg["scenario"]["load_step_time"])
    load_step_target_pq_names = [
        str(value) for value in list(scenario_cfg.get("load_step_target_pq_names") or scenario_cfg.get("load_step_pq_names") or [])
    ]
    load_step_target_owners = [
        str(value) for value in list(scenario_cfg.get("load_step_target_owners") or scenario_cfg.get("load_step_owners") or [])
    ]

    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale

    n_ibr = int(ss.REGCV1.n)
    seed_cfg = cfg.get("seed", {})
    m_seed = np.full(n_ibr, float(seed_cfg.get("M", 4.0)), dtype=float)
    d_seed = np.full(n_ibr, float(seed_cfg.get("D", 2.0)), dtype=float)
    ss.REGCV1.M.v = m_seed.tolist()
    ss.REGCV1.D.v = d_seed.tolist()

    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0
    ss.setup()

    cfg.setdefault("features", {})
    feature_cfg = cfg["features"]
    strict_from_system = bool(feature_cfg.get("strict_from_system", True))
    if strict_from_system:
        _enforce_strict_feature_policy(feature_cfg)

    feature_contingency = _resolve_feature_contingency(
        ss,
        feature_cfg=feature_cfg,
        scenario_cfg=scenario_cfg,
    )

    x_features = list(feature_cfg.get("x_features") or [])
    model_contract_features: list[str] | None = None
    if bool(feature_cfg.get("registry_align_to_model_contract", True)):
        try:
            model_contract_features = _load_feature_contract_from_model_dir(cfg.get("model", {}))
        except Exception:
            model_contract_features = None

    if bool(feature_cfg.get("x_features_from_registry", False)):
        x_features = _load_x_features_from_registry(
            feature_cfg=feature_cfg,
            ss=ss,
            base_scale=base_scale,
            step_scale=step_scale,
            load_step_time=load_step_time,
            m_seed=m_seed,
            d_seed=d_seed,
            contingency=feature_contingency,
            model_feature_order=model_contract_features,
            load_step_target_pq_names=load_step_target_pq_names,
            load_step_target_owners=load_step_target_owners,
        )
        feature_cfg["x_features"] = x_features
        logger.info("Loaded %s x_features from feature registry.", len(x_features))
    elif bool(feature_cfg.get("x_features_from_model_dir", False)):
        x_features = _load_feature_contract_from_model_dir(cfg.get("model", {}))
        feature_cfg["x_features"] = x_features
        logger.info("Loaded %s x_features from model_dir/run_config.yaml", len(x_features))
    elif not x_features:
        raise ValueError(
            "features.x_features is empty; set x_features_from_registry=true, "
            "x_features_from_model_dir=true, or provide explicit x_features."
        )

    cfg.setdefault("model", {})
    model_cfg = cfg["model"]
    configured_in_dim = int(model_cfg.get("in_dim", 0) or 0)
    inferred_in_dim = int(len(x_features))
    if configured_in_dim != inferred_in_dim:
        logger.info(
            "Adjusting model.in_dim from %s to %s to match optimization feature contract.",
            configured_in_dim,
            inferred_in_dim,
        )
        model_cfg["in_dim"] = inferred_in_dim

    model = build_model_from_cfg(model_cfg)
    x_scaler, y_scaler = load_scalers(cfg)
    if int(getattr(x_scaler, "scale_", np.asarray([])).shape[0]) != int(len(x_features)):
        raise ValueError(
            "Scaler/model feature mismatch: x_scaler dimension does not match resolved x_features length "
            f"({int(getattr(x_scaler, 'scale_', np.asarray([])).shape[0])} vs {len(x_features)})."
        )

    if strict_from_system:
        fixed_feature_values: dict[str, float] = {}
        allow_missing_features = False
        allow_constant_fill_for_unresolved_missing = False
        fixed_source_override_non_sched = False
    else:
        fixed_feature_values = _load_missing_feature_reference_values(cfg=cfg, x_features=x_features)
        allow_missing_features = bool(cfg.get("features", {}).get("allow_missing_features", False))
        allow_constant_fill_for_unresolved_missing = bool(
            cfg.get("features", {}).get("allow_constant_fill_for_unresolved_missing", False)
        )
        fixed_source_override_non_sched = bool(
            cfg.get("features", {}).get("fixed_source_override_non_sched", False)
        )

    x_seed = _build_x_seed(
        ss,
        x_features=x_features,
        base_scale=base_scale,
        step_scale=step_scale,
        load_step_time=load_step_time,
        m_seed=m_seed,
        d_seed=d_seed,
        contingency=feature_contingency,
        feature_names_path=str(feature_cfg.get("feature_names_path", "configs/data_generation_feature_names.yaml")),
        load_step_target_pq_names=load_step_target_pq_names,
        load_step_target_owners=load_step_target_owners,
        fixed_feature_values=fixed_feature_values,
        missing_fill_value=float(cfg.get("features", {}).get("missing_fill_value", 0.0)),
        allow_missing_features=allow_missing_features,
        allow_constant_fill_for_unresolved_missing=allow_constant_fill_for_unresolved_missing,
        fixed_source_override_non_sched=fixed_source_override_non_sched,
    )
    x_seed_sc = x_seed * x_scaler.scale_ + x_scaler.min_

    pg_min = np.asarray(ss.PV.pmin.v.tolist() + ss.Slack.pmin.v.tolist(), dtype=float)
    pg_max = np.asarray(ss.PV.pmax.v.tolist() + ss.Slack.pmax.v.tolist(), dtype=float)
    pd = build_post_step_p_vector(
        ss,
        step_scale=step_scale,
        target_pq_names=load_step_target_pq_names,
        target_owner_labels=load_step_target_owners,
    )
    pg_base = np.asarray(ss.PV.p0.v.tolist() + ss.Slack.p0.v.tolist(), dtype=float)

    name_to_idx = {name: i for i, name in enumerate(x_features)}
    if "P_REGCV1_SHARE" in name_to_idx and not bool(switches.ed):
        raise ValueError(
            "P_REGCV1_SHARE is in the feature contract but constraints.use_ed=false. "
            "A linear decision-linked embedding of P_REGCV1_SHARE requires ED balance."
        )
    try:
        m_idx = [name_to_idx[f"M_{i + 1}"] for i in range(n_ibr)]
        d_idx = [name_to_idx[f"D_{i + 1}"] for i in range(n_ibr)]
    except KeyError as exc:
        raise ValueError(
            f"Missing schedulable feature in x_features: {exc}. "
            "Expected M_i and D_i entries for each REGCV1 unit."
        ) from exc

    n_genrou = int(getattr(getattr(ss, "GENROU", None), "n", 0))
    n_regcv1 = int(getattr(getattr(ss, "REGCV1", None), "n", 0))
    dispatch_feature_names = [f"P_GENROU_{i + 1}" for i in range(n_genrou)] + [
        f"P_REGCV1_{i + 1}" for i in range(n_regcv1)
    ]
    missing_dispatch = [name for name in dispatch_feature_names if name not in name_to_idx]

    # Dispatch features are optional: if the surrogate model doesn't use them (e.g., trained on only M/D),
    # then pg linking is skipped and dispatch variables are unconstrained. This is acceptable as long as
    # the ED balance constraint couples them to the power balance.
    if missing_dispatch and len(missing_dispatch) == len(dispatch_feature_names):
        # All dispatch features are missing — skip pg linking
        pg_feat_idx = []
        logger.info("Dispatch features not in x_features; skipping pg feature-to-variable linking.")
    elif missing_dispatch:
        # Partial missing — this is an error
        preview = ", ".join(missing_dispatch[:12])
        if len(missing_dispatch) > 12:
            preview += ", ..."
        raise ValueError(
            "Partial dispatch feature mismatch: some P_GENROU_i/P_REGCV1_i are in x_features but not all "
            f"({len(missing_dispatch)} missing of {len(dispatch_feature_names)}). Examples: {preview}"
        )
    else:
        pg_feat_idx = [name_to_idx[name] for name in dispatch_feature_names]
    auto_pg_link_order: tuple[int, ...] | None = None
    if cfg.get("constraints", {}).get("pg_link_order") is None:
        dispatch_n = int(pg_min.size)
        raw_ibr_pos = np.asarray(cfg.get("ibr", {}).get("indices", []), dtype=int).reshape(-1)
        ibr_pos = [int(i) for i in raw_ibr_pos.tolist() if 0 <= int(i) < dispatch_n]
        if len(set(ibr_pos)) == len(ibr_pos) and len(ibr_pos) <= dispatch_n:
            ibr_set = set(ibr_pos)
            gen_pos = [i for i in range(dispatch_n) if i not in ibr_set]
            candidate = gen_pos + ibr_pos
            if len(candidate) == dispatch_n and len(set(candidate)) == dispatch_n:
                # Feature order is [P_GENROU_1..n_genrou, P_REGCV1_1..n_regcv1].
                # Dispatch decision order is [PV..., Slack], where IBR entries are given by cfg.ibr.indices.
                # Reorder pg linkage so each feature tracks the correct generator variable.
                auto_pg_link_order = tuple(int(i) for i in candidate)
                logger.info("Auto pg_link_order from ibr.indices: %s", list(auto_pg_link_order))
    configured_pg_link_order = (
        tuple(int(i) for i in list(cfg.get("constraints", {}).get("pg_link_order") or []))
        if cfg.get("constraints", {}).get("pg_link_order") is not None
        else auto_pg_link_order
    )
    effective_pg_link_order = (
        configured_pg_link_order
        if configured_pg_link_order is not None
        else tuple(range(int(pg_min.size)))
    )

    bounds = cfg["bounds"]
    m_min_sc = _scale_subset(np.full(n_ibr, float(bounds["M_bounds"][0])), x_scaler, m_idx)
    m_max_sc = _scale_subset(np.full(n_ibr, float(bounds["M_bounds"][1])), x_scaler, m_idx)
    d_min_sc = _scale_subset(np.full(n_ibr, float(bounds["D_bounds"][0])), x_scaler, d_idx)
    d_max_sc = _scale_subset(np.full(n_ibr, float(bounds["D_bounds"][1])), x_scaler, d_idx)

    y_min_raw = np.asarray(bounds["y_min"], dtype=float)
    y_max_raw = np.asarray(bounds["y_max"], dtype=float)
    y_min_sc = y_min_raw * y_scaler.scale_ + y_scaler.min_
    y_max_sc = y_max_raw * y_scaler.scale_ + y_scaler.min_

    x_min_sc = x_seed_sc.copy()
    x_max_sc = x_seed_sc.copy()
    x_min_sc[m_idx], x_max_sc[m_idx] = m_min_sc, m_max_sc
    x_min_sc[d_idx], x_max_sc[d_idx] = d_min_sc, d_max_sc

    dispatch_raw_lo = np.zeros(len(pg_feat_idx), dtype=float)
    dispatch_raw_hi = np.zeros(len(pg_feat_idx), dtype=float)

    # Only apply dispatch bounds if pg_feat_idx is non-empty (i.e., dispatch features are in x_features)
    if pg_feat_idx:
        if len(pg_feat_idx) != int(pg_min.size):
            raise ValueError(
                "Dispatch feature link mismatch between x_features and dispatch decision size "
                f"({len(pg_feat_idx)} vs {int(pg_min.size)})."
            )
        for k, idx in enumerate(pg_feat_idx):
            src_k = int(effective_pg_link_order[k]) if k < len(effective_pg_link_order) else int(k)
            lo_raw = float(min(pg_min[src_k], pg_max[src_k]))
            hi_raw = float(max(pg_min[src_k], pg_max[src_k]))
            dispatch_raw_lo[k] = lo_raw
            dispatch_raw_hi[k] = hi_raw
            _set_feature_scaled_bounds(
                x_min_sc=x_min_sc,
                x_max_sc=x_max_sc,
                x_scaler=x_scaler,
                name_to_idx=name_to_idx,
                feature_name=dispatch_feature_names[k],
                lo_raw=lo_raw,
                hi_raw=hi_raw,
            )

    # Compute total bounds for REGCV1 (used for P_REGCV1_SHARE bounds calculation)
    if n_regcv1 > 0:
        p_regcv1_lo = float(np.sum(dispatch_raw_lo[n_genrou: n_genrou + n_regcv1]))
        p_regcv1_hi = float(np.sum(dispatch_raw_hi[n_genrou: n_genrou + n_regcv1]))
    else:
        p_regcv1_lo = 0.0
        p_regcv1_hi = 0.0

    genrou_m_fixed_arr = (
        np.asarray(getattr(getattr(ss, "GENROU", None), "M", {}).v, dtype=float)
        if getattr(ss, "GENROU", None) is not None and hasattr(ss.GENROU, "M")
        else np.zeros(0, dtype=float)
    )
    genrou_d_fixed_arr = (
        np.asarray(getattr(getattr(ss, "GENROU", None), "D", {}).v, dtype=float)
        if getattr(ss, "GENROU", None) is not None and hasattr(ss.GENROU, "D")
        else np.zeros(0, dtype=float)
    )
    n_total_md = int(n_genrou + n_regcv1)
    if n_total_md > 0:
        m_agg_lo = float((np.sum(genrou_m_fixed_arr) + (n_regcv1 * float(bounds["M_bounds"][0]))) / n_total_md)
        m_agg_hi = float((np.sum(genrou_m_fixed_arr) + (n_regcv1 * float(bounds["M_bounds"][1]))) / n_total_md)
        d_agg_lo = float((np.sum(genrou_d_fixed_arr) + (n_regcv1 * float(bounds["D_bounds"][0]))) / n_total_md)
        d_agg_hi = float((np.sum(genrou_d_fixed_arr) + (n_regcv1 * float(bounds["D_bounds"][1]))) / n_total_md)
        _set_feature_scaled_bounds(
            x_min_sc=x_min_sc,
            x_max_sc=x_max_sc,
            x_scaler=x_scaler,
            name_to_idx=name_to_idx,
            feature_name="M_agg",
            lo_raw=m_agg_lo,
            hi_raw=m_agg_hi,
        )
        _set_feature_scaled_bounds(
            x_min_sc=x_min_sc,
            x_max_sc=x_max_sc,
            x_scaler=x_scaler,
            name_to_idx=name_to_idx,
            feature_name="D_agg",
            lo_raw=d_agg_lo,
            hi_raw=d_agg_hi,
        )

    # NOTE: P_GENROU_TOTAL, P_REGCV1_TOTAL, and P_DISPATCH_TOTAL are derived from decision variables
    # via equality constraints. Do NOT set explicit bounds on these derived features — they will cause
    # infeasibility if they don't perfectly match the constraint ranges. Bounds will propagate implicitly
    # through the derived constraints from P_GENROU_i and P_REGCV1_i bounds.

    # NOTE: Reserve features (P_GENROU_RESERVE_i, P_REGCV1_RESERVE_i, reserve_p_*) are derived from
    # decision variables (P_GENROU_i, P_REGCV1_i) via the equality constraint: reserve = pmax - p[i].
    # Do NOT set explicit bounds on these derived features — let the constraint propagate bounds
    # from the decision variable bounds. Setting explicit bounds here causes infeasibility if they
    # don't exactly match the derived constraint ranges.

    dispatch_total_constant = float(np.sum(pd)) / float(step_scale)
    if abs(dispatch_total_constant) > 1e-12:
        share_lo = float(p_regcv1_lo / dispatch_total_constant)
        share_hi = float(p_regcv1_hi / dispatch_total_constant)
        _set_feature_scaled_bounds(
            x_min_sc=x_min_sc,
            x_max_sc=x_max_sc,
            x_scaler=x_scaler,
            name_to_idx=name_to_idx,
            feature_name="P_REGCV1_SHARE",
            lo_raw=min(share_lo, share_hi),
            hi_raw=max(share_lo, share_hi),
        )

    # For MILP mode, widen y bounds to the NN's actual output interval so that
    # the exact NN output (y == y_nn) never conflicts with the output box bounds.
    nn_mode = str(cfg.get("constraints", {}).get("nn_mode", "milp"))
    if nn_mode.lower() == "milp" and switches.nn:
        try:
            y_lo_nn, y_hi_nn = compute_nn_output_interval(model, x_min_sc, x_max_sc)
            y_min_sc = np.minimum(y_min_sc, y_lo_nn)
            y_max_sc = np.maximum(y_max_sc, y_hi_nn)
            logger.info("MILP y-bounds widened to NN interval: y_min_sc=%s, y_max_sc=%s", y_min_sc, y_max_sc)
        except NotImplementedError:
            logger.warning("Could not compute NN output interval; keeping original y bounds.")

    pg = cp.Variable(pg_min.size, name="pg")
    x = cp.Variable(x_seed_sc.size, name="x")
    y = cp.Variable(y_min_sc.size, name="y")

    constraint_blocks: dict[str, list[cp.Constraint]] = {}
    constraint_blocks["input"] = build_input_feature_constraints(
        x=x,
        x_seed_sc=x_seed_sc,
        m_idx=m_idx,
        d_idx=d_idx,
        m_min_sc=m_min_sc,
        m_max_sc=m_max_sc,
        d_min_sc=d_min_sc,
        d_max_sc=d_max_sc,
        pg=pg if pg_feat_idx else None,
        pg_feature_idx=pg_feat_idx if pg_feat_idx else None,
        x_scaler=x_scaler,
        pg_link_order=configured_pg_link_order if pg_feat_idx else None,
        x_feature_names=x_features,
        n_genrou=n_genrou,
        n_regcv1=n_regcv1,
        genrou_m_fixed=genrou_m_fixed_arr,
        genrou_d_fixed=genrou_d_fixed_arr,
        pg_max=pg_max if pg_feat_idx else None,
        dispatch_total_constant=dispatch_total_constant if pg_feat_idx else None,
        vis_tie_groups=cfg.get("constraints", {}).get("vis_tie_groups"),
    )

    constraint_blocks["output"] = build_output_feature_constraints(
        y=y,
        y_min_sc=y_min_sc,
        y_max_sc=y_max_sc,
        enforce_dispatch_output_link=bool(cfg.get("constraints", {}).get("enforce_dispatch_output_link", True)),
        y_ibr_idx=np.asarray(cfg.get("constraints", {}).get("y_ibr_idx", [2, 3, 4, 5]), dtype=int),
        pg=pg,
        pg_min=pg_min,
        pg_max=pg_max,
        ibr_idx=np.asarray(cfg["ibr"]["indices"], dtype=int),
        y_scaler=y_scaler,
        active_output_idx=cfg.get("constraints", {}).get(
            "output_active_idx",
            cfg.get("constraints", {}).get("nn_active_output_idx"),
        ),
    )

    nn_debug_subblocks: dict[str, list[cp.Constraint]] = {}
    if switches.nn:
        nn_payload = build_nn_constraints(
            model=model,
            x=x,
            x_seed_sc=x_seed_sc,
            x_min_sc=x_min_sc,
            x_max_sc=x_max_sc,
            mode=nn_mode,
            active_output_idx=cfg.get("constraints", {}).get("nn_active_output_idx"),
            return_blocks=True,
        )
        if len(nn_payload) == 3:
            y_nn_raw, constraints_nn, nn_debug_subblocks = nn_payload
        else:
            y_nn_raw, constraints_nn = nn_payload

        if isinstance(y_nn_raw, dict):
            y_nn_map = {int(k): v for k, v in y_nn_raw.items()}
        else:
            y_nn_map = {int(i): y_nn_raw[i] for i in range(int(y_nn_raw.shape[0]))}

        for name, cons in nn_debug_subblocks.items():
            constraint_blocks[name] = list(cons)

        nn_link_blocks: dict[str, list[cp.Constraint]] = {}
        for idx, expr in y_nn_map.items():
            nn_link_blocks[f"nn_link_{int(idx)}"] = [y[int(idx)] == expr]
        for name, cons in nn_link_blocks.items():
            constraint_blocks[name] = list(cons)

        requested_nn_blocks = list(cfg.get("constraints", {}).get("nn_enabled_subblocks") or [])
        if requested_nn_blocks:
            requested = {str(name) for name in requested_nn_blocks}
            selected_names = [
                name
                for name in [*nn_debug_subblocks.keys(), *nn_link_blocks.keys()]
                if name in requested
            ]
        else:
            selected_names = [*nn_debug_subblocks.keys(), *nn_link_blocks.keys()]

        if selected_names:
            constraint_blocks["nn"] = []
            for name in selected_names:
                constraint_blocks["nn"] += constraint_blocks[name]
        else:
            constraint_blocks["nn"] = constraints_nn + [y == y_nn_raw]
    else:
        constraint_blocks["nn"] = []

    flows = None
    line_artifacts = None
    n1_stats: dict[str, int | bool] | None = None
    n1_redispatch_stats: dict[str, int | bool] | None = None
    constraint_blocks["line"] = []
    constraint_blocks["n1"] = []
    constraint_blocks["n1_redispatch"] = []
    if switches.line:
        flows, line_constraints, line_artifacts = build_basecase_line_constraints(ss, pg=pg, pd=pd)
        constraint_blocks["line"] = line_constraints

        if switches.n1:
            exclude_islanding_critical = bool(cfg.get("constraints", {}).get("exclude_islanding_critical_n1", True))
            constraints_n1, n1_stats = build_n1_constraints(
                flows,
                line_artifacts.ppci,
                line_artifacts.ptdf,
                line_artifacts.fmax,
                critical_bus_nums=line_artifacts.critical_bus_nums,
                exclude_islanding_critical=exclude_islanding_critical,
                collect_stats=True,
            )
            logger.info("n1_stats=%s", n1_stats)
            constraint_blocks["n1"] = constraints_n1

            if bool(cfg.get("constraints", {}).get("use_n1_redispatch", False)):
                constraints_n1_r, n1_redispatch_stats, _ = build_n1_redispatch_constraints(
                    pg=pg,
                    pd=pd,
                    Cg=line_artifacts.Cg,
                    Cd=line_artifacts.Cd,
                    ptdf=line_artifacts.ptdf,
                    ppci=line_artifacts.ppci,
                    fmax=line_artifacts.fmax,
                    pg_min=pg_min,
                    pg_max=pg_max,
                    critical_bus_nums=line_artifacts.critical_bus_nums,
                    exclude_islanding_critical=exclude_islanding_critical,
                    collect_stats=True,
                )
                logger.info("n1_redispatch_stats=%s", n1_redispatch_stats)
                constraint_blocks["n1_redispatch"] = constraints_n1_r

    constraint_blocks["ed"] = build_ed_constraints(
        pg=pg,
        pg_min=pg_min,
        pg_max=pg_max,
        pd=pd,
        step_scale=step_scale,
    )

    use_n1_redispatch = switches.n1 and bool(cfg.get("constraints", {}).get("use_n1_redispatch", False))
    block_enabled = {
        "input": bool(switches.input),
        "output": bool(switches.output),
        "nn": bool(switches.nn),
        "line": bool(switches.line),
        "n1": bool(switches.n1),
        "n1_redispatch": bool(use_n1_redispatch),
        "ed": bool(switches.ed),
    }

    active_constraints: list[cp.Constraint] = []
    for key in ("input", "output", "nn", "line", "n1", "n1_redispatch", "ed"):
        if block_enabled.get(key, False):
            active_constraints += constraint_blocks.get(key, [])

    for name, block in constraint_blocks.items():
        logger.info("constraint_block=%s n_constraints=%s n_scalar=%s", name, len(block), _scalar_constraint_count(block))

    a, b, c = _resolve_ed_cost_arrays(cfg, ss)

    tie_breaker = float(cfg.get("constraints", {}).get("tie_breaker", 0.0))
    objective_dispatch = cp.sum(a + cp.multiply(b, pg) + cp.multiply(c, cp.square(pg)))
    objective_expr = objective_dispatch + (float(tie_breaker) * cp.sum(y) if tie_breaker > 0 else 0)
    objective = cp.Minimize(objective_expr)

    solver_kwargs = _build_solver_kwargs(cfg, solver.name, solver.verbose, solver.reoptimize)

    if solver.feasibility_checks:
        checks = {
            "input": constraint_blocks["input"] + (constraint_blocks["ed"] if switches.ed else []),
            "output": constraint_blocks["output"] + (constraint_blocks["ed"] if switches.ed else []),
            "nn": constraint_blocks["nn"] + (constraint_blocks["ed"] if switches.ed else []),
            "input_nn": constraint_blocks["input"] + constraint_blocks["nn"] + (constraint_blocks["ed"] if switches.ed else []),
            "input_nn_output": constraint_blocks["input"]
            + constraint_blocks["nn"]
            + constraint_blocks["output"]
            + (constraint_blocks["ed"] if switches.ed else []),
            "line": constraint_blocks["line"] + (constraint_blocks["ed"] if switches.ed else []),
            "line_n1": constraint_blocks["line"] + constraint_blocks["n1"] + (constraint_blocks["ed"] if switches.ed else []),
            "line_n1_redispatch": constraint_blocks["line"]
            + constraint_blocks["n1"]
            + constraint_blocks["n1_redispatch"]
            + (constraint_blocks["ed"] if switches.ed else []),
            "combined": active_constraints,
        }
        if nn_debug_subblocks:
            nn_base = list(nn_debug_subblocks.get("nn_shared", []))
            if nn_base:
                checks["nn_shared"] = nn_base + (constraint_blocks["ed"] if switches.ed else [])
                checks["input_nn_shared"] = constraint_blocks["input"] + nn_base + (constraint_blocks["ed"] if switches.ed else [])
            for name in sorted(nn_debug_subblocks.keys()):
                if not name.startswith("nn_head_"):
                    continue
                suffix = name.replace("nn_", "", 1)
                link_name = name.replace("nn_head_", "nn_link_", 1)
                head_constraints = nn_base + list(nn_debug_subblocks[name])
                checks[suffix] = head_constraints + (constraint_blocks["ed"] if switches.ed else [])
                checks[f"{suffix}_linked"] = (
                    constraint_blocks["input"]
                    + head_constraints
                    + list(constraint_blocks.get(link_name, []))
                    + (constraint_blocks["ed"] if switches.ed else [])
                )
        for name, cons in checks.items():
            if not cons:
                continue
            p = cp.Problem(objective, cons)
            p.solve(**solver_kwargs)
            n_constraints, nnz = _problem_stats(p, solver.name)
            logger.info(
                "check=%s status=%s objective=%s n_constraints=%s nnz=%s",
                name,
                p.status,
                p.value,
                n_constraints,
                nnz,
            )
            if (
                name == "input_nn"
                and bool(switches.output)
                and str(p.status) in {"optimal", "optimal_inaccurate"}
                and y.value is not None
            ):
                y_sc_chk = np.asarray(y.value, dtype=float).reshape(-1)
                y_lo_viol = np.maximum(y_min_sc - y_sc_chk, 0.0)
                y_hi_viol = np.maximum(y_sc_chk - y_max_sc, 0.0)
                y_viol = np.maximum(y_lo_viol, y_hi_viol)
                y_n_viol = int(np.sum(y_viol > 1e-9))
                if y_n_viol > 0:
                    y_raw_chk = (y_sc_chk - y_scaler.min_) / y_scaler.scale_
                    viol_idx = np.where(y_viol > 1e-9)[0].astype(int).tolist()
                    max_idx = int(np.argmax(y_viol))
                    logger.info(
                        "nn_only_output_bound_conflict n_viol=%s viol_indices=%s max_violation_sc=%s max_idx=%s "
                        "y_raw_at_max=%s y_raw_low=%s y_raw_high=%s",
                        y_n_viol,
                        viol_idx,
                        float(y_viol[max_idx]),
                        max_idx,
                        float(y_raw_chk[max_idx]),
                        float(y_min_raw[max_idx]),
                        float(y_max_raw[max_idx]),
                    )

    prob = cp.Problem(objective, active_constraints)
    solve_start = time.perf_counter()
    prob.solve(**solver_kwargs)
    solve_wall_sec = time.perf_counter() - solve_start
    n_constraints, nnz = _problem_stats(prob, solver.name)
    logger.info(
        "status=%s objective=%s n_constraints=%s nnz=%s solve_wall_sec=%s",
        prob.status,
        prob.value,
        n_constraints,
        nnz,
        solve_wall_sec,
    )

    if (
        str(prob.status) == "infeasible_or_unbounded"
        and str(solver.name).upper() == "GUROBI"
        and not bool(solver_kwargs.get("reoptimize", False))
    ):
        retry_kwargs = dict(solver_kwargs)
        retry_kwargs["reoptimize"] = True
        retry_start = time.perf_counter()
        prob.solve(**retry_kwargs)
        retry_wall_sec = time.perf_counter() - retry_start
        solve_wall_sec += retry_wall_sec
        n_constraints, nnz = _problem_stats(prob, solver.name)
        logger.info(
            "status_after_reoptimize=%s objective=%s n_constraints=%s nnz=%s retry_wall_sec=%s",
            prob.status,
            prob.value,
            n_constraints,
            nnz,
            retry_wall_sec,
        )

    status = str(prob.status)
    infeasible_status = {"infeasible", "infeasible_inaccurate", "unbounded", "infeasible_or_unbounded"}
    is_infeasible = status in infeasible_status

    pg_opt = (
        np.asarray(pg.value, dtype=float).reshape(-1)
        if pg.value is not None
        else np.full(pg_min.size, np.nan, dtype=float)
    )
    if x.value is not None:
        x_opt_sc = np.asarray(x.value, dtype=float).reshape(-1)
    elif is_infeasible:
        x_opt_sc = np.full(x_seed_sc.size, np.nan, dtype=float)
    else:
        x_opt_sc = x_seed_sc.copy()

    y_opt_sc = (
        np.asarray(y.value, dtype=float).reshape(-1)
        if y.value is not None
        else np.full(y_min_sc.size, np.nan, dtype=float)
    )

    m_opt = _unscale_subset(x_opt_sc[m_idx], x_scaler, m_idx) if m_idx else np.zeros(0, dtype=float)
    d_opt = _unscale_subset(x_opt_sc[d_idx], x_scaler, d_idx) if d_idx else np.zeros(0, dtype=float)
    y_opt = (y_opt_sc - y_scaler.min_) / y_scaler.scale_

    _save_results_csv(
        output_paths["results_csv"],
        status=status,
        objective=float(prob.value) if prob.value is not None else None,
        pg=pg_opt,
        m=m_opt,
        d=d_opt,
    )

    block_rows = []
    block_row_order = ["input", "output", "nn", "line", "n1", "n1_redispatch", "ed"]
    block_row_order += sorted(name for name in constraint_blocks if name.startswith("nn_") and name not in {"nn"})
    for name in block_row_order:
        cons = constraint_blocks.get(name, [])
        block_rows.append(
            {
                "block": name,
                "enabled": int(bool(block_enabled.get(name, False) or (name.startswith("nn_") and switches.nn and len(cons) > 0))),
                "n_constraints": int(len(cons)),
                "n_scalar_constraints": int(_scalar_constraint_count(cons)),
            }
        )
    _write_dict_rows_csv(output_paths["constraint_blocks_csv"], block_rows)

    output_names = list(cfg.get("outputs", {}).get("y_names") or [f"y_{i + 1}" for i in range(int(y_opt.size))])
    if len(output_names) != int(y_opt.size):
        output_names = [f"y_{i + 1}" for i in range(int(y_opt.size))]
    pred_rows = []
    for i, name in enumerate(output_names):
        val = float(y_opt[i]) if i < y_opt.size else np.nan
        lo = float(y_min_raw[i]) if i < y_min_raw.size else np.nan
        hi = float(y_max_raw[i]) if i < y_max_raw.size else np.nan
        sat = int(np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi) and (val >= lo - 1e-8) and (val <= hi + 1e-8))
        pred_rows.append(
            {
                "output_idx": int(i),
                "output_name": name,
                "predicted_value": val,
                "limit_low": lo,
                "limit_high": hi,
                "is_within_limits": sat,
                "lower_margin": (val - lo) if np.isfinite(val) and np.isfinite(lo) else np.nan,
                "upper_margin": (hi - val) if np.isfinite(val) and np.isfinite(hi) else np.nan,
            }
        )
    _write_dict_rows_csv(output_paths["predicted_metrics_csv"], pred_rows)

    ibr_idx = np.asarray(cfg["ibr"]["indices"], dtype=int)
    dp_dispatch = pg_opt[ibr_idx] - pg_base[ibr_idx] if ibr_idx.size else np.zeros(0, dtype=float)
    headroom_up = pg_max[ibr_idx] - pg_opt[ibr_idx] if ibr_idx.size else np.zeros(0, dtype=float)
    headroom_down = pg_opt[ibr_idx] - pg_min[ibr_idx] if ibr_idx.size else np.zeros(0, dtype=float)
    y_ibr_idx = np.asarray(cfg.get("constraints", {}).get("y_ibr_idx", [2, 3, 4, 5]), dtype=int)

    dispatch_rows = []
    for i in range(pg_opt.size):
        dispatch_rows.append(
            {
                "row_type": "generator_dispatch",
                "index": int(i),
                "pg_base": float(pg_base[i]),
                "pg_opt": float(pg_opt[i]),
                "pg_delta": float(pg_opt[i] - pg_base[i]),
                "pg_min": float(pg_min[i]),
                "pg_max": float(pg_max[i]),
            }
        )
    for local_i, gen_i in enumerate(ibr_idx):
        pred_dp = float(y_opt[y_ibr_idx[local_i]]) if (local_i < y_ibr_idx.size and y_ibr_idx[local_i] < y_opt.size) else np.nan
        pmin_delta = float(pg_min[gen_i] - pg_base[gen_i])
        pmax_delta = float(pg_max[gen_i] - pg_base[gen_i])
        dispatch_rows.append(
            {
                "row_type": "ibr_summary",
                "index": int(local_i),
                "gen_index": int(gen_i),
                "M_opt": float(m_opt[local_i]) if local_i < m_opt.size else np.nan,
                "D_opt": float(d_opt[local_i]) if local_i < d_opt.size else np.nan,
                "delta_p_dispatch": float(dp_dispatch[local_i]),
                "delta_p_predicted": pred_dp,
                "delta_p_pred_minus_dispatch": pred_dp - float(dp_dispatch[local_i]) if np.isfinite(pred_dp) else np.nan,
                "delta_p_physical_min": pmin_delta,
                "delta_p_physical_max": pmax_delta,
                "delta_p_margin_to_max": (pmax_delta - pred_dp) if np.isfinite(pred_dp) else np.nan,
                "delta_p_margin_to_min": (pred_dp - pmin_delta) if np.isfinite(pred_dp) else np.nan,
                "headroom_up": float(headroom_up[local_i]),
                "headroom_down": float(headroom_down[local_i]),
            }
        )
    _write_dict_rows_csv(output_paths["dispatch_impact_csv"], dispatch_rows)

    line_security = {}
    if line_artifacts is not None and np.all(np.isfinite(pg_opt)):
        line_security = evaluate_line_security_metrics(
            pg_value=pg_opt,
            pd=pd,
            artifacts=line_artifacts,
            include_n1=bool(switches.n1),
            exclude_islanding_critical=bool(cfg.get("constraints", {}).get("exclude_islanding_critical_n1", True)),
        )

    dispatch_objective_value = (
        float(np.sum(a + b * pg_opt + c * np.square(pg_opt)))
        if np.all(np.isfinite(pg_opt))
        else np.nan
    )

    metrics_arr = np.asarray([row["is_within_limits"] for row in pred_rows], dtype=float) if pred_rows else np.zeros(0)
    output_limits_ok = bool(np.all(metrics_arr > 0.5)) if metrics_arr.size else False
    output_violation_count = int(np.sum(metrics_arr < 0.5)) if metrics_arr.size else 0

    size_metrics = prob.size_metrics
    problem_size = {
        **_variable_type_counts(prob),
        "n_constraints_total": int(size_metrics.num_scalar_eq_constr + size_metrics.num_scalar_leq_constr),
        "n_constraints_scalar_total": int(
            sum(_scalar_constraint_count(constraint_blocks[name]) for name in constraint_blocks if block_enabled.get(name, False))
        ),
        "n_constraints_eq": int(size_metrics.num_scalar_eq_constr),
        "n_constraints_leq": int(size_metrics.num_scalar_leq_constr),
        "nnz_total": nnz,
    }

    solver_stats = {
        "solver_name": solver.name,
        "status": status,
        "solve_time_sec": float(getattr(prob.solver_stats, "solve_time", np.nan)),
        "solve_wall_time_sec": float(solve_wall_sec),
        "num_iters": int(getattr(prob.solver_stats, "num_iters", -1)) if getattr(prob.solver_stats, "num_iters", None) is not None else -1,
        "solver_kwargs": {k: v for k, v in solver_kwargs.items() if k != "solver"},
    }

    summary_payload: dict[str, Any] = {
        "run_id": run_id,
        "formulation_id": formulation_id,
        "formulation_name": formulation_name,
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "scenario_description": scenario_description,
        "description": str(formulation_cfg.get("description", "")),
        "equation_map": equation_map,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "objective": float(prob.value) if prob.value is not None else None,
        "objective_dispatch_only": dispatch_objective_value if np.isfinite(dispatch_objective_value) else None,
        "objective_tie_breaker": float(tie_breaker),
        "system_case": str(cfg["system"]["case"]),
        "model_type": str(cfg["model"].get("type", "")),
        "model_dir": str(cfg["model"].get("model_dir", cfg["model"].get("state_dict", ""))),
        "model_retain_reason": str(cfg["model"].get("retain_reason", "")),
        "nn_mode": nn_mode,
        "solver_modeling_mode": f"{solver.name}:{nn_mode if switches.nn else 'disabled'}",
        "config_path": str(config_path) if config_path else "",
        "scenario": dict(cfg.get("scenario", {})),
        "solver_stats": solver_stats,
        "problem_size": problem_size,
        "constraint_blocks": block_rows,
        "constraint_switches": {
            "use_input": bool(switches.input),
            "use_output": bool(switches.output),
            "use_nn": bool(switches.nn),
            "use_line": bool(switches.line),
            "use_n1": bool(switches.n1),
            "use_n1_redispatch": bool(use_n1_redispatch),
            "use_ed": bool(switches.ed),
            "vis_tie_groups": list(cfg.get("constraints", {}).get("vis_tie_groups") or []),
            "nn_active_output_idx": list(cfg.get("constraints", {}).get("nn_active_output_idx") or []),
            "output_active_idx": list(
                cfg.get("constraints", {}).get(
                    "output_active_idx",
                    cfg.get("constraints", {}).get("nn_active_output_idx") or [],
                )
            ),
            "nn_enabled_subblocks": list(cfg.get("constraints", {}).get("nn_enabled_subblocks") or []),
        },
        "n1_stats": n1_stats or {},
        "n1_redispatch_stats": n1_redispatch_stats or {},
        "security_checks": {
            "line_security": line_security,
            "predicted_outputs_within_limits": bool(output_limits_ok),
            "predicted_output_violation_count": int(output_violation_count),
        },
        "dispatch_summary": {
            "pg_base": pg_base.tolist(),
            "pg_opt": pg_opt.tolist(),
            "pg_delta": (pg_opt - pg_base).tolist(),
            "m_opt": m_opt.tolist(),
            "d_opt": d_opt.tolist(),
            "delta_p_dispatch_ibr": dp_dispatch.tolist(),
            "headroom_up_ibr": headroom_up.tolist(),
            "headroom_down_ibr": headroom_down.tolist(),
        },
        "predicted_metrics": {
            "names": output_names,
            "values": y_opt.tolist(),
            "values_scaled": y_opt_sc.tolist(),
            "limits_low": y_min_raw.tolist(),
            "limits_high": y_max_raw.tolist(),
        },
        "artifacts": {
            "results_csv": str(output_paths["results_csv"]),
            "summary_csv": str(output_paths["summary_csv"]),
            "constraint_blocks_csv": str(output_paths["constraint_blocks_csv"]),
            "predicted_metrics_csv": str(output_paths["predicted_metrics_csv"]),
            "dispatch_impact_csv": str(output_paths["dispatch_impact_csv"]),
        },
        "wall_runtime_sec": float(time.perf_counter() - wall_start),
    }

    _write_json(output_paths["summary_json"], summary_payload)
    _write_dict_rows_csv(output_paths["summary_csv"], [_summary_row_from_payload(summary_payload)])

    if plot_options.enabled and np.all(np.isfinite(pg_opt)) and np.all(np.isfinite(x_opt_sc)) and np.all(np.isfinite(y_opt_sc)):
        try:
            from optimization.plots import maybe_make_plots

            maybe_make_plots(
                options=plot_options,
                model=model,
                x_opt_sc=x_opt_sc,
                y_opt_sc=y_opt_sc,
                y_scaler=y_scaler,
                m_opt=m_opt,
                d_opt=d_opt,
                ibr_idx=np.asarray(cfg["ibr"]["indices"], dtype=int),
                pg_opt=pg_opt,
                ss=ss,
                pd=pd,
            )
        except Exception as exc:
            logger.warning("plotting_failed=%s", exc)

    if is_infeasible and bool(output_cfg.get("fail_on_infeasible", False)):
        raise RuntimeError(f"Optimization failed with status={status}")

    return {
        "run_id": run_id,
        "formulation_id": formulation_id,
        "status": status,
        "objective": float(prob.value) if prob.value is not None else None,
        "objective_dispatch_only": dispatch_objective_value if np.isfinite(dispatch_objective_value) else None,
        "summary_json": str(output_paths["summary_json"]),
        "summary_csv": str(output_paths["summary_csv"]),
        "results_csv": str(output_paths["results_csv"]),
        "pg_opt": pg_opt,
        "m_opt": m_opt,
        "d_opt": d_opt,
        "y_opt": y_opt,
        "y_opt_sc": y_opt_sc,
    }


def main():
    parser = argparse.ArgumentParser(description="Final optimization problem with separated constraint blocks.")
    parser.add_argument(
        "--config",
        default="results/thesis_optimization_results/configs/base_optimization.yaml",
        help="Path to optimization config YAML.",
    )
    args = parser.parse_args()

    cfg = load_optimization_config(args.config)
    res = run_optimization(cfg, config_path=str(Path(args.config).resolve()))
    print(
        f"[scheduling.problem] status={res['status']} "
        f"objective={res['objective']} summary={res['summary_json']}"
    )


if __name__ == "__main__":
    main()
