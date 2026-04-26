# extract_metrics.py
# Feature/target extraction utilities for simulation outputs and operating snapshots.

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from data_generation import line_utils as lu


DEFAULT_FEATURE_NAMES_PATH = "configs/shared/data_generation_feature_names.yaml"
_NUMERIC_SUFFIX_RE = re.compile(r"(\d+)\s*$")
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z_]+")


def _resolve_repo_path(path_str: Optional[str], default_rel: str) -> Path:
    """Helper to resolve repo path."""
    path = Path(path_str) if path_str else Path(default_rel)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


@lru_cache(maxsize=None)
def load_feature_name_config(path_str: Optional[str] = None) -> Dict[str, object]:
    """Load the feature-name configuration mapping."""
    path = _resolve_repo_path(path_str, DEFAULT_FEATURE_NAMES_PATH)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid feature-name config at {path}: expected a mapping.")
    return payload


def _schema_fields(
    feature_names_path: Optional[str],
    section: str,
    field_key: str,
) -> List[str]:
    """Helper to schema fields."""
    schema = load_feature_name_config(feature_names_path)
    section_cfg = schema.get(section, {})
    if not isinstance(section_cfg, dict):
        raise RuntimeError(f"Invalid feature-name config section '{section}'.")
    values = list(section_cfg.get(field_key, []) or [])
    return [str(value) for value in values]


def _schema_prefix(
    feature_names_path: Optional[str],
    section: str,
    prefix_key: str,
) -> str:
    """Helper to schema prefix."""
    schema = load_feature_name_config(feature_names_path)
    section_cfg = schema.get(section, {})
    if not isinstance(section_cfg, dict):
        raise RuntimeError(f"Invalid feature-name config section '{section}'.")
    prefixes = section_cfg.get("prefixes", {})
    if not isinstance(prefixes, dict):
        raise RuntimeError(f"Invalid feature-name config prefixes for section '{section}'.")
    prefix = prefixes.get(prefix_key)
    if prefix is None:
        raise RuntimeError(f"Missing prefix '{prefix_key}' in feature-name config section '{section}'.")
    return str(prefix)


def _ordered_unique(values: Sequence[str]) -> List[str]:
    """Helper to ordered unique."""
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _as_float_array(values: Optional[Sequence[float]]) -> np.ndarray:
    """Helper to as float array."""
    if values is None:
        return np.zeros(0, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr if arr.size else np.zeros(0, dtype=float)


def _sum_or_zero(values: Optional[Sequence[float]]) -> float:
    """Helper to sum or zero."""
    arr = _as_float_array(values)
    return float(np.nansum(arr)) if arr.size else 0.0


def _sum_or_nan(values: Optional[Sequence[float]]) -> float:
    """Helper to sum or nan."""
    arr = _as_float_array(values)
    return float(np.nansum(arr)) if arr.size else np.nan


def _min_or_nan(values: Optional[Sequence[float]]) -> float:
    """Helper to min or nan."""
    arr = _as_float_array(values)
    return float(np.nanmin(arr)) if arr.size else np.nan


def _max_or_nan(values: Optional[Sequence[float]]) -> float:
    """Helper to max or nan."""
    arr = _as_float_array(values)
    return float(np.nanmax(arr)) if arr.size else np.nan


def _mean_or_nan(values: Optional[Sequence[float]]) -> float:
    """Helper to mean or nan."""
    arr = _as_float_array(values)
    return float(np.nanmean(arr)) if arr.size else np.nan


def _std_or_nan(values: Optional[Sequence[float]]) -> float:
    """Helper to std or nan."""
    arr = _as_float_array(values)
    return float(np.nanstd(arr)) if arr.size else np.nan


def _to_float_or_nan(value) -> float:
    """Helper to float or nan."""
    return lu._to_float_or_nan(value)


def _normalize_plotter_indices(indices) -> List[int]:
    """Helper to normalize plotter indices."""
    if indices is None:
        return []
    if isinstance(indices, (list, tuple, np.ndarray)):
        return [int(i) for i in indices]
    return [int(indices)]


def _plotter_channel_names(plotter, indices: Optional[Sequence[int]] = None) -> List[str]:
    """Helper to plotter channel names."""
    names = [str(value) for value in list(getattr(plotter, "_uname", []))]
    if indices is None:
        return names
    out: List[str] = []
    for idx in indices:
        if 0 <= int(idx) < len(names):
            out.append(names[int(idx)])
    return out


def _plotter_series_matrix(plotter, indices: Sequence[int]) -> np.ndarray:
    """Helper to plotter series matrix."""
    idx = _normalize_plotter_indices(indices)
    if not idx:
        return np.zeros((0, 0), dtype=float)
    values = np.asarray(plotter.get_values(idx), dtype=float)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.shape[1] == len(idx):
        return values
    if values.shape[0] == len(idx):
        return values.transpose()
    return values


def _plotter_channel_indices_by_prefix(plotter, prefix: str) -> List[int]:
    """Helper to plotter channel indices by prefix."""
    names = _plotter_channel_names(plotter)
    return [i for i, name in enumerate(names) if name.startswith(prefix)]


def _plotter_channel_indices_by_prefixes(plotter, prefixes: Sequence[str]) -> List[int]:
    """Helper to plotter channel indices by prefixes."""
    for prefix in prefixes:
        indices = _plotter_channel_indices_by_prefix(plotter, prefix)
        if indices:
            return indices
    return []


def _extract_numeric_suffix(name: str) -> Optional[int]:
    """Helper to extract numeric suffix."""
    match = _NUMERIC_SUFFIX_RE.search(str(name))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _plotter_matrix_with_names(plotter) -> tuple[np.ndarray, List[str]]:
    """Helper to plotter matrix with names."""
    names = _plotter_channel_names(plotter)
    if not names:
        return np.zeros((0, 0), dtype=float), []

    values = np.asarray(getattr(plotter, "_data", []), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        return np.zeros((0, len(names)), dtype=float), names
    if values.shape[1] == len(names):
        return values, names
    if values.shape[0] == len(names):
        return values.transpose(), names
    raise RuntimeError(
        f"Plotter shape mismatch: values={values.shape}, channels={len(names)}."
    )


def _sanitize_feature_token(value: str) -> str:
    """Helper to sanitize feature token."""
    text = _NON_ALNUM_RE.sub("_", str(value)).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = "unnamed"
    if text[0].isdigit():
        text = f"channel_{text}"
    return text


def _build_initial_state_fieldnames(
    channel_names: Sequence[str],
    feature_names_path: Optional[str] = None,
) -> List[str]:
    """Helper to build initial state fieldnames."""
    prefix = _schema_prefix(feature_names_path, "x_op", "initial_state")
    counts: Dict[str, int] = {}
    fieldnames: List[str] = []
    for channel_name in channel_names:
        base_name = f"{prefix}{_sanitize_feature_token(channel_name)}"
        counts[base_name] = counts.get(base_name, 0) + 1
        if counts[base_name] > 1:
            fieldnames.append(f"{base_name}__{counts[base_name]}")
        else:
            fieldnames.append(base_name)
    return fieldnames


def initial_state_fieldnames_from_plotter(
    plotter,
    *,
    feature_names_path: Optional[str] = None,
) -> List[str]:
    """Build initial-state fieldnames from plotter channel names."""
    return _build_initial_state_fieldnames(
        _plotter_channel_names(plotter),
        feature_names_path=feature_names_path,
    )


def extract_initial_state_metrics(
    plotter,
    *,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract initial state metrics."""
    values, names = _plotter_matrix_with_names(plotter)
    if values.size == 0 or not names:
        return {}

    first_row = values[0]
    fieldnames = _build_initial_state_fieldnames(names, feature_names_path=feature_names_path)
    out: Dict[str, float] = {}
    for name, value in zip(fieldnames, first_row):
        out[name] = float(value) if np.isfinite(value) else np.nan
    return out


def _sum_model_attr(ss, model_names: Sequence[str], attr_candidates: Sequence[str]) -> float:
    """Helper to sum model attr."""
    total = 0.0
    found = False
    for model_name in model_names:
        model = getattr(ss, model_name, None)
        if model is None or getattr(model, "n", 0) <= 0:
            continue
        for attr in attr_candidates:
            values = getattr(getattr(model, attr, None), "v", None)
            if values is None:
                continue
            total += float(np.nansum(np.asarray(values, dtype=float)))
            found = True
            break
    return total if found else np.nan


def _selected_pq_records(
    *,
    ss,
    selected_pq_names: Optional[Sequence[str]],
    pq_owners: Optional[Sequence[str]],
    pq_p_before: Optional[Sequence[float]],
    pq_q_before: Optional[Sequence[float]] = None,
    pq_p_after: Optional[Sequence[float]] = None,
) -> List[Dict[str, float | str]]:
    """Helper to selected pq records."""
    actual_names = [str(value) for value in list(getattr(getattr(ss.PQ, "name", None), "v", []))]
    if not actual_names:
        return []

    selected_names = [str(value) for value in list(selected_pq_names or actual_names)]
    owner_values = [str(value) for value in list(pq_owners or [])]
    if len(owner_values) < len(actual_names):
        owner_values.extend([""] * (len(actual_names) - len(owner_values)))

    p_before = _as_float_array(pq_p_before)
    q_before = _as_float_array(pq_q_before)
    p_after = _as_float_array(pq_p_after)

    records_by_name: Dict[str, Dict[str, float | str]] = {}
    for idx, name in enumerate(actual_names):
        records_by_name[name] = {
            "name": name,
            "owner": owner_values[idx] if idx < len(owner_values) else "",
            "p_before": float(p_before[idx]) if idx < p_before.size else np.nan,
            "q_before": float(q_before[idx]) if idx < q_before.size else np.nan,
            "p_after": float(p_after[idx]) if idx < p_after.size else np.nan,
        }

    selected_records: List[Dict[str, float | str]] = []
    for name in selected_names:
        record = records_by_name.get(name)
        if record is not None:
            selected_records.append(record)
    return selected_records


def simulation_row_fieldnames(
    *,
    pq_names: Sequence[str],
    owner_labels: Sequence[str],
    n_ibr: int,
    n_genrou: int,
    line_uids: Sequence[int],
    bus_numbers: Sequence[int],
    initial_state_fields: Optional[Sequence[str]] = None,
    include_plotter: Optional[bool] = None,
    feature_names_path: Optional[str] = None,
) -> List[str]:
    """Build CSV fieldnames for one simulation row."""
    _ = include_plotter

    metadata_fields = _schema_fields(feature_names_path, "metadata", "fields")
    x_op_fields = _schema_fields(feature_names_path, "x_op", "scalar_fields")
    x_cont_load_fields = _schema_fields(feature_names_path, "x_cont", "load_scalar_fields")
    x_cont_line_identity_fields = _schema_fields(feature_names_path, "x_cont", "line_identity_fields")
    x_cont_line_flow_fields = _schema_fields(feature_names_path, "x_cont", "line_flow_fields")
    x_cont_line_bus_fields = _schema_fields(feature_names_path, "x_cont", "line_bus_fields")
    x_cont_line_severity_fields = _schema_fields(feature_names_path, "x_cont", "line_severity_fields")
    x_sched_fields = _schema_fields(feature_names_path, "x_sched", "scalar_fields")
    y_coi_fields = _schema_fields(feature_names_path, "y", "coi_fields")
    y_bus_frequency_fields = _schema_fields(feature_names_path, "y", "bus_frequency_summary_fields")
    y_bus_voltage_fields = _schema_fields(feature_names_path, "y", "bus_voltage_summary_fields")
    diagnostic_fields = _schema_fields(feature_names_path, "diagnostics", "fields")

    pq_p_base_prefix = _schema_prefix(feature_names_path, "x_op", "pq_p_base")
    pq_q_base_prefix = _schema_prefix(feature_names_path, "x_op", "pq_q_base")
    pq_delta_prefix = _schema_prefix(feature_names_path, "x_cont", "pq_delta_p")
    owner_delta_prefix = _schema_prefix(feature_names_path, "x_cont", "owner_delta_p")
    line_one_hot_prefix = _schema_prefix(feature_names_path, "x_cont", "line_one_hot")
    m_prefix = _schema_prefix(feature_names_path, "x_sched", "M")
    d_prefix = _schema_prefix(feature_names_path, "x_sched", "D")
    genrou_prefix = _schema_prefix(feature_names_path, "x_sched", "p_genrou")
    regcv1_prefix = _schema_prefix(feature_names_path, "x_sched", "p_regcv1")
    genrou_reserve_prefix = _schema_prefix(feature_names_path, "x_sched", "p_genrou_reserve")
    regcv1_reserve_prefix = _schema_prefix(feature_names_path, "x_sched", "p_regcv1_reserve")
    ibr_peak_prefix = _schema_prefix(feature_names_path, "y", "delta_p_ibr")
    ibr_peak_abs_prefix = _schema_prefix(feature_names_path, "y", "delta_p_ibr_abs")
    bus_freq_prefix = _schema_prefix(feature_names_path, "y", "bus_freq_max_abs_dev")
    bus_v_prefix = _schema_prefix(feature_names_path, "y", "bus_v_max_abs_dev")
    bus_rocof_prefix = _schema_prefix(feature_names_path, "y", "bus_rocof_max_abs")

    fields: List[str] = list(metadata_fields)
    fields.extend(x_op_fields)
    for pq_name in pq_names:
        fields.append(f"{pq_p_base_prefix}{pq_name}")
        fields.append(f"{pq_q_base_prefix}{pq_name}")
    fields.extend(initial_state_fields or [])

    fields.extend(x_cont_load_fields)
    for pq_name in pq_names:
        fields.append(f"{pq_delta_prefix}{pq_name}")
    for owner in owner_labels:
        fields.append(f"{owner_delta_prefix}{owner}")

    fields.extend(x_cont_line_identity_fields)
    fields.extend(x_cont_line_flow_fields)
    fields.extend(x_cont_line_bus_fields)
    fields.extend(x_cont_line_severity_fields)
    for line_uid in line_uids:
        fields.append(f"{line_one_hot_prefix}{int(line_uid)}")

    fields.extend(x_sched_fields)
    for i in range(1, n_ibr + 1):
        fields.append(f"{m_prefix}{i}")
        fields.append(f"{d_prefix}{i}")
    for i in range(1, n_genrou + 1):
        fields.append(f"{genrou_prefix}{i}")
        fields.append(f"{genrou_reserve_prefix}{i}")
    for i in range(1, n_ibr + 1):
        fields.append(f"{regcv1_prefix}{i}")
        fields.append(f"{regcv1_reserve_prefix}{i}")
        fields.append(f"{ibr_peak_prefix}{i}")
        fields.append(f"{ibr_peak_abs_prefix}{i}")

    fields.extend(y_coi_fields)
    fields.extend(y_bus_frequency_fields)
    fields.extend(y_bus_voltage_fields)
    for bus in bus_numbers:
        fields.append(f"{bus_freq_prefix}{int(bus)}")
    for bus in bus_numbers:
        fields.append(f"{bus_v_prefix}{int(bus)}")
    for bus in bus_numbers:
        fields.append(f"{bus_rocof_prefix}{int(bus)}")
    fields.extend(diagnostic_fields)

    return _ordered_unique(fields)


def compute_freq_metrics(t, f, f0=50, r=None, tol_hz=0.01):
    """Compute freq metrics."""
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    if t.size == 0 or f.size == 0:
        return {}
    if t.size != f.size:
        raise ValueError("Time and frequency arrays must have the same length.")

    if r is None:
        if t.size > 1:
            rocof = np.gradient(f, t, edge_order=2 if t.size > 2 else 1)
        else:
            rocof = np.zeros_like(f)
    else:
        rocof = np.asarray(r, dtype=float)
        if rocof.size != f.size:
            raise ValueError("ROCOF array must have the same length as frequency array.")

    tail_len = max(10, t.size // 10)
    f_ss = np.mean(f[-tail_len:])

    idx_min = int(np.argmin(f))
    idx_max = int(np.argmax(f))
    f_min = f[idx_min]
    f_max = f[idx_max]

    within_band = np.abs(f - f_ss) <= tol_hz
    suffix_ok = np.logical_and.accumulate(within_band[::-1])[::-1]
    t_settle = t[np.argmax(suffix_ok)] if np.any(suffix_ok) else np.nan

    idx_r_min = int(np.argmin(rocof))
    idx_r_max = int(np.argmax(rocof))
    idx_r_abs = int(np.argmax(np.abs(rocof)))

    return {
        "f_ss": float(f_ss),
        "f_min": float(f_min),
        "t_min": float(t[idx_min]),
        "f_max": float(f_max),
        "t_max": float(t[idx_max]),
        "dev_down": float(f0 - f_min),
        "dev_up": float(f_max - f0),
        "max_abs_dev": float(max(abs(f_min - f0), abs(f_max - f0))),
        "t_settle": float(t_settle) if np.isfinite(t_settle) else np.nan,
        "rocof_min": float(rocof[idx_r_min]),
        "t_rocof_min": float(t[idx_r_min]),
        "rocof_max": float(rocof[idx_r_max]),
        "t_rocof_max": float(t[idx_r_max]),
        "rocof_max_abs": float(np.max(np.abs(rocof))),
        "t_rocof_max_abs": float(t[idx_r_abs]),
        "rocof_mean": float(np.mean(rocof)),
        "rocof_rms": float(np.sqrt(np.mean(rocof**2))),
    }


def extract_ibr_peak_metrics(
    plotter,
    *,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract ibr peak metrics."""
    signed_prefix = _schema_prefix(feature_names_path, "y", "delta_p_ibr")
    abs_prefix = _schema_prefix(feature_names_path, "y", "delta_p_ibr_abs")

    indices = _normalize_plotter_indices(plotter.find("Pe REGCV1", idx_only=True))
    if not indices:
        return {}

    names = _plotter_channel_names(plotter, indices)
    p_matrix = _plotter_series_matrix(plotter, indices)
    peaks: Dict[str, float] = {}
    for col, name in enumerate(names):
        unit_id = _extract_numeric_suffix(name) or (col + 1)
        series = p_matrix[:, col]
        if series.size == 0:
            continue
        baseline = float(series[0])
        delta = series - baseline
        abs_idx = int(np.argmax(np.abs(delta)))
        peaks[f"{signed_prefix}{unit_id}"] = float(delta[abs_idx])
        peaks[f"{abs_prefix}{unit_id}"] = float(np.max(np.abs(delta)))
    return peaks


def extract_ibr_peaks(plotter) -> Dict[str, float]:
    """Extract ibr peaks."""
    metrics = extract_ibr_peak_metrics(plotter)
    return {key: value for key, value in metrics.items() if "_abs_" not in key}


def extract_line_metrics(
    ss,
    contingency: Optional[Dict[str, object]],
    line_uids: Optional[Sequence[int]],
    *,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract line metrics."""
    line_one_hot_prefix = _schema_prefix(feature_names_path, "x_cont", "line_one_hot")
    line_identity_fields = _schema_fields(feature_names_path, "x_cont", "line_identity_fields")
    line_flow_fields = _schema_fields(feature_names_path, "x_cont", "line_flow_fields")
    line_bus_fields = _schema_fields(feature_names_path, "x_cont", "line_bus_fields")
    line_severity_fields = _schema_fields(feature_names_path, "x_cont", "line_severity_fields")

    line_uids = list(line_uids or [])
    base_mva = float(ss.config.mva)
    cont_uid = int(contingency["uid"]) if contingency is not None and contingency.get("uid") is not None else None
    out: Dict[str, float] = {}

    one_hot = lu._identity_one_hot(line_uids, cont_uid)
    for key, value in one_hot.items():
        raw_uid = _extract_numeric_suffix(key)
        if raw_uid is None:
            continue
        out[f"{line_one_hot_prefix}{raw_uid}"] = value

    ratings = lu._line_ratings_from_pandapower(ss)
    records = lu._line_records(ss)
    by_uid = {int(record["uid"]): record for record in records if record.get("uid") is not None}
    out.update(lu._global_stress(ss, ratings, base_mva))

    if cont_uid is None:
        for field in line_identity_fields:
            out[field] = -1
        for field in line_flow_fields:
            out[field] = -1
        for field in line_bus_fields:
            out[field] = -1
        for field in line_severity_fields:
            out[field] = -1
        return out

    record = by_uid.get(cont_uid, contingency)
    line_rating_raw = lu._valid_rating(record.get("rating"))
    if not np.isfinite(line_rating_raw):
        line_rating_raw = lu._valid_rating(record.get("Sn"))
    if not np.isfinite(line_rating_raw) and ratings is not None and cont_uid < len(ratings):
        line_rating_raw = lu._valid_rating(ratings[cont_uid])
    line_rating = lu._rating_to_pu(line_rating_raw, base_mva)

    p_from = lu._line_flow_component(
        ss,
        cont_uid,
        (("Pij", "v"), ("p1", "v"), ("P1", "v"), ("pf", "v"), ("a1", "e")),
    )
    pre_fault_loading = (
        abs(p_from) / line_rating * 100.0
        if np.isfinite(p_from) and np.isfinite(line_rating) and line_rating > 0
        else np.nan
    )
    out["line_rating"] = _to_float_or_nan(line_rating)
    out["pre_fault_flow"] = _to_float_or_nan(p_from)
    out["pre_fault_loading"] = _to_float_or_nan(pre_fault_loading)

    out.update(lu._line_parameters(record))

    flow = lu._line_prefault_flows(ss, cont_uid)
    p_from_abs = abs(flow.get("pre_p_from", np.nan)) if np.isfinite(flow.get("pre_p_from", np.nan)) else np.nan
    p_to_abs = abs(flow.get("pre_p_to", np.nan)) if np.isfinite(flow.get("pre_p_to", np.nan)) else np.nan
    flow["pre_loading_from"] = (
        _to_float_or_nan(p_from_abs / line_rating * 100.0)
        if np.isfinite(p_from_abs) and np.isfinite(line_rating) and line_rating > 0
        else np.nan
    )
    flow["pre_loading_to"] = (
        _to_float_or_nan(p_to_abs / line_rating * 100.0)
        if np.isfinite(p_to_abs) and np.isfinite(line_rating) and line_rating > 0
        else np.nan
    )
    p0 = flow.get("pre_p_from", np.nan)
    flow["pre_flow_direction_p"] = _to_float_or_nan(np.sign(p0)) if np.isfinite(p0) else np.nan
    out.update(flow)

    bus1 = _to_float_or_nan(record.get("bus1"))
    bus2 = _to_float_or_nan(record.get("bus2"))
    v_from, a_from = lu._bus_state(ss, bus1)
    v_to, a_to = lu._bus_state(ss, bus2)
    out.update(
        {
            "pre_v_from": _to_float_or_nan(v_from),
            "pre_v_to": _to_float_or_nan(v_to),
            "pre_theta_from": _to_float_or_nan(a_from),
            "pre_theta_to": _to_float_or_nan(a_to),
            "pre_delta_theta": _to_float_or_nan(a_from - a_to)
            if np.isfinite(a_from) and np.isfinite(a_to)
            else np.nan,
        }
    )
    out.update(lu._topology_criticality(records, cont_uid, bus1, bus2))
    out.update(lu._dc_sensitivity(ss, cont_uid, bus1, bus2, base_mva))
    return out


def _default_ed_metadata() -> Dict[str, float | str]:
    """Helper to default ed metadata."""
    return {
        "ed_enabled": 0,
        "ed_solver": "",
        "ed_status": "",
        "ed_total_cost": np.nan,
        "ed_constant_cost": np.nan,
        "ed_energy_cost": np.nan,
        "ed_reserve_cost": np.nan,
        "ed_quadratic_cost": np.nan,
    }


def extract_row_metadata(
    *,
    contingency: Optional[Dict[str, object]],
    load_step_enabled: bool,
    load_step_time: float,
    trip_time: float,
    ed_meta: Optional[Mapping[str, object]] = None,
) -> Dict[str, float | str]:
    """Extract row metadata."""
    metadata = _default_ed_metadata()
    if ed_meta:
        metadata.update(ed_meta)

    if contingency is None:
        metadata.update(
            {
                "cont_type": "load" if load_step_enabled else "none",
                "contingency_time": float(load_step_time) if load_step_enabled else np.nan,
                "line_uid": -1,
                "line_from_bus": -1,
                "line_to_bus": -1,
            }
        )
        return metadata

    bus1_val = contingency.get("bus1")
    bus2_val = contingency.get("bus2")
    metadata.update(
        {
            "cont_type": "line_plus_load" if load_step_enabled else "line",
            "contingency_time": float(trip_time if trip_time is not None else np.nan),
            "line_uid": int(contingency.get("uid")) if contingency.get("uid") is not None else -1,
            "line_from_bus": int(bus1_val) if bus1_val is not None and np.isfinite(float(bus1_val)) else -1,
            "line_to_bus": int(bus2_val) if bus2_val is not None and np.isfinite(float(bus2_val)) else -1,
        }
    )
    return metadata


def extract_operating_point_snapshot(ss) -> Dict[str, float]:
    """Extract operating point snapshot."""
    base_mva = float(getattr(getattr(ss, "config", None), "mva", np.nan))
    ratings = lu._line_ratings_from_pandapower(ss)
    out = dict(lu._global_stress(ss, ratings, base_mva))
    reserve_metrics = _generator_reserve_metrics(ss)

    q_load_values = getattr(getattr(ss.PQ, "Qpf", None), "v", None)
    if q_load_values is None:
        q_load_values = getattr(getattr(ss.PQ, "q0", None), "v", None)
    out["total_load_q_prefault"] = _sum_or_nan(q_load_values)
    out["total_gen_q_prefault"] = _sum_model_attr(
        ss,
        model_names=("PV", "Slack"),
        attr_candidates=("Qg", "q", "q0"),
    )
    out["reserve_p_total_prefault"] = float(
        reserve_metrics["reserve_p_genrou"] + reserve_metrics["reserve_p_ibr"]
    )
    out["reserve_q_total_prefault"] = float(
        reserve_metrics["reserve_q_genrou"] + reserve_metrics["reserve_q_ibr"]
    )
    out.pop("reserve_proxy_prefault", None)

    bus_v = _as_float_array(getattr(getattr(ss.Bus, "v", None), "v", None))
    bus_a = _as_float_array(getattr(getattr(ss.Bus, "a", None), "v", None))
    out.update(
        {
            "bus_v_min_prefault": _min_or_nan(bus_v),
            "bus_v_max_prefault": _max_or_nan(bus_v),
            "bus_v_mean_prefault": _mean_or_nan(bus_v),
            "bus_v_std_prefault": _std_or_nan(bus_v),
            "bus_v_max_abs_dev_prefault": float(np.nanmax(np.abs(bus_v - 1.0))) if bus_v.size else np.nan,
            "bus_angle_min_prefault": _min_or_nan(bus_a),
            "bus_angle_max_prefault": _max_or_nan(bus_a),
            "bus_angle_spread_prefault": float(np.nanmax(bus_a) - np.nanmin(bus_a)) if bus_a.size else np.nan,
            "n_buses": int(getattr(ss.Bus, "n", 0)),
            "n_lines": int(getattr(ss.Line, "n", 0)),
            "n_pq_loads": int(getattr(ss.PQ, "n", 0)),
            "n_genrou": int(getattr(getattr(ss, "GENROU", None), "n", 0)),
            "n_regcv1": int(getattr(getattr(ss, "REGCV1", None), "n", 0)),
        }
    )
    return out


def extract_x_op(
    *,
    ss,
    base_load_scale: float,
    pq_names: Optional[Sequence[str]],
    pq_owners: Optional[Sequence[str]],
    pq_p_before: Optional[Sequence[float]],
    pq_q_before: Optional[Sequence[float]] = None,
    operating_point_snapshot: Optional[Dict[str, float]] = None,
    initial_state_snapshot: Optional[Mapping[str, float]] = None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x_op features."""
    pq_p_base_prefix = _schema_prefix(feature_names_path, "x_op", "pq_p_base")
    pq_q_base_prefix = _schema_prefix(feature_names_path, "x_op", "pq_q_base")

    snapshot = dict(operating_point_snapshot or extract_operating_point_snapshot(ss))
    reserve_metrics = _generator_reserve_metrics(ss)
    records = _selected_pq_records(
        ss=ss,
        selected_pq_names=pq_names,
        pq_owners=pq_owners,
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
    )

    out: Dict[str, float] = {
        "base_load_scale": float(base_load_scale),
        "base_load_p_total": _sum_or_zero(pq_p_before),
        "base_load_q_total": _sum_or_zero(pq_q_before),
        "reserve_p_genrou": float(reserve_metrics["reserve_p_genrou"]),
        "reserve_p_ibr": float(reserve_metrics["reserve_p_ibr"]),
        "reserve_q_genrou": float(reserve_metrics["reserve_q_genrou"]),
        "reserve_q_ibr": float(reserve_metrics["reserve_q_ibr"]),
    }
    for record in records:
        name = str(record["name"])
        out[f"{pq_p_base_prefix}{name}"] = float(record["p_before"])
        out[f"{pq_q_base_prefix}{name}"] = float(record["q_before"])

    out.update(snapshot)
    if initial_state_snapshot:
        out.update({str(key): float(value) if np.isfinite(value) else np.nan for key, value in initial_state_snapshot.items()})
    return out


def extract_x_cont_load_mismatch(
    *,
    ss,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Optional[Sequence[str]],
    pq_owners: Optional[Sequence[str]],
    pq_p_before: Optional[Sequence[float]],
    pq_p_after: Optional[Sequence[float]],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x cont load mismatch."""
    pq_delta_prefix = _schema_prefix(feature_names_path, "x_cont", "pq_delta_p")
    owner_delta_prefix = _schema_prefix(feature_names_path, "x_cont", "owner_delta_p")

    records = _selected_pq_records(
        ss=ss,
        selected_pq_names=pq_names,
        pq_owners=pq_owners,
        pq_p_before=pq_p_before,
        pq_p_after=pq_p_after,
    )

    out: Dict[str, float] = {
        "load_step_scale": float(load_step_scale),
        "load_step_time": float(load_step_time),
        "DELTA_PQ_tot": 0.0,
    }

    delta_p_total = 0.0
    owner_totals: Dict[str, float] = {}
    for record in records:
        name = str(record["name"])
        owner = str(record["owner"])
        p_before = float(record["p_before"])
        p_after = float(record["p_after"])
        delta_p = p_after - p_before if np.isfinite(p_before) and np.isfinite(p_after) else np.nan
        out[f"{pq_delta_prefix}{name}"] = float(delta_p) if np.isfinite(delta_p) else np.nan
        if np.isfinite(delta_p):
            delta_p_total += delta_p
            owner_totals[owner] = owner_totals.get(owner, 0.0) + delta_p

    for owner, total in owner_totals.items():
        out[f"{owner_delta_prefix}{owner}"] = float(total)

    out["DELTA_PQ_tot"] = float(delta_p_total)
    return out


def extract_x_cont_line_identity(
    *,
    line_metrics: Mapping[str, float],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x cont line identity."""
    fields = _schema_fields(feature_names_path, "x_cont", "line_identity_fields")
    one_hot_prefix = _schema_prefix(feature_names_path, "x_cont", "line_one_hot")
    out = {key: line_metrics.get(key, np.nan) for key in fields}
    out.update({key: value for key, value in line_metrics.items() if str(key).startswith(one_hot_prefix)})
    return out


def extract_x_cont_line_flow(
    *,
    line_metrics: Mapping[str, float],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x cont line flow."""
    fields = _schema_fields(feature_names_path, "x_cont", "line_flow_fields")
    return {key: line_metrics.get(key, np.nan) for key in fields}


def extract_x_cont_line_bus(
    *,
    line_metrics: Mapping[str, float],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x cont line bus."""
    fields = _schema_fields(feature_names_path, "x_cont", "line_bus_fields")
    return {key: line_metrics.get(key, np.nan) for key in fields}


def extract_x_cont_line_severity(
    *,
    line_metrics: Mapping[str, float],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x cont line severity."""
    fields = _schema_fields(feature_names_path, "x_cont", "line_severity_fields")
    return {key: line_metrics.get(key, np.nan) for key in fields}


def extract_x_cont(
    *,
    ss,
    contingency: Optional[Dict[str, object]],
    load_step_scale: float,
    load_step_time: float,
    pq_names: Optional[Sequence[str]],
    pq_owners: Optional[Sequence[str]],
    pq_p_before: Optional[Sequence[float]],
    pq_p_after: Optional[Sequence[float]],
    line_uids: Optional[Sequence[int]] = None,
    line_metrics_snapshot: Optional[Dict[str, float]] = None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract x_cont features."""
    line_metrics = line_metrics_snapshot or extract_line_metrics(
        ss=ss,
        contingency=contingency,
        line_uids=line_uids,
        feature_names_path=feature_names_path,
    )
    return {
        "load_mismatch": extract_x_cont_load_mismatch(
            ss=ss,
            load_step_scale=load_step_scale,
            load_step_time=load_step_time,
            pq_names=pq_names,
            pq_owners=pq_owners,
            pq_p_before=pq_p_before,
            pq_p_after=pq_p_after,
            feature_names_path=feature_names_path,
        ),
        "line_identity": extract_x_cont_line_identity(
            line_metrics=line_metrics,
            feature_names_path=feature_names_path,
        ),
        "line_flow": extract_x_cont_line_flow(
            line_metrics=line_metrics,
            feature_names_path=feature_names_path,
        ),
        "line_bus": extract_x_cont_line_bus(
            line_metrics=line_metrics,
            feature_names_path=feature_names_path,
        ),
        "line_severity": extract_x_cont_line_severity(
            line_metrics=line_metrics,
            feature_names_path=feature_names_path,
        ),
    }


def _current_schedule_aggregates(ss) -> tuple[float, float]:
    """Helper to current schedule aggregates."""
    gen_m = _as_float_array(getattr(getattr(ss.GENROU, "M", None), "v", None))
    gen_d = _as_float_array(getattr(getattr(ss.GENROU, "D", None), "v", None))
    ibr_m = _as_float_array(getattr(getattr(ss.REGCV1, "M", None), "v", None))
    ibr_d = _as_float_array(getattr(getattr(ss.REGCV1, "D", None), "v", None))

    m_all = np.concatenate([gen_m, ibr_m]) if (gen_m.size or ibr_m.size) else np.array([0.0], dtype=float)
    d_all = np.concatenate([gen_d, ibr_d]) if (gen_d.size or ibr_d.size) else np.array([0.0], dtype=float)
    return float(np.mean(m_all)), float(np.mean(d_all))


def _derive_dispatch_vectors(ss) -> tuple[np.ndarray, np.ndarray]:
    """Helper to derive dispatch vectors."""
    genrou_pg = _as_float_array(getattr(ss.GENROU, "Pg", np.zeros(0)))
    if genrou_pg.size == 0 and hasattr(ss.GENROU, "p0"):
        genrou_pg = _as_float_array(ss.GENROU.p0.v)

    regcv1_pg = np.zeros(int(getattr(ss.REGCV1, "n", 0)), dtype=float)
    if hasattr(ss.REGCV1, "pref"):
        regcv1_pg = _as_float_array(ss.REGCV1.pref.v)
    return genrou_pg, regcv1_pg


def _model_attr_array(model, attr_candidates: Sequence[str], *, fill_value: float = np.nan) -> np.ndarray:
    """Helper to model attr array."""
    model_n = int(getattr(model, "n", 0))
    if model_n <= 0:
        return np.zeros(0, dtype=float)
    for attr in attr_candidates:
        values = getattr(getattr(model, attr, None), "v", None)
        if values is not None:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if arr.size == model_n:
                return arr
    return np.full(model_n, fill_value, dtype=float)


def _static_generator_attr_vector(
    ss,
    *,
    attr_candidates: Sequence[str],
    fill_value: float = np.nan,
) -> np.ndarray:
    """Helper to static generator attr vector."""
    arrays: List[np.ndarray] = []
    for model_name in ("PV", "Slack"):
        model = getattr(ss, model_name, None)
        if model is None or int(getattr(model, "n", 0)) <= 0:
            continue
        arrays.append(_model_attr_array(model, attr_candidates, fill_value=fill_value))
    if not arrays:
        return np.zeros(0, dtype=float)
    return np.concatenate(arrays)


def _dispatch_ibr_positions(ss, n_dispatch: int) -> List[int]:
    """Helper to dispatch ibr positions."""
    gen_values = getattr(getattr(ss.REGCV1, "gen", None), "v", None)
    positions: List[int] = []
    if gen_values is not None:
        for value in gen_values:
            try:
                position = int(value) - 1
            except Exception:
                continue
            if 0 <= position < n_dispatch and position not in positions:
                positions.append(position)
    if positions:
        return positions

    fallback_count = min(int(getattr(ss.REGCV1, "n", 0)), n_dispatch)
    return list(range(fallback_count))


def _select_positions(values: np.ndarray, positions: Sequence[int], *, expected_len: int) -> np.ndarray:
    """Helper to select positions."""
    out = np.full(expected_len, np.nan, dtype=float)
    if values.size == 0 or expected_len <= 0:
        return out
    for idx, position in enumerate(list(positions)[:expected_len]):
        if 0 <= int(position) < values.size:
            out[idx] = float(values[int(position)])
    return out


def _generator_reserve_metrics(ss) -> Dict[str, object]:
    """Helper to generator reserve metrics."""
    p_dispatch = _static_generator_attr_vector(ss, attr_candidates=("Pg", "p", "p0"))
    p_max = _static_generator_attr_vector(ss, attr_candidates=("pmax",))
    q_dispatch = _static_generator_attr_vector(ss, attr_candidates=("Qg", "q", "q0"))
    q_max = _static_generator_attr_vector(ss, attr_candidates=("qmax", "Qmax"))

    p_reserve = np.where(
        np.isfinite(p_dispatch) & np.isfinite(p_max),
        np.maximum(p_max - p_dispatch, 0.0),
        np.nan,
    )
    q_reserve = np.where(
        np.isfinite(q_dispatch) & np.isfinite(q_max),
        np.maximum(q_max - q_dispatch, 0.0),
        np.nan,
    )

    n_dispatch = p_dispatch.size
    ibr_positions = _dispatch_ibr_positions(ss, n_dispatch)
    ibr_position_set = set(ibr_positions)
    genrou_positions = [idx for idx in range(n_dispatch) if idx not in ibr_position_set]

    n_regcv1 = int(getattr(ss.REGCV1, "n", 0))
    n_genrou = int(getattr(getattr(ss, "GENROU", None), "n", 0))

    regcv1_p_reserve = _select_positions(p_reserve, ibr_positions, expected_len=n_regcv1)
    genrou_p_reserve = _select_positions(p_reserve, genrou_positions, expected_len=n_genrou)
    regcv1_q_reserve = _select_positions(q_reserve, ibr_positions, expected_len=n_regcv1)
    genrou_q_reserve = _select_positions(q_reserve, genrou_positions, expected_len=n_genrou)

    return {
        "reserve_p_genrou": float(np.nansum(genrou_p_reserve)) if genrou_p_reserve.size else np.nan,
        "reserve_p_ibr": float(np.nansum(regcv1_p_reserve)) if regcv1_p_reserve.size else np.nan,
        "reserve_q_genrou": float(np.nansum(genrou_q_reserve)) if genrou_q_reserve.size else np.nan,
        "reserve_q_ibr": float(np.nansum(regcv1_q_reserve)) if regcv1_q_reserve.size else np.nan,
        "genrou_p_reserve": genrou_p_reserve,
        "regcv1_p_reserve": regcv1_p_reserve,
    }


def extract_x_sched(
    *,
    ss,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    genrou_pg: Optional[Sequence[float]] = None,
    regcv1_pg: Optional[Sequence[float]] = None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract x_sched features."""
    m_prefix = _schema_prefix(feature_names_path, "x_sched", "M")
    d_prefix = _schema_prefix(feature_names_path, "x_sched", "D")
    genrou_prefix = _schema_prefix(feature_names_path, "x_sched", "p_genrou")
    regcv1_prefix = _schema_prefix(feature_names_path, "x_sched", "p_regcv1")
    genrou_reserve_prefix = _schema_prefix(feature_names_path, "x_sched", "p_genrou_reserve")
    regcv1_reserve_prefix = _schema_prefix(feature_names_path, "x_sched", "p_regcv1_reserve")

    if genrou_pg is None or regcv1_pg is None:
        genrou_pg_values, regcv1_pg_values = _derive_dispatch_vectors(ss)
    else:
        genrou_pg_values = _as_float_array(genrou_pg)
        regcv1_pg_values = _as_float_array(regcv1_pg)

    reserve_metrics = _generator_reserve_metrics(ss)
    genrou_p_reserve = _as_float_array(reserve_metrics.get("genrou_p_reserve"))
    regcv1_p_reserve = _as_float_array(reserve_metrics.get("regcv1_p_reserve"))

    m_agg, d_agg = _current_schedule_aggregates(ss)
    gen_total = _sum_or_zero(genrou_pg_values)
    reg_total = _sum_or_zero(regcv1_pg_values)
    dispatch_total = gen_total + reg_total

    out: Dict[str, float] = {
        "M_agg": float(m_agg),
        "D_agg": float(d_agg),
        "P_GENROU_TOTAL": float(gen_total),
        "P_REGCV1_TOTAL": float(reg_total),
        "P_DISPATCH_TOTAL": float(dispatch_total),
        "P_REGCV1_SHARE": float(reg_total / dispatch_total)
        if np.isfinite(dispatch_total) and abs(dispatch_total) > 0
        else np.nan,
    }

    for i, (m_value, d_value) in enumerate(zip(M_vec, D_vec), start=1):
        out[f"{m_prefix}{i}"] = float(m_value)
        out[f"{d_prefix}{i}"] = float(d_value)
    for i, value in enumerate(genrou_pg_values, start=1):
        out[f"{genrou_prefix}{i}"] = float(value)
    for i, value in enumerate(genrou_p_reserve, start=1):
        out[f"{genrou_reserve_prefix}{i}"] = float(value) if np.isfinite(value) else np.nan
    for i, value in enumerate(regcv1_pg_values, start=1):
        out[f"{regcv1_prefix}{i}"] = float(value)
    for i, value in enumerate(regcv1_p_reserve, start=1):
        out[f"{regcv1_reserve_prefix}{i}"] = float(value) if np.isfinite(value) else np.nan
    return out


def extract_coi_dynamic_metrics(
    ss,
    *,
    plotter=None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract coi dynamic metrics."""
    fields = _schema_fields(feature_names_path, "y", "coi_fields")
    if plotter is None:
        ss.TDS.load_plotter()
        plotter = ss.TDS.plotter

    time = _as_float_array(plotter.get_values(0))
    coi_indices = _normalize_plotter_indices(plotter.find("omega COI", idx_only=True))
    if not coi_indices:
        return {key: np.nan for key in fields}

    f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
    f_coi = _plotter_series_matrix(plotter, coi_indices).reshape(-1) * f0
    rocof = np.gradient(f_coi, float(ss.TDS.config.tstep), axis=0)
    metrics = compute_freq_metrics(time, f=f_coi, f0=f0, r=rocof)
    if not metrics:
        return {key: np.nan for key in fields}

    dev_down = float(metrics.get("dev_down", np.nan))
    dev_up = float(metrics.get("dev_up", np.nan))
    f_min = float(metrics.get("f_min", np.nan))
    f_max = float(metrics.get("f_max", np.nan))

    time_of_max_dev = np.nan
    if np.isfinite(dev_down) and np.isfinite(dev_up):
        if dev_down >= dev_up:
            dev_signed = f_min - f0
            time_of_max_dev = float(metrics.get("t_min", np.nan))
        else:
            dev_signed = f_max - f0
            time_of_max_dev = float(metrics.get("t_max", np.nan))
    else:
        dev_signed = np.nan

    rocof_min = float(metrics.get("rocof_min", np.nan))
    rocof_max = float(metrics.get("rocof_max", np.nan))
    if np.isfinite(rocof_min) and np.isfinite(rocof_max):
        rocof_signed = rocof_min if abs(rocof_min) >= abs(rocof_max) else rocof_max
    else:
        rocof_signed = np.nan

    return {
        "time_max_dev": float(time_of_max_dev) if np.isfinite(time_of_max_dev) else np.nan,
        "time_max_rocof": float(metrics.get("t_rocof_max_abs", np.nan)),
        "f_ss_COI": float(metrics.get("f_ss", np.nan)),
        "f_min_COI": f_min,
        "f_max_COI": f_max,
        "t_settle_COI": float(metrics.get("t_settle", np.nan)),
        "rocof_COI": float(rocof_signed) if np.isfinite(rocof_signed) else np.nan,
        "rocof_abs_COI": float(metrics.get("rocof_max_abs", np.nan)),
        "rocof_mean_COI": float(metrics.get("rocof_mean", np.nan)),
        "rocof_rms_COI": float(metrics.get("rocof_rms", np.nan)),
        "dev_COI": float(dev_signed) if np.isfinite(dev_signed) else np.nan,
        "dev_abs_COI": float(metrics.get("max_abs_dev", np.nan)),
    }


def extract_bus_frequency_metrics(
    *,
    plotter,
    f0: float,
    bus_numbers: Sequence[int],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract bus frequency metrics."""
    summary_fields = _schema_fields(feature_names_path, "y", "bus_frequency_summary_fields")
    per_bus_prefix = _schema_prefix(feature_names_path, "y", "bus_freq_max_abs_dev")
    rocof_per_bus_prefix = _schema_prefix(feature_names_path, "y", "bus_rocof_max_abs")

    out: Dict[str, float] = {key: np.nan for key in summary_fields}
    for bus in bus_numbers:
        out[f"{per_bus_prefix}{int(bus)}"] = np.nan
        out[f"{rocof_per_bus_prefix}{int(bus)}"] = np.nan

    time = _as_float_array(plotter.get_values(0))
    if time.size < 2:
        return out

    direct_rocof_indices = _plotter_channel_indices_by_prefixes(
        plotter,
        ("Wf_y BusROCOF ",),
    )
    freq_indices = _plotter_channel_indices_by_prefixes(
        plotter,
        ("f BusROCOF ", "f BusFreq "),
    )
    if not freq_indices and not direct_rocof_indices:
        return out

    freq_names: List[str] = []
    f_matrix = np.empty((0, 0), dtype=float)
    if freq_indices:
        freq_names = _plotter_channel_names(plotter, freq_indices)
        f_matrix = _plotter_series_matrix(plotter, freq_indices)
        if f_matrix.size == 0:
            freq_names = []

    direct_rocof_by_bus: Dict[int, np.ndarray] = {}
    if direct_rocof_indices:
        rocof_names = _plotter_channel_names(plotter, direct_rocof_indices)
        rocof_matrix = _plotter_series_matrix(plotter, direct_rocof_indices)
        if rocof_matrix.size:
            for col, name in enumerate(rocof_names):
                bus = _extract_numeric_suffix(name)
                if bus is None:
                    continue
                rocof = rocof_matrix[:, col]
                if rocof.size != time.size:
                    continue
                direct_rocof_by_bus[int(bus)] = rocof * float(f0)

    per_bus_max_abs: List[float] = []
    bus_mins: List[float] = []
    bus_maxs: List[float] = []
    rocof_per_bus_max_abs: List[float] = []
    rocof_mins: List[float] = []
    rocof_maxs: List[float] = []
    edge_order = 2 if time.size > 2 else 1

    for col, name in enumerate(freq_names):
        bus = _extract_numeric_suffix(name)
        if bus is None:
            continue
        series_pu = f_matrix[:, col]
        if series_pu.size == 0 or series_pu.size != time.size:
            continue
        series_hz = series_pu * float(f0)
        deviation_hz = series_hz - float(f0)
        max_abs = float(np.nanmax(np.abs(deviation_hz))) if series_hz.size else np.nan
        out[f"{per_bus_prefix}{bus}"] = max_abs
        if np.isfinite(max_abs):
            per_bus_max_abs.append(max_abs)
        if series_hz.size:
            bus_mins.append(float(np.nanmin(series_hz)))
            bus_maxs.append(float(np.nanmax(series_hz)))

        rocof = direct_rocof_by_bus.get(int(bus))
        if rocof is None:
            rocof = np.gradient(series_hz, time, edge_order=edge_order)
        rocof_max_abs = float(np.nanmax(np.abs(rocof))) if rocof.size else np.nan
        out[f"{rocof_per_bus_prefix}{bus}"] = rocof_max_abs
        if np.isfinite(rocof_max_abs):
            rocof_per_bus_max_abs.append(rocof_max_abs)
        if rocof.size:
            rocof_mins.append(float(np.nanmin(rocof)))
            rocof_maxs.append(float(np.nanmax(rocof)))

    for bus, rocof in direct_rocof_by_bus.items():
        key = f"{rocof_per_bus_prefix}{bus}"
        if np.isfinite(out.get(key, np.nan)):
            continue
        rocof_max_abs = float(np.nanmax(np.abs(rocof))) if rocof.size else np.nan
        out[key] = rocof_max_abs
        if np.isfinite(rocof_max_abs):
            rocof_per_bus_max_abs.append(rocof_max_abs)
        if rocof.size:
            rocof_mins.append(float(np.nanmin(rocof)))
            rocof_maxs.append(float(np.nanmax(rocof)))

    if per_bus_max_abs:
        values = np.asarray(per_bus_max_abs, dtype=float)
        out["bus_freq_max_abs_dev_any"] = float(np.nanmax(values))
        out["bus_freq_mean_max_abs_dev"] = float(np.nanmean(values))
        out["bus_freq_p95_max_abs_dev"] = float(np.nanpercentile(values, 95))
        out["bus_freq_n_buses_over_0p2"] = int(np.sum(values > 0.2))
        out["bus_freq_n_buses_over_0p5"] = int(np.sum(values > 0.5))
    if bus_mins:
        out["bus_freq_nadir_min"] = float(np.nanmin(np.asarray(bus_mins, dtype=float)))
    if bus_maxs:
        out["bus_freq_zenith_max"] = float(np.nanmax(np.asarray(bus_maxs, dtype=float)))
    if rocof_per_bus_max_abs:
        values = np.asarray(rocof_per_bus_max_abs, dtype=float)
        out["bus_rocof_max_abs_any"] = float(np.nanmax(values))
        out["bus_rocof_mean_max_abs"] = float(np.nanmean(values))
        out["bus_rocof_p95_max_abs"] = float(np.nanpercentile(values, 95))
    if rocof_mins:
        out["bus_rocof_min_any"] = float(np.nanmin(np.asarray(rocof_mins, dtype=float)))
    if rocof_maxs:
        out["bus_rocof_max_any"] = float(np.nanmax(np.asarray(rocof_maxs, dtype=float)))
    return out


def extract_bus_voltage_metrics(
    *,
    plotter,
    bus_numbers: Sequence[int],
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract bus voltage metrics."""
    summary_fields = _schema_fields(feature_names_path, "y", "bus_voltage_summary_fields")
    per_bus_prefix = _schema_prefix(feature_names_path, "y", "bus_v_max_abs_dev")

    out: Dict[str, float] = {key: np.nan for key in summary_fields}
    for bus in bus_numbers:
        out[f"{per_bus_prefix}{int(bus)}"] = np.nan

    indices = _plotter_channel_indices_by_prefix(plotter, "v Bus ")
    if not indices:
        return out

    names = _plotter_channel_names(plotter, indices)
    v_matrix = _plotter_series_matrix(plotter, indices)
    if v_matrix.size == 0:
        return out

    per_bus_max_abs: List[float] = []
    bus_mins: List[float] = []
    bus_maxs: List[float] = []

    for col, name in enumerate(names):
        bus = _extract_numeric_suffix(name)
        if bus is None:
            continue
        series = v_matrix[:, col]
        if series.size == 0:
            continue
        baseline = float(series[0])
        max_abs = float(np.nanmax(np.abs(series - baseline)))
        out[f"{per_bus_prefix}{bus}"] = max_abs
        per_bus_max_abs.append(max_abs)
        bus_mins.append(float(np.nanmin(series)))
        bus_maxs.append(float(np.nanmax(series)))

    if per_bus_max_abs:
        values = np.asarray(per_bus_max_abs, dtype=float)
        out["bus_v_max_abs_dev_any"] = float(np.nanmax(values))
        out["bus_v_mean_max_abs_dev"] = float(np.nanmean(values))
        out["bus_v_p95_max_abs_dev"] = float(np.nanpercentile(values, 95))
    if bus_mins:
        min_values = np.asarray(bus_mins, dtype=float)
        out["bus_v_min_any"] = float(np.nanmin(min_values))
        out["bus_v_n_buses_below_0p95"] = int(np.sum(min_values < 0.95))
    if bus_maxs:
        max_values = np.asarray(bus_maxs, dtype=float)
        out["bus_v_max_any"] = float(np.nanmax(max_values))
        out["bus_v_n_buses_above_1p05"] = int(np.sum(max_values > 1.05))
    return out


def extract_y_metrics(
    *,
    ss,
    plotter=None,
    bus_numbers: Optional[Sequence[int]] = None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract y-target metrics."""
    if plotter is None:
        ss.TDS.load_plotter()
        plotter = ss.TDS.plotter

    bus_ids = [int(value) for value in list(bus_numbers or getattr(getattr(ss.Bus, "idx", None), "v", []))]
    f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
    return {
        "coi": extract_coi_dynamic_metrics(
            ss,
            plotter=plotter,
            feature_names_path=feature_names_path,
        ),
        "ibr_response": extract_ibr_peak_metrics(
            plotter,
            feature_names_path=feature_names_path,
        ),
        "bus_frequency": extract_bus_frequency_metrics(
            plotter=plotter,
            f0=f0,
            bus_numbers=bus_ids,
            feature_names_path=feature_names_path,
        ),
        "bus_voltage": extract_bus_voltage_metrics(
            plotter=plotter,
            bus_numbers=bus_ids,
            feature_names_path=feature_names_path,
        ),
    }


def extract_feature_blocks(
    *,
    ss,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Optional[Sequence[str]],
    pq_owners: Optional[Sequence[str]],
    pq_p_before: Optional[Sequence[float]],
    pq_p_after: Optional[Sequence[float]],
    pq_q_before: Optional[Sequence[float]] = None,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    genrou_pg: Optional[Sequence[float]] = None,
    regcv1_pg: Optional[Sequence[float]] = None,
    contingency: Optional[Dict[str, object]] = None,
    load_step_enabled: bool = False,
    trip_time: float = np.nan,
    line_uids: Optional[Sequence[int]] = None,
    line_metrics_snapshot: Optional[Dict[str, float]] = None,
    operating_point_snapshot: Optional[Dict[str, float]] = None,
    initial_state_snapshot: Optional[Mapping[str, float]] = None,
    include_initial_state: bool = True,
    ed_meta: Optional[Mapping[str, object]] = None,
    plotter=None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Extract feature blocks."""
    if plotter is None:
        ss.TDS.load_plotter()
        plotter = ss.TDS.plotter

    if include_initial_state:
        if initial_state_snapshot is None:
            initial_state = extract_initial_state_metrics(
                plotter,
                feature_names_path=feature_names_path,
            )
        else:
            initial_state = dict(initial_state_snapshot)
    else:
        initial_state = {}

    return {
        "metadata": extract_row_metadata(
            contingency=contingency,
            load_step_enabled=load_step_enabled,
            load_step_time=load_step_time,
            trip_time=trip_time,
            ed_meta=ed_meta,
        ),
        "x_op": extract_x_op(
            ss=ss,
            base_load_scale=base_load_scale,
            pq_names=pq_names,
            pq_owners=pq_owners,
            pq_p_before=pq_p_before,
            pq_q_before=pq_q_before,
            operating_point_snapshot=operating_point_snapshot,
            initial_state_snapshot=initial_state,
            feature_names_path=feature_names_path,
        ),
        "x_cont": extract_x_cont(
            ss=ss,
            contingency=contingency,
            load_step_scale=load_step_scale,
            load_step_time=load_step_time,
            pq_names=pq_names,
            pq_owners=pq_owners,
            pq_p_before=pq_p_before,
            pq_p_after=pq_p_after,
            line_uids=line_uids,
            line_metrics_snapshot=line_metrics_snapshot,
            feature_names_path=feature_names_path,
        ),
        "x_sched": extract_x_sched(
            ss=ss,
            M_vec=M_vec,
            D_vec=D_vec,
            genrou_pg=genrou_pg,
            regcv1_pg=regcv1_pg,
            feature_names_path=feature_names_path,
        ),
        "y": extract_y_metrics(
            ss=ss,
            plotter=plotter,
            bus_numbers=list(getattr(getattr(ss.Bus, "idx", None), "v", [])),
            feature_names_path=feature_names_path,
        ),
    }


def extract_simulation_row(
    *,
    ss,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Optional[Sequence[str]] = None,
    pq_owners: Optional[Sequence[str]] = None,
    pq_p_before: Optional[Sequence[float]] = None,
    pq_p_after: Optional[Sequence[float]] = None,
    pq_q_before: Optional[Sequence[float]] = None,
    pq_q_after: Optional[Sequence[float]] = None,
    base_load_q_total: Optional[float] = None,
    M_vec: Sequence[float] = (),
    D_vec: Sequence[float] = (),
    genrou_pg: Optional[Sequence[float]] = None,
    regcv1_pg: Optional[Sequence[float]] = None,
    success: bool,
    contingency: Optional[Dict[str, object]] = None,
    load_step_enabled: bool = False,
    trip_time: float = np.nan,
    line_uids: Optional[Sequence[int]] = None,
    line_metrics_snapshot: Optional[Dict[str, float]] = None,
    operating_point_snapshot: Optional[Dict[str, float]] = None,
    initial_state_snapshot: Optional[Mapping[str, float]] = None,
    include_initial_state: bool = True,
    ed_meta: Optional[Mapping[str, object]] = None,
    plotter=None,
    plotter_csv: Optional[str] = None,
    feature_names_path: Optional[str] = None,
) -> Dict[str, float]:
    """Extract simulation row."""
    _ = pq_q_after
    _ = base_load_q_total
    _ = plotter_csv

    if pq_names is None:
        pq_names = list(getattr(getattr(ss.PQ, "name", None), "v", []))
    if pq_owners is None:
        pq_owners = [str(value) for value in list(getattr(getattr(ss.PQ, "owner", None), "v", []))]

    blocks = extract_feature_blocks(
        ss=ss,
        base_load_scale=base_load_scale,
        load_step_scale=load_step_scale,
        load_step_time=load_step_time,
        pq_names=pq_names,
        pq_owners=pq_owners,
        pq_p_before=pq_p_before,
        pq_p_after=pq_p_after,
        pq_q_before=pq_q_before,
        M_vec=M_vec,
        D_vec=D_vec,
        genrou_pg=genrou_pg,
        regcv1_pg=regcv1_pg,
        contingency=contingency,
        load_step_enabled=load_step_enabled,
        trip_time=trip_time,
        line_uids=line_uids,
        line_metrics_snapshot=line_metrics_snapshot,
        operating_point_snapshot=operating_point_snapshot,
        initial_state_snapshot=initial_state_snapshot,
        include_initial_state=include_initial_state,
        ed_meta=ed_meta,
        plotter=plotter,
        feature_names_path=feature_names_path,
    )

    row: Dict[str, float] = {}
    row.update(blocks["metadata"])
    row.update(blocks["x_op"])
    for section in ("load_mismatch", "line_identity", "line_flow", "line_bus", "line_severity"):
        row.update(blocks["x_cont"].get(section, {}))
    row.update(blocks["x_sched"])
    for section in ("coi", "ibr_response", "bus_frequency", "bus_voltage"):
        row.update(blocks["y"].get(section, {}))

    row["success"] = bool(success)
    return row
