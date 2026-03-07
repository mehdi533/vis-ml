import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Local import is always available via run_sims sys.path tweak.
import line_utils as lu



def export_plotter_all(plotter, path: str) -> str:
    """
    Export all plotter channels (including time) to a CSV file.
    """
    if not hasattr(plotter, "_data"):
        raise RuntimeError("TDS plotter has no _data attribute to export.")
    data = dict()
    initial_values = np.asarray(plotter._data[0]).reshape(-1)
    for name, value in zip(plotter._uname, initial_values):
        data[name] = float(value)
    df = pd.DataFrame(data, index=[0])
    df.to_csv(path, index=False)
    return path


def compute_freq_metrics(t, f, f0=50, r=None, tol_hz=0.01):
    """
    Compute basic frequency and ROCOF metrics for a single trajectory f(t).
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)

    if t.size == 0 or f.size == 0:
        return {}

    if t.size != f.size:
        raise ValueError("Time and frequency arrays must have the same length.")

    # Compute ROCOF if not provided
    if r is None:
        if t.size > 1:
            rocof = np.gradient(f, t, edge_order=2 if t.size > 2 else 1)
        else:
            rocof = np.zeros_like(f)
    else:
        rocof = np.asarray(r, dtype=float)
        if rocof.size != f.size:
            raise ValueError("ROCOF array must have the same length as frequency array.")

    # Steady-state frequency: average over the last 10% (at least 10 samples)
    tail_len = max(10, t.size // 10)
    f_ss = np.mean(f[-tail_len:])

    # Nadir and maximum frequency
    idx_min = np.argmin(f)
    idx_max = np.argmax(f)
    f_min = f[idx_min]
    f_max = f[idx_max]

    # Deviations from nominal frequency
    dev_down = f0 - f_min
    dev_up = f_max - f0
    max_abs_dev = max(abs(f_min - f0), abs(f_max - f0))

    # Overshoot after nadir, relative to steady-state
    f_after_nadir = f[idx_min:]
    overshoot_up = max(0.0, np.max(f_after_nadir) - f_ss) if f_after_nadir.size > 0 else 0.0

    # Settling time: first time |f - f_ss| <= tol_hz for all subsequent samples
    within_band = np.abs(f - f_ss) <= tol_hz
    suffix_ok = np.logical_and.accumulate(within_band[::-1])[::-1]
    t_settle = t[np.argmax(suffix_ok)] if np.any(suffix_ok) else np.nan

    # ROCOF metrics
    idx_r_min = np.argmin(rocof)
    idx_r_max = np.argmax(rocof)
    rocof_min = rocof[idx_r_min]
    rocof_max = rocof[idx_r_max]
    rocof_max_abs = np.max(np.abs(rocof))
    idx_r_abs = np.argmax(np.abs(rocof))

    # Compile metrics
    metrics = {
        "f_ss": f_ss,
        "f_min": f_min,
        "t_min": t[idx_min],
        "f_max": f_max,
        "t_max": t[idx_max],
        "dev_down": dev_down,
        "dev_up": dev_up,
        "max_abs_dev": max_abs_dev,
        "overshoot_up": overshoot_up,
        "t_settle": t_settle,
        "rocof_min": rocof_min,
        "t_rocof_min": t[idx_r_min],
        "rocof_max": rocof_max,
        "t_rocof_max": t[idx_r_max],
        "rocof_max_abs": rocof_max_abs,
        "t_rocof_max_abs": t[idx_r_abs],
        "rocof_mean": np.mean(rocof),
        "rocof_rms": np.sqrt(np.mean(rocof**2)),
    }

    return metrics


def extract_ibr_peaks(plotter) -> Dict[str, float]:
    """
    Extract peak |Delta P| for each REGCV1 unit from plotter data.
    """
    idx = plotter.find("Pe REGCV1", idx_only=True)
    if not idx:
        return {}
    p_mat = np.asarray(plotter.get_values(idx)).transpose()
    peaks: Dict[str, float] = {}
    for i, series in enumerate(p_mat):
        baseline = float(series[0])
        delta = series - baseline
        peak_max = float(np.max(delta))
        peak_min = float(np.min(delta))
        peak = peak_max if np.abs(peak_max) > np.abs(peak_min) else peak_min
        peaks[f"Delta_P_IBR_{i + 1}"] = peak
    return peaks


def extract_line_metrics(
    ss,
    contingency: Optional[Dict[str, object]],
    line_uids: Optional[Sequence[int]],
) -> Dict[str, float]:
    """Extract line/topology/DC sensitivity metrics for one contingency."""
    line_uids = list(line_uids or [])
    base_mva = ss.config.mva
    cont_uid = int(contingency["uid"]) if contingency is not None and contingency.get("uid") is not None else None
    out: Dict[str, float] = {}
    out.update(lu._identity_one_hot(line_uids, cont_uid))

    ratings = lu._line_ratings_from_pandapower(ss)
    records = lu._line_records(ss)
    by_uid = {int(r["uid"]): r for r in records if r.get("uid") is not None}
    out.update(lu._global_stress(ss, ratings, base_mva))

    if cont_uid is None:
        out["line_rating"] = np.nan
        out["pre_fault_flow"] = np.nan
        out["pre_fault_loading"] = np.nan
        return out

    rec = by_uid.get(cont_uid, contingency)
    line_rating_raw = lu._valid_rating(rec.get("rating"))
    if not np.isfinite(line_rating_raw):
        line_rating_raw = lu._valid_rating(rec.get("Sn"))
    if not np.isfinite(line_rating_raw) and ratings is not None and cont_uid < len(ratings):
        line_rating_raw = lu._valid_rating(ratings[cont_uid])
    line_rating = lu._rating_to_pu(line_rating_raw, base_mva)
    p_from = lu._line_flow_component(ss, cont_uid, (("Pij", "v"), ("p1", "v"), ("P1", "v"), ("pf", "v"), ("a1", "e")))
    pre_fault_loading = (
        abs(p_from) / line_rating * 100.0
        if np.isfinite(p_from) and np.isfinite(line_rating) and line_rating > 0
        else np.nan
    )
    out["line_rating"] = lu._to_float_or_nan(line_rating)
    out["pre_fault_flow"] = lu._to_float_or_nan(p_from)
    out["pre_fault_loading"] = lu._to_float_or_nan(pre_fault_loading)

    out.update(lu._line_parameters(rec))
    flow = lu._line_prefault_flows(ss, cont_uid)
    p_from_abs = abs(flow.get("pre_p_from", np.nan)) if np.isfinite(flow.get("pre_p_from", np.nan)) else np.nan
    p_to_abs = abs(flow.get("pre_p_to", np.nan)) if np.isfinite(flow.get("pre_p_to", np.nan)) else np.nan
    flow["pre_loading_from"] = lu._to_float_or_nan(p_from_abs / line_rating * 100.0) if np.isfinite(p_from_abs) and np.isfinite(line_rating) and line_rating > 0 else np.nan
    flow["pre_loading_to"] = lu._to_float_or_nan(p_to_abs / line_rating * 100.0) if np.isfinite(p_to_abs) and np.isfinite(line_rating) and line_rating > 0 else np.nan
    p0 = flow.get("pre_p_from", np.nan)
    flow["pre_flow_direction_p"] = lu._to_float_or_nan(np.sign(p0)) if np.isfinite(p0) else np.nan
    out.update(flow)

    bus1 = lu._to_float_or_nan(rec.get("bus1"))
    bus2 = lu._to_float_or_nan(rec.get("bus2"))
    v_from, a_from = lu._bus_state(ss, bus1)
    v_to, a_to = lu._bus_state(ss, bus2)
    out.update(
        {
            "pre_v_from": lu._to_float_or_nan(v_from),
            "pre_v_to": lu._to_float_or_nan(v_to),
            "pre_theta_from": lu._to_float_or_nan(a_from),
            "pre_theta_to": lu._to_float_or_nan(a_to),
            "pre_delta_theta": lu._to_float_or_nan(a_from - a_to) if np.isfinite(a_from) and np.isfinite(a_to) else np.nan,
        }
    )
    out.update(lu._topology_criticality(records, cont_uid, bus1, bus2))
    out.update(lu._dc_sensitivity(ss, cont_uid, bus1, bus2, base_mva))
    return out


def build_feature_row(
    *,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Sequence[str],
    pq_owners: Sequence[str],
    pq_p_before: np.ndarray,
    pq_p_after: np.ndarray,
    base_load_q_total: float,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    M_agg: float,
    D_agg: float,
    genrou_pg: np.ndarray,
    regcv1_pg: np.ndarray,
) -> Dict[str, float]:
    """
    Assemble feature dictionary from inputs, PQ deltas, owner aggregates, and dispatch setpoints.
    """
    
    features: Dict[str, float] = {
        "base_load_scale": float(base_load_scale),
        "load_step_scale": float(load_step_scale),
        "load_step_time": float(load_step_time),
        "DELTA_PQ_tot": 0.0,
        "M_agg": float(M_agg),
        "D_agg": float(D_agg),
        "base_load_p_total": float(np.sum(pq_p_before)) if pq_p_before else 0.0,
        "base_load_q_total": float(base_load_q_total),
    }

    for i, (m_val, d_val) in enumerate(zip(M_vec, D_vec), start=1):
        features[f"M_{i}"] = float(m_val)
        features[f"D_{i}"] = float(d_val)

    delta_p_total = 0.0
    owner_totals: Dict[str, float] = {}
    for name, owner, p_before, p_after in zip(
        pq_names, pq_owners, pq_p_before, pq_p_after
    ):
        dp = float(p_after - p_before)
        features[f"DELTA_P_{name}"] = dp
        delta_p_total += dp
        owner_totals[owner] = owner_totals.get(owner, 0.0) + dp

    for owner, total in owner_totals.items():
        features[f"DELTA_P_OWNER_{owner}"] = float(total)

    features["DELTA_PQ_tot"] = float(delta_p_total)

    for i, val in enumerate(genrou_pg, start=1):
        features[f"P_GENROU_{i}"] = float(val)
    for i, val in enumerate(regcv1_pg, start=1):
        features[f"P_REGCV1_{i}"] = float(val)

    return features


def extract_simulation_row(
    *,
    ss,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Sequence[str],
    pq_owners: Sequence[str],
    pq_p_before: np.ndarray,
    pq_p_after: np.ndarray,
    base_load_q_total: float,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    genrou_pg: np.ndarray,
    regcv1_pg: np.ndarray,
    success: bool,
    contingency: Optional[Dict[str, object]] = None,
    load_step_enabled: bool = False,
    trip_time: float = np.nan,
    line_uids: Optional[Sequence[int]] = None,
    line_metrics_snapshot: Optional[Dict[str, float]] = None,
    plotter_csv: Optional[str] = None,
) -> Dict[str, float]:
    """
    Build a row containing features, labels, and metadata for one simulation.
    """

    # ============== Extract features ================
    # TODO: check this
    gen_m = np.asarray(ss.GENROU.M.v, dtype=float) if getattr(ss.GENROU, "n", 0) > 0 else np.zeros(0, dtype=float)
    gen_d = np.asarray(ss.GENROU.D.v, dtype=float) if getattr(ss.GENROU, "n", 0) > 0 else np.zeros(0, dtype=float)
    ibr_m = np.asarray(ss.REGCV1.M.v, dtype=float) if getattr(ss.REGCV1, "n", 0) > 0 else np.zeros(0, dtype=float)
    ibr_d = np.asarray(ss.REGCV1.D.v, dtype=float) if getattr(ss.REGCV1, "n", 0) > 0 else np.zeros(0, dtype=float)

    m_all = np.concatenate([gen_m, ibr_m]) if (gen_m.size or ibr_m.size) else np.array([0.0], dtype=float)
    d_all = np.concatenate([gen_d, ibr_d]) if (gen_d.size or ibr_d.size) else np.array([0.0], dtype=float)
    M_agg = float(np.mean(m_all))
    D_agg = float(np.mean(d_all))

    features = build_feature_row(
        base_load_scale=base_load_scale,
        load_step_scale=load_step_scale,
        load_step_time=load_step_time,
        pq_names=pq_names,
        pq_owners=pq_owners,
        pq_p_before=pq_p_before,
        pq_p_after=pq_p_after,
        base_load_q_total=base_load_q_total,
        M_vec=M_vec,
        D_vec=D_vec,
        M_agg=M_agg,
        D_agg=D_agg,
        genrou_pg=genrou_pg,
        regcv1_pg=regcv1_pg,
    )

    # ============== Extract labels ================
    ss.TDS.load_plotter()
    plotter = ss.TDS.plotter
    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    f_coi = np.asarray(plotter.get_values(plotter.find("omega COI", idx_only=True)), dtype=float).reshape(-1) * 50
    r_coi = np.gradient(f_coi, float(ss.TDS.config.tstep), axis=0)
    
    assert time is not None and f_coi is not None and r_coi is not None, "COI signals are missing"

    f0 = getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0
    metrics = compute_freq_metrics(time, f=f_coi, f0=f0, r=r_coi)

    assert metrics, "COI metrics dictionary is empty."
    
    labels: Dict[str, float] = {}
    time_of_max_dev = np.nan

    dev_down = float(metrics.get("dev_down", np.nan))  # f0 - f_min (>=0)
    dev_up = float(metrics.get("dev_up", np.nan))      # f_max - f0 (>=0)
    f_min = float(metrics.get("f_min", np.nan))
    f_max = float(metrics.get("f_max", np.nan))

    # Pick the deviation with largest magnitude and keep the sign
    if np.isfinite(dev_down) and np.isfinite(dev_up):
        if dev_down >= dev_up:
            dev_signed = f_min - f0  # negative or zero
            time_of_max_dev = float(metrics.get("t_min", np.nan))
        else:
            dev_signed = f_max - f0  # positive or zero
            time_of_max_dev = float(metrics.get("t_max", np.nan))
    else:
        dev_signed = np.nan

    rocof_min = float(metrics.get("rocof_min", np.nan))
    rocof_max = float(metrics.get("rocof_max", np.nan))
    if np.isfinite(rocof_min) and np.isfinite(rocof_max):
        rocof_signed = rocof_min if abs(rocof_min) >= abs(rocof_max) else rocof_max
    else:
        rocof_signed = np.nan

    labels["rocof_COI"] = rocof_signed
    labels["dev_COI"] = dev_signed
    ibr_peaks = extract_ibr_peaks(ss.TDS.plotter)
    labels.update(ibr_peaks)

    # ================================================ 
    row: Dict[str, float] = {}
    row.update(features)
    row.update(labels)
    row["time_max_dev"] = float(time_of_max_dev) if np.isfinite(time_of_max_dev) else np.nan
    row["success"] = bool(success)
    line_metrics = line_metrics_snapshot or extract_line_metrics(ss=ss, contingency=contingency, line_uids=line_uids)

    if contingency is None:
        if load_step_enabled:
            row["cont_type"] = "load"
            row["contingency_time"] = float(load_step_time)
        else:
            row["cont_type"] = "none"
            row["contingency_time"] = np.nan
        row["line_uid"] = np.nan
        row["line_name"] = ""
        row["line_from_bus"] = np.nan
        row["line_to_bus"] = np.nan
        row["line_rating"] = float(line_metrics.get("line_rating", np.nan))
        row["pre_fault_flow"] = float(line_metrics.get("pre_fault_flow", np.nan))
        row["pre_fault_loading"] = float(line_metrics.get("pre_fault_loading", np.nan))
    else:
        row["cont_type"] = "line_plus_load" if load_step_enabled else "line"
        row["contingency_time"] = float(trip_time if trip_time is not None else np.nan)
        row["line_uid"] = int(contingency.get("uid")) if contingency.get("uid") is not None else np.nan
        row["line_name"] = str(contingency["name"])
        bus1_val = contingency.get("bus1")
        bus2_val = contingency.get("bus2")
        row["line_from_bus"] = int(bus1_val) if bus1_val is not None and np.isfinite(float(bus1_val)) else np.nan
        row["line_to_bus"] = int(bus2_val) if bus2_val is not None and np.isfinite(float(bus2_val)) else np.nan
        row["line_rating"] = float(line_metrics.get("line_rating", np.nan))
        row["pre_fault_flow"] = float(line_metrics.get("pre_fault_flow", np.nan))
        row["pre_fault_loading"] = float(line_metrics.get("pre_fault_loading", np.nan))

    row.update(line_metrics)

    if plotter_csv:
        row["plotter_csv"] = os.path.abspath(plotter_csv)

    return row
