import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd



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


def build_feature_row(
    *,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Sequence[str],
    pq_p_before: np.ndarray,
    pq_q_before: np.ndarray,
    pq_p_after: np.ndarray,
    pq_q_after: np.ndarray,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    M_agg: float,
    D_agg: float,
) -> Dict[str, float]:
    """
    Assemble feature dictionary from inputs and PQ deltas.
    """
    features: Dict[str, float] = {
        "base_load_scale": float(base_load_scale),
        "load_step_scale": float(load_step_scale),
        "load_step_time": float(load_step_time),
        "DELTA_PQ_tot": 0.0,
        "M_agg": float(M_agg),
        "D_agg": float(D_agg),
        "base_load_p_total": float(np.sum(pq_p_before)) if pq_p_before else 0.0,
        "base_load_q_total": float(np.sum(pq_q_before)) if pq_q_before else 0.0,
    }

    for i, (m_val, d_val) in enumerate(zip(M_vec, D_vec), start=1):
        features[f"M_{i}"] = float(m_val)
        features[f"D_{i}"] = float(d_val)

    delta_p_total = 0.0
    delta_q_total = 0.0
    for name, p_before, p_after, q_before, q_after in zip(
        pq_names, pq_p_before, pq_p_after, pq_q_before, pq_q_after
    ):
        dp = float(p_after - p_before)
        dq = float(q_after - q_before)
        features[f"DELTA_P_{name}"] = dp
        features[f"DELTA_Q_{name}"] = dq
        delta_p_total += dp
        delta_q_total += dq

    features["DELTA_PQ_tot"] = float(delta_p_total + delta_q_total)
    return features


def extract_simulation_row(
    *,
    ss,
    base_load_scale: float,
    load_step_scale: float,
    load_step_time: float,
    pq_names: Sequence[str],
    pq_p_before: np.ndarray,
    pq_q_before: np.ndarray,
    pq_p_after: np.ndarray,
    pq_q_after: np.ndarray,
    M_vec: Sequence[float],
    D_vec: Sequence[float],
    success: bool,
    plotter_csv: Optional[str] = None,
) -> Dict[str, float]:
    """
    Build a row containing features, labels, and metadata for one simulation.
    """

    # ============== Extract features ================
    # TODO: check this
    M_agg = np.mean(np.concatenate([ss.GENROU.M.v, ss.REGCV1.M.v])).sum()
    D_agg = np.mean(np.concatenate([ss.GENROU.D.v, ss.REGCV1.D.v])).sum()

    features = build_feature_row(
        base_load_scale=base_load_scale,
        load_step_scale=load_step_scale,
        load_step_time=load_step_time,
        pq_names=pq_names,
        pq_p_before=pq_p_before,
        pq_q_before=pq_q_before,
        pq_p_after=pq_p_after,
        pq_q_after=pq_q_after,
        M_vec=M_vec,
        D_vec=D_vec,
        M_agg=M_agg,
        D_agg=D_agg,
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
    
    labels: Dict[str, float] = {
        "rocof_max_COI": np.nan,
        "rocof_min_COI": np.nan,
        "devdown_COI": np.nan,
        "devup_COI": np.nan,
    }
    time_of_max_dev = np.nan

    dev_down = metrics.get("dev_down", np.nan)
    dev_up = metrics.get("dev_up", np.nan)
    if np.isfinite(dev_down) and np.isfinite(dev_up):
        if dev_down >= dev_up:
            time_of_max_dev = float(metrics.get("t_min", np.nan))
        else:
            time_of_max_dev = float(metrics.get("t_max", np.nan))

    labels["rocof_max_COI"] = float(metrics.get("rocof_max", np.nan))
    labels["rocof_min_COI"] = float(metrics.get("rocof_min", np.nan))
    labels["devdown_COI"] = float(metrics.get("dev_down", np.nan))
    labels["devup_COI"] = float(metrics.get("dev_up", np.nan))
    ibr_peaks = extract_ibr_peaks(ss.TDS.plotter)
    labels.update(ibr_peaks)

    # ================================================ 
    row: Dict[str, float | str | bool] = {}
    row.update(features)
    row.update(labels)
    row["time_max_dev"] = float(time_of_max_dev) if np.isfinite(time_of_max_dev) else np.nan
    row["success"] = bool(success)
    if plotter_csv:
        row["plotter_csv"] = os.path.abspath(plotter_csv)

    return row
