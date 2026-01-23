import numpy as np
import pandas as pd


def compute_freq_metrics(t, f, f0=50, r=None, tol_hz=0.01, per_trace=False):
    t = np.asarray(t, dtype=float).reshape(-1)
    n = t.size
    if n == 0:
        return {}

    def _to_2d(arr, name):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size != n:
                raise ValueError(f"t and {name} must have the same length.")
            return arr.reshape(1, n)
        if arr.ndim == 2:
            if arr.shape[1] == n:
                return arr
            if arr.shape[0] == n:
                return arr.T
        raise ValueError(f"{name} must be 1D of length {n} or 2D with one dimension equal to len(t).")

    f_mat = _to_2d(f, "f")
    if f_mat.size == 0:
        return {}

    n_traj, n_samples = f_mat.shape

    f0_arr = np.asarray(f0, dtype=float).reshape(-1)
    if f0_arr.size == 1:
        f0_arr = np.full(n_traj, float(f0_arr[0]))
    elif f0_arr.size != n_traj:
        raise ValueError("f0 must be a scalar or have one entry per trajectory.")

    if r is None:
        if n_samples == 1:
            rocof = np.zeros_like(f_mat)
        else:
            rocof = np.gradient(f_mat, t, axis=1, edge_order=2 if n_samples > 2 else 1)
    else:
        rocof_mat = _to_2d(r, "r")
        if rocof_mat.shape[0] not in (1, n_traj):
            raise ValueError("r must match the number of trajectories or be a single trajectory.")
        rocof = np.broadcast_to(rocof_mat, (n_traj, n_samples))

    tail_len = max(10, n_samples // 10)
    f_ss = np.mean(f_mat[:, -tail_len:], axis=1)

    idx_min = np.argmin(f_mat, axis=1)
    idx_max = np.argmax(f_mat, axis=1)
    row_idx = np.arange(n_traj)
    f_min = f_mat[row_idx, idx_min]
    f_max = f_mat[row_idx, idx_max]

    dev_down = f0_arr - f_min
    dev_up = f_max - f0_arr
    max_abs_dev = np.maximum(np.abs(f_min - f0_arr), np.abs(f_max - f0_arr))

    overshoot_up = np.zeros(n_traj, dtype=float)
    for i in range(n_traj):
        f_after = f_mat[i, idx_min[i] :]
        if f_after.size:
            overshoot_up[i] = max(0.0, float(np.max(f_after) - f_ss[i]))

    within_band = np.abs(f_mat - f_ss[:, None]) <= tol_hz
    suffix_ok = np.logical_and.accumulate(within_band[:, ::-1], axis=1)[:, ::-1]
    any_true = np.any(suffix_ok, axis=1)
    idx_settle = np.argmax(suffix_ok, axis=1)
    t_settle = np.full(n_traj, np.nan, dtype=float)
    t_settle[any_true] = t[idx_settle[any_true]]

    idx_r_min = np.argmin(rocof, axis=1)
    idx_r_max = np.argmax(rocof, axis=1)
    abs_rocof = np.abs(rocof)
    idx_r_abs = np.argmax(abs_rocof, axis=1)

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
        "rocof_min": rocof[row_idx, idx_r_min],
        "t_rocof_min": t[idx_r_min],
        "rocof_max": rocof[row_idx, idx_r_max],
        "t_rocof_max": t[idx_r_max],
        "rocof_max_abs": abs_rocof[row_idx, idx_r_abs],
        "t_rocof_max_abs": t[idx_r_abs],
        "rocof_mean": np.mean(rocof, axis=1),
        "rocof_rms": np.sqrt(np.mean(rocof * rocof, axis=1)),
    }

    if n_traj == 1 or per_trace:
        if n_traj == 1:
            return {k: float(np.asarray(v)[0]) for k, v in metrics.items()}
        return metrics

    agg = {}
    f_min_idx = int(np.argmin(metrics["f_min"]))
    f_max_idx = int(np.argmax(metrics["f_max"]))
    r_min_idx = int(np.argmin(metrics["rocof_min"]))
    r_max_idx = int(np.argmax(metrics["rocof_max"]))
    r_abs_idx = int(np.argmax(metrics["rocof_max_abs"]))

    def _first_valid_or_nan(arr, idx):
        arr = np.asarray(arr, dtype=float)
        return float(arr[idx]) if arr.size > idx else np.nan

    agg["f_ss"] = float(np.mean(metrics["f_ss"]))
    agg["f_min"] = float(metrics["f_min"][f_min_idx])
    agg["t_min"] = _first_valid_or_nan(metrics["t_min"], f_min_idx)
    agg["f_max"] = float(metrics["f_max"][f_max_idx])
    agg["t_max"] = _first_valid_or_nan(metrics["t_max"], f_max_idx)
    agg["dev_down"] = float(np.max(metrics["dev_down"]))
    agg["dev_up"] = float(np.max(metrics["dev_up"]))
    agg["max_abs_dev"] = float(np.max(metrics["max_abs_dev"]))
    agg["overshoot_up"] = float(np.max(metrics["overshoot_up"]))

    t_settle_arr = np.asarray(metrics["t_settle"], dtype=float)
    agg["t_settle"] = float(np.nanmax(t_settle_arr)) if np.any(np.isfinite(t_settle_arr)) else np.nan

    agg["rocof_min"] = float(metrics["rocof_min"][r_min_idx])
    agg["t_rocof_min"] = _first_valid_or_nan(metrics["t_rocof_min"], r_min_idx)
    agg["rocof_max"] = float(metrics["rocof_max"][r_max_idx])
    agg["t_rocof_max"] = _first_valid_or_nan(metrics["t_rocof_max"], r_max_idx)
    agg["rocof_max_abs"] = float(metrics["rocof_max_abs"][r_abs_idx])
    agg["t_rocof_max_abs"] = _first_valid_or_nan(metrics["t_rocof_max_abs"], r_abs_idx)
    agg["rocof_mean"] = float(np.mean(metrics["rocof_mean"]))
    agg["rocof_rms"] = float(np.sqrt(np.mean(rocof * rocof)))

    return agg


def compute_coi_frequency_and_rocof(ss):
    if not hasattr(ss, "GENROU") or ss.GENROU.n == 0:
        return None, None, None

    ss.TDS.load_plotter()
    plotter = ss.TDS.plotter

    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    omega_idx, headers = plotter.find("omega GENROU,omega REGCV1")[0], plotter.find("omega GENROU,omega REGCV1")[1]
    names = ["_".join(h.split(" ")[1:]) for h in headers]
    gen_df = ss.GENROU.as_df()
    ren_df = ss.REGCV1.as_df()
    df = pd.concat([gen_df, ren_df])

    weights = []
    for name in names:
        row = df.loc[df["name"] == name]
        if row.empty:
            weights.append(np.nan)
            continue
        row = row.iloc[0]
        M = row.get("M", np.nan) * row.get("u", 0)
        weights.append(float(M))

    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(weights)
    if not np.any(valid):
        return None, None, None

    omega_raw = np.asarray(plotter.get_values(omega_idx).transpose())
    if omega_raw.ndim == 1:
        omega_mat = omega_raw.reshape(1, -1)
    elif omega_raw.shape[0] == len(omega_idx):
        omega_mat = omega_raw
    elif omega_raw.shape[1] == len(omega_idx):
        omega_mat = omega_raw.T
    else:
        return None, None, None

    omega_mat = omega_mat[valid]
    weights = weights[valid]
    if omega_mat.shape[0] == 0:
        return None, None, None

    denom = weights.sum()
    coi_omega = (weights @ omega_mat) / denom

    f0 = getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0
    coi_freq_hz = coi_omega * f0

    if time.size <= 1:
        coi_rocof = np.zeros_like(coi_freq_hz)
    else:
        coi_rocof = np.gradient(coi_freq_hz, time, edge_order=2 if time.size > 2 else 1)

    return time, coi_freq_hz, coi_rocof


def aggregate_system_parameters(ss):
    S_base = ss.config.mva

    reg_df = ss.REGCV1.as_df()
    S_ibr = reg_df["Sn"].to_numpy(dtype=float)
    M_ibr = reg_df["M"].to_numpy(dtype=float)
    D_ibr = reg_df["D"].to_numpy(dtype=float)
    kw_ibr = reg_df["kw"].to_numpy(dtype=float)

    sg_df = ss.GENROU.as_df()[ss.GENROU.as_df()["u"] == 1]
    S_sg = sg_df["Sn"].to_numpy(dtype=float)
    M_sg = sg_df["M"].to_numpy(dtype=float)
    D_sg = sg_df["D"].to_numpy(dtype=float)

    S_total = S_sg.sum() + S_ibr.sum()
    if S_total == 0:
        raise ValueError("Total S_sg + S_ibr is zero; cannot aggregate M and D.")

    g_df = ss.TGOV1N.as_df()[ss.TGOV1N.as_df()["u"] == 1]
    R_i = g_df["R"].to_numpy(dtype=float)
    K_i = np.ones_like(R_i)
    F_i = np.ones_like(R_i) * S_sg / S_total
    T_i = g_df["T3"].to_numpy(dtype=float)

    M_agg = np.mean(np.concatenate([M_sg, M_ibr])) * S_base / S_total
    D_agg = np.mean(np.concatenate([D_sg, D_ibr])) * S_base / S_total

    S_sg_total = S_sg.sum()
    if S_sg_total == 0:
        raise ValueError("No synchronous generators found; cannot compute R,F aggregation.")

    R_agg = (K_i * S_sg / (R_i * S_sg / S_base)).sum() / S_sg_total
    F_agg = (K_i * F_i * S_sg / (R_i * S_sg / S_base)).sum() / S_sg_total
    T_agg = float((T_i * S_sg).sum() / S_sg_total)

    return M_agg, D_agg, R_agg, F_agg, T_agg, M_ibr, D_ibr, kw_ibr
