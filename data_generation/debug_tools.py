from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def save_debug_coi_plot(
    *,
    ss,
    plotter,
    output_dir: Path,
    sim_id: int,
    contingency: Optional[Dict],
    step_scale: float,
) -> None:
    """Save a per-simulation COI frequency/RoCoF plot next to the CSV."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[debug-plot] sim_id={sim_id} skipped COI plot: matplotlib unavailable ({exc})")
        return

    try:
        time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
        coi_indices = list(plotter.find("omega COI", idx_only=True))
        ibr_freq_indices = list(plotter.find("omega REGCV1", idx_only=True))
        if not ibr_freq_indices:
            ibr_freq_indices = list(plotter.find("dw REGCV1", idx_only=True))
        genrou_freq_indices = list(plotter.find("omega GENROU", idx_only=True))
        ibr_indices = list(plotter.find("Pe REGCV1", idx_only=True))
        ibr_q_indices = list(plotter.find("Qe REGCV1", idx_only=True))
        pref2_indices = list(plotter.find("Pref2 REGCV1", idx_only=True))
        genrou_indices = list(plotter.find("Pe GENROU", idx_only=True))
        genrou_q_indices = list(plotter.find("Qe GENROU", idx_only=True))
        channel_names = [str(value) for value in list(getattr(plotter, "_uname", []))]
    except Exception as exc:
        print(f"[debug-plot] sim_id={sim_id} skipped COI plot: unable to read plotter ({exc})")
        return

    if time.size < 2 or not coi_indices:
        return

    try:
        f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
        f_coi_hz = np.asarray(plotter.get_values([int(coi_indices[0])]), dtype=float).reshape(-1) * f0
        if f_coi_hz.size != time.size:
            return
        rocof_hz_s = np.gradient(f_coi_hz, time, edge_order=2 if time.size > 2 else 1)

        ibr_freq_series: List[Tuple[str, np.ndarray]] = []
        for idx in ibr_freq_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"REGCV1_freq_{idx}"
            if label.startswith("dw REGCV1"):
                series_hz = (1.0 + series) * f0
            else:
                series_hz = series * f0
            ibr_freq_series.append((label, series_hz))

        genrou_freq_series: List[Tuple[str, np.ndarray]] = []
        for idx in genrou_freq_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"GENROU_freq_{idx}"
            genrou_freq_series.append((label, series * f0))

        ibr_series: List[Tuple[str, np.ndarray]] = []
        for idx in ibr_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            baseline = float(series[0])
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"REGCV1_{idx}"
            ibr_series.append((label, series - baseline))

        ibr_q_series: List[Tuple[str, np.ndarray]] = []
        for idx in ibr_q_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            baseline = float(series[0])
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"REGCV1_Q_{idx}"
            ibr_q_series.append((label, series - baseline))

        pref2_series: List[Tuple[str, np.ndarray]] = []
        for idx in pref2_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"Pref2_REGCV1_{idx}"
            pref2_series.append((label, series))

        genrou_series: List[Tuple[str, np.ndarray]] = []
        for idx in genrou_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            baseline = float(series[0])
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"GENROU_{idx}"
            genrou_series.append((label, series - baseline))

        genrou_q_series: List[Tuple[str, np.ndarray]] = []
        for idx in genrou_q_indices:
            series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
            if series.size != time.size or series.size == 0:
                continue
            baseline = float(series[0])
            label = channel_names[int(idx)] if 0 <= int(idx) < len(channel_names) else f"GENROU_Q_{idx}"
            genrou_q_series.append((label, series - baseline))
    except Exception as exc:
        print(f"[debug-plot] sim_id={sim_id} skipped COI plot: unable to build series ({exc})")
        return

    line_uid = -1 if contingency is None else int(contingency.get("uid", -1))
    plot_path = output_dir / f"sim_{sim_id:06d}_line_{line_uid}_coi_debug.png"

    fig, axes = plt.subplots(5, 2, figsize=(15, 17), sharex=True)
    axes = axes.reshape(-1)

    axes[0].plot(time, f_coi_hz, color="tab:blue", linewidth=1.5)
    axes[0].axhline(f0, color="0.5", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("COI frequency [Hz]")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, rocof_hz_s, color="tab:red", linewidth=1.5)
    axes[1].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("COI RoCoF [Hz/s]")
    axes[1].grid(True, alpha=0.3)

    for label, series in ibr_freq_series:
        axes[2].plot(time, series, linewidth=1.2, label=label)
    axes[2].axhline(f0, color="0.5", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("IBR frequency [Hz]")
    axes[2].grid(True, alpha=0.3)
    if ibr_freq_series:
        axes[2].legend(loc="best", fontsize=8, ncol=2)

    for label, series in genrou_freq_series:
        axes[3].plot(time, series, linewidth=1.2, label=label)
    axes[3].axhline(f0, color="0.5", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("GENROU frequency [Hz]")
    axes[3].grid(True, alpha=0.3)
    if genrou_freq_series:
        axes[3].legend(loc="best", fontsize=8, ncol=2)

    for label, series in ibr_series:
        axes[4].plot(time, series, linewidth=1.2, label=label)
    axes[4].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[4].set_ylabel("IBR dPe [p.u.]")
    axes[4].grid(True, alpha=0.3)
    if ibr_series:
        axes[4].legend(loc="best", fontsize=8, ncol=2)

    for label, series in genrou_series:
        axes[5].plot(time, series, linewidth=1.2, label=label)
    axes[5].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[5].set_ylabel("GENROU dPe [p.u.]")
    axes[5].grid(True, alpha=0.3)
    if genrou_series:
        axes[5].legend(loc="best", fontsize=8, ncol=2)

    for label, series in pref2_series:
        axes[6].plot(time, series, linewidth=1.2, label=label)
    axes[6].set_ylabel("REGCV1 Pref2 [p.u.]")
    axes[6].grid(True, alpha=0.3)
    if pref2_series:
        axes[6].legend(loc="best", fontsize=8, ncol=2)

    for label, series in ibr_q_series:
        axes[7].plot(time, series, linewidth=1.2, label=label)
    axes[7].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[7].set_ylabel("IBR dQe [p.u.]")
    axes[7].grid(True, alpha=0.3)
    if ibr_q_series:
        axes[7].legend(loc="best", fontsize=8, ncol=2)

    for label, series in genrou_q_series:
        axes[8].plot(time, series, linewidth=1.2, label=label)
    axes[8].axhline(0.0, color="0.5", linestyle="--", linewidth=1.0)
    axes[8].set_ylabel("GENROU dQe [p.u.]")
    axes[8].grid(True, alpha=0.3)
    if genrou_q_series:
        axes[8].legend(loc="best", fontsize=8, ncol=2)

    axes[9].axis("off")

    for idx in (8, 9):
        axes[idx].set_xlabel("Time [s]")

    fig.suptitle(f"sim_id={sim_id}, line_uid={line_uid}, step_scale={step_scale:.4f}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_coi_trace_csv(*, ss, plotter, output_dir: Path, sim_id: int) -> None:
    """Save COI trace CSV for one simulation."""
    try:
        time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
        coi_indices = list(plotter.find("omega COI", idx_only=True))
    except Exception as exc:
        print(f"[debug-trace] sim_id={sim_id} skipped COI trace: unable to read plotter ({exc})")
        return

    if time.size < 2 or not coi_indices:
        return

    try:
        f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
        f_coi_hz = np.asarray(plotter.get_values([int(coi_indices[0])]), dtype=float).reshape(-1) * f0
        if f_coi_hz.size != time.size:
            return
        rocof_hz_s = np.gradient(f_coi_hz, time, edge_order=2 if time.size > 2 else 1)
    except Exception as exc:
        print(f"[debug-trace] sim_id={sim_id} skipped COI trace: unable to compute series ({exc})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"coi_trace_sim_{sim_id:06d}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "f_coi_hz", "rocof_coi_hz_per_s"])
        for t, f_hz, r_hz_s in zip(time, f_coi_hz, rocof_hz_s):
            writer.writerow([float(t), float(f_hz), float(r_hz_s)])
