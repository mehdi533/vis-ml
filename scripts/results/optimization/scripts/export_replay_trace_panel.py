from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import andes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_GEN_DIR = ROOT / "data_generation"
if str(DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_GEN_DIR))

from extract_metrics import _normalize_plotter_indices, _plotter_channel_names, _plotter_series_matrix  # type: ignore  # noqa: E402
from scheduling.replay_validation import (  # noqa: E402
    _add_line_trip_contingency,
    _add_load_step_contingency,
    _apply_base_scale,
    _apply_dispatch,
    _apply_md,
    _configure_tds,
    _setup_pq_model,
)


DEFAULT_STYLE_CONFIG = ROOT / "configs/figure/thesis_plot_style.yaml"


def _resolve(path_like: str, base: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p

    repo_relative_prefixes = (
        "results/",
        "LaTeX/",
        "configs/",
        "scheduling/",
        "models/",
        "data_generation/",
    )
    path_str = str(path_like).replace("\\", "/")
    root_candidate = (ROOT / p).resolve()
    base_candidate = (base / p).resolve()

    if root_candidate.exists():
        return root_candidate
    if base_candidate.exists():
        return base_candidate
    if path_str.startswith(repo_relative_prefixes):
        return root_candidate
    return base_candidate


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_") or "trace"


def _extract_numeric_suffix(name: str) -> int | None:
    match = re.search(r"(\d+)\s*$", str(name))
    return int(match.group(1)) if match else None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_style_config(style_config_path: str | None, cfg_path: Path) -> dict[str, Any]:
    if style_config_path:
        candidate = _resolve(style_config_path, cfg_path.parent)
    else:
        candidate = DEFAULT_STYLE_CONFIG
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _style_palette(style_cfg: dict[str, Any], palette_name: str) -> dict[str, str]:
    return dict(style_cfg.get("style", {}).get("palettes", {}).get(palette_name, {}) or {})


def _style_line_cfg(style_cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(style_cfg.get("style", {}).get("line", {}) or {})


def _style_font_sizes(style_cfg: dict[str, Any]) -> dict[str, float]:
    cfg = dict(style_cfg.get("style", {}).get("font_sizes", {}) or {})
    return {
        "axis_label": float(cfg.get("axis_label", 12.5)),
        "tick": float(cfg.get("tick", 11.0)),
        "legend": float(cfg.get("legend", 12.0)),
        "title": float(cfg.get("title", 13.5)),
    }


def _apply_style_rcparams(style_cfg: dict[str, Any]) -> None:
    rc_from_cfg = dict(style_cfg.get("style", {}).get("matplotlib", {}).get("rc_params", {}) or {})
    if rc_from_cfg:
        plt.rcParams.update(rc_from_cfg)


def _extract_frequency_trace(ss, plotter) -> pd.DataFrame:
    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    coi_indices = _normalize_plotter_indices(plotter.find("omega COI", idx_only=True))
    if not coi_indices:
        raise RuntimeError("Could not find 'omega COI' trace in the ANDES plotter.")
    f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
    f_coi = _plotter_series_matrix(plotter, coi_indices).reshape(-1) * f0
    # Different ANDES plotter exports can be subsampled relative to the internal
    # TDS step. Use the actual exported time vector to avoid overstating RoCoF.
    rocof = np.gradient(f_coi, time, axis=0)
    return pd.DataFrame(
        {
            "time_s": time,
            "f_coi_hz": f_coi,
            "delta_f_coi_hz": f_coi - f0,
            "rocof_coi_hz_per_s": rocof,
        }
    )


def _extract_ibr_delta_p_traces(plotter) -> pd.DataFrame:
    indices = _normalize_plotter_indices(plotter.find("Pe REGCV1", idx_only=True))
    if not indices:
        return pd.DataFrame()
    names = _plotter_channel_names(plotter, indices)
    p_matrix = _plotter_series_matrix(plotter, indices)
    if p_matrix.size == 0:
        return pd.DataFrame()
    out: dict[str, np.ndarray] = {}
    abs_stack: list[np.ndarray] = []
    for col, name in enumerate(names):
        unit_id = _extract_numeric_suffix(name) or (col + 1)
        series = np.asarray(p_matrix[:, col], dtype=float)
        delta = series - float(series[0])
        out[f"Delta_P_IBR_{unit_id}"] = delta
        abs_stack.append(np.abs(delta))
    if abs_stack:
        out["max_abs_delta_p_ibr"] = np.max(np.vstack(abs_stack), axis=0)
    return pd.DataFrame(out)


def _extract_total_delta_trace(plotter, query: str, out_col: str) -> pd.DataFrame:
    indices = _normalize_plotter_indices(plotter.find(query, idx_only=True))
    if not indices:
        return pd.DataFrame()
    p_matrix = _plotter_series_matrix(plotter, indices)
    if p_matrix.size == 0:
        return pd.DataFrame()
    total = np.asarray(np.nansum(p_matrix, axis=1), dtype=float)
    delta = total - float(total[0])
    return pd.DataFrame({out_col: delta})


def _extract_genrou_load_delta_traces(plotter) -> pd.DataFrame:
    # Best-effort extraction across common ANDES channel names.
    gen_df = _extract_total_delta_trace(plotter, "Pe GENROU", "Delta_P_GENROU_total")
    load_df = pd.DataFrame()
    for load_query in ("Pe PQ", "Psum PQ", "Psum load", "Pload", "PQ"):
        load_df = _extract_total_delta_trace(plotter, load_query, "Delta_P_Load_total")
        if not load_df.empty:
            break
    if gen_df.empty and load_df.empty:
        return pd.DataFrame()
    return pd.concat([gen_df, load_df], axis=1)


def _bus_names_by_idx(ss) -> dict[int, str]:
    if not hasattr(ss, "Bus"):
        return {}
    idx_vals = np.asarray(list(getattr(ss.Bus, "idx").v), dtype=int) if hasattr(ss.Bus, "idx") else np.asarray([], dtype=int)
    if hasattr(ss.Bus, "name"):
        name_vals = [str(v) for v in list(getattr(ss.Bus, "name").v)]
    else:
        name_vals = [str(v) for v in idx_vals.tolist()]
    return {int(i): str(n) for i, n in zip(idx_vals.tolist(), name_vals)}


def _parse_scenario_owner_id(scenario_id: str) -> int | None:
    m = re.search(r"zone_owner_(\d+)", str(scenario_id))
    return int(m.group(1)) if m else None


def _auto_select_bus_ids_by_owner(ss, *, owner_ids: list[int] | None = None) -> list[int]:
    if not hasattr(ss, "PQ") or not hasattr(ss.PQ, "owner") or not hasattr(ss.PQ, "bus") or not hasattr(ss.PQ, "Ppf"):
        return []
    pq_owner = np.asarray(list(ss.PQ.owner.v), dtype=int)
    pq_bus = np.asarray(list(ss.PQ.bus.v), dtype=int)
    pq_p = np.asarray(list(ss.PQ.Ppf.v), dtype=float)
    if owner_ids is None or not owner_ids:
        owner_ids = sorted(set(int(v) for v in pq_owner.tolist()))
    selected: list[int] = []
    for owner in owner_ids:
        mask = pq_owner == int(owner)
        if not np.any(mask):
            continue
        bus_owner = pq_bus[mask]
        p_owner = pq_p[mask]
        bus_totals: dict[int, float] = {}
        for b, p in zip(bus_owner.tolist(), p_owner.tolist()):
            b_int = int(b)
            bus_totals[b_int] = bus_totals.get(b_int, 0.0) + float(p)
        if not bus_totals:
            continue
        # Deterministic: highest prefault active load, then lowest bus id.
        chosen = sorted(bus_totals.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        selected.append(int(chosen))
    return selected


def _resolve_bus_selection(
    ss,
    *,
    bus_selection_cfg: dict[str, Any] | None,
    scenario_id: str,
    run_cfg: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    cfg = dict(bus_selection_cfg or {})
    explicit = cfg.get("bus_ids") or cfg.get("explicit_bus_ids") or run_cfg.get("bus_ids")
    if explicit:
        chosen = [int(v) for v in list(explicit)]
        return chosen, {"mode": "explicit", "bus_ids": chosen}

    mode = str(cfg.get("mode", "auto_zone_owner")).lower()
    if mode not in {"auto_zone_owner", "auto"}:
        mode = "auto_zone_owner"
    owner_hint = run_cfg.get("owner_id")
    owner_ids_cfg = cfg.get("owner_ids")
    if owner_hint is not None:
        owner_ids = [int(owner_hint)]
    elif owner_ids_cfg:
        owner_ids = [int(v) for v in list(owner_ids_cfg)]
    else:
        parsed = _parse_scenario_owner_id(scenario_id)
        owner_ids = [int(parsed)] if parsed is not None else None
    chosen = _auto_select_bus_ids_by_owner(ss, owner_ids=owner_ids)
    return chosen, {"mode": mode, "owner_ids": owner_ids or [], "bus_ids": chosen}


def _pq_names_from_bus_ids(ss, bus_ids: list[int]) -> list[str]:
    if not bus_ids:
        return []
    if not hasattr(ss, "PQ") or not hasattr(ss.PQ, "bus") or not hasattr(ss.PQ, "name"):
        return []
    bus_set = {int(v) for v in bus_ids}
    pq_bus = np.asarray(list(ss.PQ.bus.v), dtype=int)
    pq_name = [str(v) for v in list(ss.PQ.name.v)]
    out: list[str] = []
    for b, n in zip(pq_bus.tolist(), pq_name):
        if int(b) in bus_set:
            out.append(str(n))
    # Stable deterministic ordering by name
    return sorted(set(out))


def _extract_bus_frequency_traces(ss, plotter, selected_bus_ids: list[int]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not selected_bus_ids:
        return pd.DataFrame(), []
    indices: list[int] = []
    for q in ("omega Bus", "omega bus", "omega BUS", "omega BusFreq", "f Bus"):
        found = _normalize_plotter_indices(plotter.find(q, idx_only=True))
        if found:
            indices = list(found)
            break
    if not indices:
        return pd.DataFrame(), []
    names = _plotter_channel_names(plotter, indices)
    mat = _plotter_series_matrix(plotter, indices)
    if mat.size == 0:
        return pd.DataFrame(), []
    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    f0 = float(getattr(getattr(ss, "config", None), "freq", 50.0) or 50.0)
    bus_name_map = _bus_names_by_idx(ss)
    df = pd.DataFrame({"time_s": time})
    selected_set = {int(v) for v in selected_bus_ids}
    selected_meta: list[dict[str, Any]] = []
    for col, name in enumerate(names):
        bus_id = _extract_numeric_suffix(str(name))
        if bus_id is None or int(bus_id) not in selected_set:
            continue
        f_bus = np.asarray(mat[:, col], dtype=float).reshape(-1) * f0
        delta = f_bus - f0
        out_col = f"delta_f_bus_{int(bus_id)}"
        if out_col in df.columns:
            # Multiple plotter channels can resolve to the same bus suffix.
            # Keep the first occurrence to avoid duplicate-column ambiguity.
            continue
        df[out_col] = delta
        selected_meta.append(
            {
                "bus_id": int(bus_id),
                "bus_name": str(bus_name_map.get(int(bus_id), f"Bus {int(bus_id)}")),
                "channel_name": str(name),
                "column": out_col,
            }
        )
    # Keep deterministic order by bus id.
    selected_meta = sorted(selected_meta, key=lambda x: int(x["bus_id"]))
    keep_cols = ["time_s"] + [m["column"] for m in selected_meta]
    if len(keep_cols) == 1:
        return pd.DataFrame(), []
    return df.loc[:, keep_cols], selected_meta


def _build_trace_for_run(
    run_cfg: dict[str, Any],
    cfg_path: Path,
    tds_cfg: dict[str, Any],
    contingency_cfg: dict[str, Any],
    *,
    bus_selection_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary_path = _resolve(str(run_cfg["summary_json"]), cfg_path.parent)
    summary = _load_json(summary_path)

    opt_cfg_raw = run_cfg.get("optimization_config") or summary.get("config_path")
    if not opt_cfg_raw:
        raise ValueError(f"Missing optimization_config for replay trace export: {summary_path}")
    opt_cfg = _load_yaml(_resolve(str(opt_cfg_raw), cfg_path.parent))

    label = str(run_cfg.get("label") or summary.get("formulation_name") or summary.get("formulation_id") or summary_path.stem)
    formulation_id = str(summary.get("formulation_id", "")).strip()
    scenario_id = str(summary.get("scenario_id", "")).strip()

    ss = andes.load(str(opt_cfg["system"]["case"]), setup=False)
    ss.config.freq = float(opt_cfg.get("system", {}).get("frequency_hz", 50.0))
    _apply_base_scale(ss, float(opt_cfg["scenario"]["base_scale"]))
    _apply_dispatch(ss, np.asarray(summary.get("dispatch_summary", {}).get("pg_opt", []), dtype=float))
    md_override = dict(run_cfg.get("md_override", {}) or {})
    m_override = md_override.get("m")
    d_override = md_override.get("d")
    m_values = (
        np.asarray(m_override, dtype=float)
        if m_override is not None
        else np.asarray(summary.get("dispatch_summary", {}).get("m_opt", []), dtype=float)
    )
    d_values = (
        np.asarray(d_override, dtype=float)
        if d_override is not None
        else np.asarray(summary.get("dispatch_summary", {}).get("d_opt", []), dtype=float)
    )
    _apply_md(
        ss,
        m_values,
        d_values,
    )
    _setup_pq_model(ss)

    run_cont = dict(run_cfg.get("contingency", {}) or {})
    cont_type = str(run_cont.get("type", contingency_cfg.get("type", "load_step"))).lower()
    default_step_time = float(opt_cfg["scenario"]["load_step_time"])
    load_step_time_used = None
    load_step_scale_used = None
    load_step_targets_used: list[str] | None = None
    if cont_type == "load_step":
        load_step_time_used = float(run_cont.get("time", contingency_cfg.get("time", default_step_time)))
        load_step_scale_used = float(run_cont.get("scale", contingency_cfg.get("scale", opt_cfg["scenario"]["step_scale"])))
        load_step_targets_used = [str(v) for v in list(run_cont.get("pq_targets") or contingency_cfg.get("pq_targets") or [])] or None
        if load_step_targets_used is None:
            target_bus_ids = run_cont.get("pq_target_bus_ids", contingency_cfg.get("pq_target_bus_ids"))
            if target_bus_ids:
                load_step_targets_used = _pq_names_from_bus_ids(ss, [int(v) for v in list(target_bus_ids)])
                if not load_step_targets_used:
                    raise ValueError(f"No PQ loads found for requested target buses: {target_bus_ids}")
        _add_load_step_contingency(
            ss,
            time=load_step_time_used,
            scale=load_step_scale_used,
            pq_targets=load_step_targets_used,
        )
    elif cont_type == "line_trip":
        raw_line_uid = contingency_cfg.get("line_uid")
        raw_line_dev = contingency_cfg.get("line_dev")
        line_uid = int(raw_line_uid) if raw_line_uid is not None else None
        line_dev = raw_line_dev if raw_line_dev is not None else None
        _add_line_trip_contingency(
            ss,
            time=float(contingency_cfg.get("time", default_step_time)),
            line_uid=line_uid,
            line_dev=line_dev,
        )
    elif cont_type != "none":
        raise ValueError(f"Unsupported contingency type: {cont_type}")

    ss.setup()
    ss.PFlow.run()
    _configure_tds(ss, tds_cfg)
    selected_bus_ids, bus_selection_meta = _resolve_bus_selection(
        ss, bus_selection_cfg=bus_selection_cfg, scenario_id=scenario_id, run_cfg=run_cfg
    )
    ss.TDS.init()
    success = bool(ss.TDS.run())
    ss.TDS.load_plotter()
    plotter = ss.TDS.plotter

    freq_df = _extract_frequency_trace(ss, plotter)
    bus_freq_df, bus_freq_meta = _extract_bus_frequency_traces(ss, plotter, selected_bus_ids)
    ibr_df = _extract_ibr_delta_p_traces(plotter)
    gen_load_df = _extract_genrou_load_delta_traces(plotter)
    if cont_type == "load_step" and "Delta_P_Load_total" not in gen_load_df.columns:
        pq_names = [str(v) for v in list(ss.PQ.name.v)]
        pq_p = np.asarray(ss.PQ.Ppf.v, dtype=float)
        if load_step_targets_used:
            target_set = set(load_step_targets_used)
            mask = np.asarray([name in target_set for name in pq_names], dtype=bool)
        else:
            mask = np.ones_like(pq_p, dtype=bool)
        p_total_target = float(np.nansum(pq_p[mask]))
        scale = float(load_step_scale_used) if load_step_scale_used is not None else 1.0
        step_time = float(load_step_time_used) if load_step_time_used is not None else default_step_time
        delta_load = np.where(
            np.asarray(freq_df["time_s"], dtype=float) >= step_time,
            p_total_target * (scale - 1.0),
            0.0,
        )
        gen_load_df = pd.concat([gen_load_df, pd.DataFrame({"Delta_P_Load_total": delta_load})], axis=1)
    trace_df = pd.concat([freq_df, bus_freq_df.drop(columns=["time_s"], errors="ignore"), ibr_df, gen_load_df], axis=1)
    trace_df.insert(0, "label", label)
    trace_df.insert(1, "formulation_id", formulation_id)
    trace_df.insert(2, "scenario_id", scenario_id)

    disp = dict(summary.get("dispatch_summary", {}) or {})
    meta_m = m_values if m_values.size else np.asarray(disp.get("m_opt", []), dtype=float)
    meta_d = d_values if d_values.size else np.asarray(disp.get("d_opt", []), dtype=float)
    meta = {
        "label": label,
        "formulation_id": formulation_id,
        "scenario_id": scenario_id,
        "linestyle": str(run_cfg.get("linestyle", "-")),
        "color": str(run_cfg.get("color", "")).strip(),
        "alpha": float(run_cfg.get("alpha", 1.0)),
        "linewidth": float(run_cfg.get("linewidth", 1.8)),
        "summary_json": str(summary_path),
        "optimization_config": str(_resolve(str(opt_cfg_raw), cfg_path.parent)),
        "tds_success": int(success),
        "M_1": float(meta_m[0]) if len(meta_m) > 0 else np.nan,
        "M_2": float(meta_m[1]) if len(meta_m) > 1 else np.nan,
        "M_3": float(meta_m[2]) if len(meta_m) > 2 else np.nan,
        "M_4": float(meta_m[3]) if len(meta_m) > 3 else np.nan,
        "D_1": float(meta_d[0]) if len(meta_d) > 0 else np.nan,
        "D_2": float(meta_d[1]) if len(meta_d) > 1 else np.nan,
        "D_3": float(meta_d[2]) if len(meta_d) > 2 else np.nan,
        "D_4": float(meta_d[3]) if len(meta_d) > 3 else np.nan,
        "max_abs_dev_replayed": float(np.nanmax(np.abs(pd.to_numeric(trace_df["delta_f_coi_hz"], errors="coerce")))),
        "max_abs_rocof_replayed": float(np.nanmax(np.abs(pd.to_numeric(trace_df["rocof_coi_hz_per_s"], errors="coerce")))),
        "max_abs_delta_p_ibr": float(np.nanmax(pd.to_numeric(trace_df.get("max_abs_delta_p_ibr", np.nan), errors="coerce"))),
        "selected_bus_ids": json.dumps([int(v) for v in selected_bus_ids]),
        "selected_bus_info": json.dumps(bus_freq_meta),
        "bus_selection_meta": json.dumps(bus_selection_meta),
    }
    return trace_df, meta


def _plot_trace_panel(
    trace_frames: list[pd.DataFrame],
    meta_rows: list[dict[str, Any]],
    out_path: Path,
    style_cfg: dict[str, Any],
    *,
    panel_title: str = "Replay traces for selected formulations",
    frequency_hz: float = 50.0,
    delta_f_limit_hz: float = 0.8,
    rocof_limit_hz_per_s: float = 1.0,
    plot_mode: str = "combined",
    ibr_scale: float = 1.0,
    ibr_unit: str = "p.u.",
) -> None:
    _apply_style_rcparams(style_cfg)
    replay_palette = _style_palette(style_cfg, "replay")
    colors = [
        replay_palette.get("default", "#1f77b4"),
        replay_palette.get("comparison", "#ff7f0e"),
        replay_palette.get("local", "#d62728"),
        replay_palette.get("tertiary", "#2ca02c"),
    ]
    font_sizes = _style_font_sizes(style_cfg)
    font_label = font_sizes["axis_label"]
    font_tick = font_sizes["tick"]
    font_legend = font_sizes["legend"]
    font_title = font_sizes["title"]
    line_cfg = _style_line_cfg(style_cfg)
    limit_color = str(style_cfg.get("style", {}).get("accents", {}).get("constraint_limit", "#c46646"))
    limit_style = str(line_cfg.get("limit_style", "--"))
    limit_width = float(line_cfg.get("limit_width", 1.0))
    plot_mode = str(plot_mode or "combined").lower()
    if plot_mode == "ibr_response":
        plot_mode = "ibr_only"

    if plot_mode == "frequency_only":
        fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True, constrained_layout=True)
        ax_df, ax_rocof = axes
        for idx, (df, meta) in enumerate(zip(trace_frames, meta_rows)):
            color = colors[idx % len(colors)]
            label = str(meta["label"])
            linestyle = str(meta.get("linestyle", "-"))
            color_cfg = str(meta.get("color", "")).strip()
            if color_cfg:
                color = color_cfg
            alpha = float(meta.get("alpha", 1.0))
            linewidth = float(meta.get("linewidth", 2.0))
            ax_df.plot(df["time_s"], df["delta_f_coi_hz"], label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            ax_rocof.plot(df["time_s"], df["rocof_coi_hz_per_s"], label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        ax_df.axhline(delta_f_limit_hz, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_df.axhline(-delta_f_limit_hz, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_rocof.axhline(rocof_limit_hz_per_s, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_rocof.axhline(-rocof_limit_hz_per_s, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_df.set_ylabel(r"$\Delta f_{\mathrm{COI}}$ [Hz]")
        ax_rocof.set_ylabel(r"RoCoF$_{\mathrm{COI}}$ [Hz/s]")
        ax_rocof.set_xlabel("Time [s]")
        for ax in axes.ravel():
            ax.grid(alpha=0.18)
        handles, labels = ax_df.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.03), fontsize=font_legend, frameon=False)
        for ax in axes.ravel():
            ax.tick_params(labelsize=font_tick)
            ax.xaxis.label.set_size(font_label)
            ax.yaxis.label.set_size(font_label)
    elif plot_mode == "ibr_only":
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.8), sharex=True, constrained_layout=True)
        ibr_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
        for idx, (df, meta) in enumerate(zip(trace_frames, meta_rows)):
            color = colors[idx % len(colors)]
            label = str(meta["label"])
            linestyle = str(meta.get("linestyle", "-"))
            color_cfg = str(meta.get("color", "")).strip()
            if color_cfg:
                color = color_cfg
            alpha = float(meta.get("alpha", 1.0))
            linewidth = float(meta.get("linewidth", 2.0))
            for i, ax in enumerate(ibr_axes, start=1):
                col = f"Delta_P_IBR_{i}"
                if col in df.columns:
                    ax.plot(df["time_s"], pd.to_numeric(df[col], errors="coerce") * ibr_scale, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        for i, ax in enumerate(ibr_axes, start=1):
            ax.set_ylabel(rf"$\Delta P_{{\mathrm{{IBR}},{i}}}$ [{ibr_unit}]")
            ax.grid(alpha=0.18)
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 1].set_xlabel("Time [s]")
        handles, labels = ibr_axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.03), fontsize=font_legend, frameon=False)
        for ax in axes.ravel():
            ax.tick_params(labelsize=font_tick)
            ax.xaxis.label.set_size(font_label)
            ax.yaxis.label.set_size(font_label)
    elif plot_mode == "bus_frequency":
        fig, axes = plt.subplots(2, 1, figsize=(11.2, 6.4), sharex=True, constrained_layout=True)
        ax_df, ax_rocof = axes
        for idx, (df, meta) in enumerate(zip(trace_frames, meta_rows)):
            color = colors[idx % len(colors)]
            label = str(meta["label"])
            linestyle = str(meta.get("linestyle", "-"))
            color_cfg = str(meta.get("color", "")).strip()
            if color_cfg:
                color = color_cfg
            alpha = float(meta.get("alpha", 1.0))
            linewidth = float(meta.get("linewidth", 2.0))
            ax_df.plot(
                df["time_s"],
                pd.to_numeric(df["delta_f_coi_hz"], errors="coerce"),
                label=f"{label} (COI)",
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
            for col in sorted([c for c in df.columns if str(c).startswith("delta_f_bus_")], key=lambda x: int(str(x).split("_")[-1])):
                bus_id = int(str(col).split("_")[-1])
                ax_df.plot(
                    df["time_s"],
                    pd.to_numeric(df[col], errors="coerce"),
                    label=f"{label} (Bus {bus_id})",
                    color=color,
                    linewidth=max(1.2, linewidth - 0.2),
                    linestyle="--",
                    alpha=min(1.0, alpha * 0.95),
                )
            ax_rocof.plot(
                df["time_s"],
                pd.to_numeric(df["rocof_coi_hz_per_s"], errors="coerce"),
                label=label,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
            )
        ax_df.axhline(delta_f_limit_hz, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_df.axhline(-delta_f_limit_hz, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_rocof.axhline(rocof_limit_hz_per_s, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_rocof.axhline(-rocof_limit_hz_per_s, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_df.set_ylabel(r"$\Delta f$ [Hz]")
        ax_rocof.set_ylabel(r"RoCoF$_{\mathrm{COI}}$ [Hz/s]")
        ax_rocof.set_xlabel("Time [s]")
        for ax in axes.ravel():
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=font_tick)
            ax.xaxis.label.set_size(font_label)
            ax.yaxis.label.set_size(font_label)
        handles, labels = ax_df.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(max(2, len(labels) // 2), 4), bbox_to_anchor=(0.5, 1.04), fontsize=font_legend, frameon=False)
    else:
        fig = plt.figure(figsize=(12.8, 10.6), constrained_layout=True)
        gs = fig.add_gridspec(4, 2)
        ax_df = fig.add_subplot(gs[0, 0])
        ax_rocof = fig.add_subplot(gs[0, 1], sharex=ax_df)
        ax_p1 = fig.add_subplot(gs[1, 0], sharex=ax_df)
        ax_p2 = fig.add_subplot(gs[1, 1], sharex=ax_df)
        ax_p3 = fig.add_subplot(gs[2, 0], sharex=ax_df)
        ax_p4 = fig.add_subplot(gs[2, 1], sharex=ax_df)
        ax_bottom = fig.add_subplot(gs[3, :], sharex=ax_df)
        ax_bottom_r = ax_bottom.twinx()
        axes = np.asarray([[ax_df, ax_rocof], [ax_p1, ax_p2], [ax_p3, ax_p4]])
        for idx, (df, meta) in enumerate(zip(trace_frames, meta_rows)):
            color = colors[idx % len(colors)]
            label = str(meta["label"])
            linestyle = str(meta.get("linestyle", "-"))
            color_cfg = str(meta.get("color", "")).strip()
            if color_cfg:
                color = color_cfg
            alpha = float(meta.get("alpha", 1.0))
            linewidth = float(meta.get("linewidth", 1.8))
            ax_df.plot(df["time_s"], df["delta_f_coi_hz"], label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            ax_rocof.plot(df["time_s"], df["rocof_coi_hz_per_s"], label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            if "Delta_P_IBR_1" in df.columns:
                ax_p1.plot(df["time_s"], pd.to_numeric(df["Delta_P_IBR_1"], errors="coerce") * ibr_scale, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            if "Delta_P_IBR_2" in df.columns:
                ax_p2.plot(df["time_s"], pd.to_numeric(df["Delta_P_IBR_2"], errors="coerce") * ibr_scale, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            if "Delta_P_IBR_3" in df.columns:
                ax_p3.plot(df["time_s"], pd.to_numeric(df["Delta_P_IBR_3"], errors="coerce") * ibr_scale, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            if "Delta_P_IBR_4" in df.columns:
                ax_p4.plot(df["time_s"], pd.to_numeric(df["Delta_P_IBR_4"], errors="coerce") * ibr_scale, label=label, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            if "Delta_P_GENROU_total" in df.columns:
                ax_bottom.plot(
                    df["time_s"],
                    pd.to_numeric(df["Delta_P_GENROU_total"], errors="coerce") * 100.0,
                    label=label,
                    color=color,
                    linewidth=max(1.8, linewidth),
                    linestyle=linestyle,
                    alpha=alpha,
                )
            if "Delta_P_Load_total" in df.columns:
                ax_bottom_r.plot(
                    df["time_s"],
                    pd.to_numeric(df["Delta_P_Load_total"], errors="coerce") * 100.0,
                    label=label,
                    color=color,
                    linewidth=max(1.8, linewidth),
                    linestyle="--",
                    alpha=alpha,
                )

        ax_df.axhline(delta_f_limit_hz, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_df.axhline(-delta_f_limit_hz, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_rocof.axhline(rocof_limit_hz_per_s, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_rocof.axhline(-rocof_limit_hz_per_s, color=limit_color, linestyle=limit_style, linewidth=limit_width)
        ax_df.set_ylabel(r"$\Delta f_{\mathrm{COI}}$ [Hz]")
        ax_rocof.set_ylabel(r"RoCoF$_{\mathrm{COI}}$ [Hz/s]")
        ax_p1.set_ylabel(rf"$\Delta P_{{\mathrm{{IBR}},1}}$ [{ibr_unit}]")
        ax_p2.set_ylabel(rf"$\Delta P_{{\mathrm{{IBR}},2}}$ [{ibr_unit}]")
        ax_p3.set_ylabel(rf"$\Delta P_{{\mathrm{{IBR}},3}}$ [{ibr_unit}]")
        ax_p4.set_ylabel(rf"$\Delta P_{{\mathrm{{IBR}},4}}$ [{ibr_unit}]")
        ax_bottom.set_ylabel(r"$\Delta P_{\mathrm{GENROU,tot}}$ [MW]")
        ax_bottom_r.set_ylabel(r"$\Delta P_{\mathrm{Load,tot}}$ [MW]")
        ax_bottom.set_xlabel("Time [s]")
        for ax in axes.ravel():
            ax.grid(alpha=0.18)
        ax_bottom.grid(alpha=0.18)
        ax_bottom_r.grid(False)
        ax_bottom.spines["left"].set_position(("outward", 2))
        ax_bottom_r.spines["right"].set_position(("outward", 8))
        for ax in [*list(axes.ravel()), ax_bottom, ax_bottom_r]:
            ax.tick_params(labelsize=font_tick)
            ax.xaxis.label.set_size(font_label)
            ax.yaxis.label.set_size(font_label)
        handles, labels = ax_df.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), bbox_to_anchor=(0.5, 1.045), fontsize=font_legend, frameon=False)

        out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_path.suffix.lower() != ".png":
        fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export lightweight replay traces for selected optimization runs.")
    parser.add_argument("--config", required=True, help="YAML config describing the runs to replay.")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_yaml(cfg_path)
    runs = list(cfg.get("runs") or [])
    if not runs:
        raise ValueError(f"No runs configured in {cfg_path}")

    tds_cfg = dict(cfg.get("tds", {}) or {})
    contingency_cfg = dict(cfg.get("contingency", {}) or {})
    output_cfg = dict(cfg.get("output", {}) or {})
    style_cfg = _load_style_config(output_cfg.get("style_config"), cfg_path)
    limits_cfg = dict(cfg.get("limits", {}) or output_cfg.get("limits", {}) or {})
    out_dir = _resolve(str(output_cfg.get("directory", "results/thesis_optimization_results/local_validation/replay_trace_panel")), cfg_path.parent)
    panel_title = str(output_cfg.get("panel_title", output_cfg.get("title", "Replay traces for selected formulations")))
    plot_mode = str(output_cfg.get("plot_mode", "combined"))
    ibr_scale = float(output_cfg.get("ibr_scale", 1.0))
    ibr_unit = str(output_cfg.get("ibr_unit", "p.u."))
    bus_selection_cfg = dict(output_cfg.get("bus_selection", {}) or {})
    frequency_hz = float(limits_cfg.get("frequency_hz", 50.0))
    delta_f_limit_hz = float(limits_cfg.get("delta_f_hz", 0.8))
    rocof_limit_hz_per_s = float(limits_cfg.get("rocof_hz_per_s", 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_frames: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    for run_cfg in runs:
        trace_df, meta = _build_trace_for_run(
            dict(run_cfg or {}),
            cfg_path,
            tds_cfg,
            contingency_cfg,
            bus_selection_cfg=bus_selection_cfg,
        )
        stem = _slug(str(meta["label"]))
        trace_path = out_dir / f"{stem}_trace.csv"
        trace_df.to_csv(trace_path, index=False)
        meta["trace_csv"] = str(trace_path)
        trace_frames.append(trace_df)
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_dir / "trace_summary.csv", index=False)
    generated_panels: list[str] = []
    if str(plot_mode).lower() == "topk_dual":
        freq_out = out_dir / "replay_trace_frequency_panel.pdf"
        ibr_out = out_dir / "replay_trace_ibr_panel.pdf"
        _plot_trace_panel(
            trace_frames,
            meta_rows,
            freq_out,
            style_cfg=style_cfg,
            panel_title=panel_title,
            frequency_hz=frequency_hz,
            delta_f_limit_hz=delta_f_limit_hz,
            rocof_limit_hz_per_s=rocof_limit_hz_per_s,
            plot_mode="bus_frequency",
            ibr_scale=ibr_scale,
            ibr_unit=ibr_unit,
        )
        _plot_trace_panel(
            trace_frames,
            meta_rows,
            ibr_out,
            style_cfg=style_cfg,
            panel_title=panel_title,
            frequency_hz=frequency_hz,
            delta_f_limit_hz=delta_f_limit_hz,
            rocof_limit_hz_per_s=rocof_limit_hz_per_s,
            plot_mode="ibr_response",
            ibr_scale=ibr_scale,
            ibr_unit=ibr_unit,
        )
        generated_panels.extend([str(freq_out), str(ibr_out)])
    else:
        out_path = out_dir / "replay_trace_panel.pdf"
        _plot_trace_panel(
            trace_frames,
            meta_rows,
            out_path,
            style_cfg=style_cfg,
            panel_title=panel_title,
            frequency_hz=frequency_hz,
            delta_f_limit_hz=delta_f_limit_hz,
            rocof_limit_hz_per_s=rocof_limit_hz_per_s,
            plot_mode=plot_mode,
            ibr_scale=ibr_scale,
            ibr_unit=ibr_unit,
        )
        generated_panels.append(str(out_path))
    with (out_dir / "trace_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": str(cfg_path), "runs": meta_rows, "generated_panels": generated_panels}, f, indent=2)

    print(f"[replay_trace_panel] Wrote: {out_dir / 'trace_summary.csv'}")
    for p in generated_panels:
        print(f"[replay_trace_panel] Wrote: {p}")


if __name__ == "__main__":
    main()
