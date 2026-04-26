from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import andes


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation import (  # type: ignore
    configure_tds,
    define_operating_point,
    pick_line_contingencies,
    select_step_targets,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_style_config(cfg: dict[str, Any]) -> dict[str, Any]:
    style_path_raw = cfg.get("plot_style_config", "configs/figure/thesis_plot_style.yaml")
    style_path = REPO_ROOT / str(style_path_raw)
    if not style_path.exists():
        return {}
    return load_yaml(style_path)


def _style_palette(style_cfg: dict[str, Any], palette_name: str) -> dict[str, str]:
    return dict(style_cfg.get("style", {}).get("palettes", {}).get(palette_name, {}) or {})


def _style_accent(style_cfg: dict[str, Any], key: str, default: str) -> str:
    return str(style_cfg.get("style", {}).get("accents", {}).get(key, default))


def add_measurement_devices(ss: andes.System) -> None:
    existing_rocof = set()
    if getattr(ss, "BusROCOF", None) is not None and getattr(ss.BusROCOF, "n", 0) > 0:
        try:
            existing_rocof = {str(value) for value in list(ss.BusROCOF.idx.v)}
        except Exception:
            existing_rocof = set()

    for bus in ss.Bus.as_df().idx.values:
        idx = f"BusROCOF_{bus}"
        if idx in existing_rocof:
            continue
        ss.add(
            model="BusROCOF",
            idx=idx,
            name=f"BusROCOF {bus}",
            param_dict=dict(bus=bus, Tr=0.02, Tw=0.1, Tf=0.02),
        )

    existing_pmus = list(ss.PMU.as_df().bus.values) if getattr(ss, "PMU", None) is not None and ss.PMU.n > 0 else []
    for bus in ss.Bus.as_df().idx.values:
        if bus not in existing_pmus:
            ss.add(model="PMU", param_dict=dict(bus=bus))


def set_thesis_style(style_cfg: dict[str, Any] | None = None) -> None:
    rc_from_cfg = dict((style_cfg or {}).get("style", {}).get("matplotlib", {}).get("rc_params", {}) or {})
    rc_defaults = {
        "figure.figsize": (7.0, 4.4),
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Serif",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.color": "#8a8179",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#4b433d",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 10.5,
        "axes.labelcolor": "#2b2826",
        "axes.labelpad": 4.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "legend.borderaxespad": 0.6,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "lines.linewidth": 1.8,
        "patch.edgecolor": "#2b2826",
        "patch.linewidth": 0.6,
    }
    rc_defaults.update(rc_from_cfg)
    plt.rcParams.update(rc_defaults)


def write_parquet_compat(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        pass


def save_plot_data(df: pd.DataFrame, stem: str, plot_data_dir: Path) -> None:
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plot_data_dir / f"{stem}.csv"
    parquet_path = plot_data_dir / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    write_parquet_compat(df, parquet_path)


def save_table(df: pd.DataFrame, stem: str, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{stem}.csv"
    parquet_path = tables_dir / f"{stem}.parquet"
    df.to_csv(csv_path, index=False)
    write_parquet_compat(df, parquet_path)


def save_figure(fig: plt.Figure, stem: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_case_tables(case_path: Path) -> dict[str, pd.DataFrame]:
    sheets = {}
    for sheet in ["Bus", "Line", "PQ", "PV", "Slack", "REGCV1"]:
        sheets[sheet] = pd.read_excel(case_path, sheet_name=sheet)
    return sheets


def load_regcv1_device_bases(case_tables: dict[str, pd.DataFrame]) -> dict[int, float]:
    regcv1 = case_tables["REGCV1"].copy()
    bases: dict[int, float] = {}
    for _, row in regcv1.iterrows():
        idx = str(row["idx"])
        if not idx.startswith("REGCV1_"):
            continue
        unit_id = int(idx.split("_")[-1])
        bases[unit_id] = float(row["Sn"])
    return bases


def build_graph(ss: andes.System) -> nx.Graph:
    graph = nx.Graph()
    bus_df = ss.Bus.as_df()
    line_df = ss.Line.as_df() if getattr(ss, "Line", None) is not None else None
    for _, row in bus_df.iterrows():
        graph.add_node(int(row["idx"]), name=str(row.get("name", row["idx"])))
    if line_df is not None and not line_df.empty:
        for line_pos, (_, row) in enumerate(line_df.iterrows()):
            b1 = int(row["bus1"])
            b2 = int(row["bus2"])
            if b1 in graph.nodes and b2 in graph.nodes:
                graph.add_edge(
                    b1,
                    b2,
                    line_pos=line_pos,
                    line_idx=str(row.get("idx", f"Line_{line_pos + 1}")),
                )
    return graph


def positions_for_graph(graph: nx.Graph, layout: str) -> dict[int, np.ndarray]:
    key = str(layout).lower()
    if key == "spring":
        return nx.spring_layout(graph, seed=42, k=0.8)
    if key == "circular":
        return nx.circular_layout(graph)
    if key == "shell":
        return nx.shell_layout(graph)
    return nx.kamada_kawai_layout(graph)


def resolve_outage_severity_table(benchmark_cfg: dict[str, Any]) -> pd.DataFrame:
    line_cfg = dict(benchmark_cfg.get("line_outage", {}) or {})
    severity_cfg = dict(line_cfg.get("severity_proxy", {}) or {})
    weights = dict(severity_cfg.get("weights") or {})
    if not weights:
        weights = {
            "pre_fault_loading": 0.40,
            "ptdf_l1_norm_outaged_line": 0.35,
            "max_abs_lodf_row": 0.25,
        }

    csv_path = REPO_ROOT / str(
        severity_cfg.get(
            "source_csv",
            "results/thesis_data_generation_results/results/line_outages_only/simulation_results.csv",
        )
    )
    header = pd.read_csv(csv_path, nrows=0)
    usecols = ["line_uid"] + [col for col in weights if col in header.columns]
    df = pd.read_csv(csv_path, usecols=usecols)
    grouped = df.groupby("line_uid", dropna=False).median(numeric_only=True).reset_index()
    grouped["line_uid"] = pd.to_numeric(grouped["line_uid"], errors="coerce").astype("Int64")
    grouped = grouped.loc[grouped["line_uid"].notna()].copy()
    grouped["line_uid"] = grouped["line_uid"].astype(int)

    grouped["severity_score"] = 0.0
    for column, weight in weights.items():
        ranks = grouped[column].rank(method="average", pct=True)
        grouped[f"{column}_rank"] = ranks
        grouped["severity_score"] += float(weight) * ranks.fillna(0.0)

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
    return grouped


def select_screened_outages(severity_df: pd.DataFrame, benchmark_cfg: dict[str, Any]) -> pd.DataFrame:
    line_cfg = dict(benchmark_cfg.get("line_outage", {}) or {})
    target_count = int(line_cfg.get("n_screened_outages", 10))
    excluded = {
        int(value)
        for value in list(line_cfg.get("exclude_line_uids") or [])
        if pd.notna(value)
    }
    candidates = severity_df.loc[~severity_df["line_uid"].isin(excluded)].copy()
    if candidates.empty:
        return candidates

    bins = sorted(candidates["severity_bin"].dropna().astype(str).unique().tolist())
    per_bin_target = max(1, math.ceil(target_count / max(len(bins), 1)))
    picks: list[pd.DataFrame] = []
    for severity_bin in bins:
        subset = candidates.loc[candidates["severity_bin"].astype(str) == severity_bin].copy()
        subset = subset.sort_values(["severity_score", "line_uid"], ascending=[False, True])
        picks.append(subset.head(per_bin_target))

    selected = pd.concat(picks, ignore_index=True, sort=False) if picks else candidates.iloc[0:0].copy()
    selected = selected.drop_duplicates(subset=["line_uid"]).copy()
    if selected.shape[0] < target_count:
        remaining = candidates.loc[~candidates["line_uid"].isin(selected["line_uid"])].copy()
        remaining = remaining.sort_values(["severity_score", "line_uid"], ascending=[False, True])
        selected = pd.concat(
            [selected, remaining.head(target_count - selected.shape[0])],
            ignore_index=True,
            sort=False,
        )
    selected = selected.sort_values(["severity_bin", "severity_score", "line_uid"], ascending=[True, False, True])
    return selected.head(target_count).reset_index(drop=True)


def network_frames(case_path: Path, layout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ss = andes.load(str(case_path), setup=False)
    ss.setup()
    graph = build_graph(ss)
    positions = positions_for_graph(graph, layout)

    bus_df = ss.Bus.as_df().copy()
    node_rows: list[dict[str, Any]] = []
    for _, row in bus_df.iterrows():
        bus = int(row["idx"])
        x, y = positions[bus]
        node_rows.append(
            {
                "layer": "node",
                "bus": bus,
                "bus_name": str(row.get("name", bus)),
                "x": float(x),
                "y": float(y),
            }
        )
    edge_rows: list[dict[str, Any]] = []
    line_df = ss.Line.as_df().copy()
    for uid, row in line_df.iterrows():
        b1 = int(row["bus1"])
        b2 = int(row["bus2"])
        x1, y1 = positions[b1]
        x2, y2 = positions[b2]
        edge_rows.append(
            {
                "layer": "edge",
                "line_uid": int(uid),
                "line_idx": str(row.get("idx", "")),
                "bus1": b1,
                "bus2": b2,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def base_network_metadata(case_tables: dict[str, pd.DataFrame], node_df: pd.DataFrame, edge_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus_df = case_tables["Bus"][["idx", "name"]].copy()
    pq_df = case_tables["PQ"][["bus", "owner"]].copy().rename(columns={"owner": "zone_owner"})
    pq_load_df = (
        case_tables["PQ"][["bus", "p0"]]
        .copy()
        .groupby("bus", as_index=False)["p0"]
        .sum()
        .rename(columns={"p0": "p_load_pu"})
    )
    pv_gen_df = (
        case_tables["PV"][["bus", "p0"]]
        .copy()
        .groupby("bus", as_index=False)["p0"]
        .sum()
        .rename(columns={"p0": "p_gen_pu"})
    )
    slack_gen_df = (
        case_tables["Slack"][["bus", "p0"]]
        .copy()
        .groupby("bus", as_index=False)["p0"]
        .sum()
        .rename(columns={"p0": "p_slack_pu"})
    )
    reg_df = case_tables["REGCV1"][["idx", "bus", "Sn"]].copy()
    reg_df["ibr_unit"] = reg_df["idx"].astype(str).str.split("_").str[-1].astype(int)
    reg_df = reg_df.rename(columns={"Sn": "ibr_base_mva"})
    slack_buses = set(case_tables["Slack"]["bus"].astype(int).tolist())
    syngen_buses = set(case_tables["PV"]["bus"].astype(int).tolist()) | slack_buses

    nodes = node_df.merge(bus_df, left_on="bus", right_on="idx", how="left")
    nodes = nodes.drop(columns=["idx"])
    nodes = nodes.merge(pq_df, left_on="bus", right_on="bus", how="left")
    nodes = nodes.merge(pq_load_df, on="bus", how="left")
    nodes = nodes.merge(pv_gen_df, on="bus", how="left")
    nodes = nodes.merge(slack_gen_df, on="bus", how="left")
    nodes = nodes.merge(reg_df[["bus", "ibr_unit", "ibr_base_mva"]], on="bus", how="left")
    nodes["is_load_bus"] = nodes["zone_owner"].notna()
    nodes["is_ibr_bus"] = nodes["ibr_unit"].notna()
    nodes["zone_owner"] = nodes["zone_owner"].astype("Int64")
    nodes["p_load_pu"] = pd.to_numeric(nodes["p_load_pu"], errors="coerce").fillna(0.0)
    nodes["p_gen_pu"] = (
        pd.to_numeric(nodes["p_gen_pu"], errors="coerce").fillna(0.0)
        + pd.to_numeric(nodes["p_slack_pu"], errors="coerce").fillna(0.0)
    )
    nodes["is_slack_bus"] = nodes["bus"].astype(int).isin(slack_buses)
    nodes["is_syngen_bus"] = nodes["bus"].astype(int).isin(syngen_buses)
    nodes["node_category"] = "Other bus"
    nodes.loc[nodes["p_load_pu"] > 0.0, "node_category"] = "PQ"
    nodes.loc[nodes["is_syngen_bus"], "node_category"] = "SynGen"
    nodes.loc[nodes["is_ibr_bus"], "node_category"] = "IBR"
    nodes.loc[nodes["is_slack_bus"], "node_category"] = "Slack"

    line_df = case_tables["Line"][["uid", "Sn", "rate_a", "rate_b", "rate_c"]].copy()
    edges = edge_df.merge(line_df, left_on="line_uid", right_on="uid", how="left")
    edges = edges.drop(columns=["uid"])
    # For thesis Figure 3.3, use the line apparent-power base Sn from the case
    # as the limit reference, as requested.
    edges["thermal_limit_mva"] = pd.to_numeric(edges["Sn"], errors="coerce")
    edges["limit_bin"] = pd.cut(edges["thermal_limit_mva"], bins=4, labels=["L1", "L2", "L3", "L4"], include_lowest=True)
    return nodes, edges


def build_network_plot_data(nodes: pd.DataFrame, edges: pd.DataFrame, *, figure_id: str) -> pd.DataFrame:
    node_cols = [
        "layer",
        "bus",
        "bus_name",
        "x",
        "y",
        "zone_owner",
        "is_load_bus",
        "is_ibr_bus",
        "ibr_unit",
        "ibr_base_mva",
    ]
    edge_cols = [
        "layer",
        "line_uid",
        "line_idx",
        "bus1",
        "bus2",
        "x1",
        "y1",
        "x2",
        "y2",
        "thermal_limit_mva",
        "limit_bin",
    ]
    node_block = nodes[node_cols].copy()
    edge_block = edges[edge_cols].copy()
    node_block["figure_id"] = figure_id
    edge_block["figure_id"] = figure_id
    return pd.concat([node_block, edge_block], ignore_index=True, sort=False)


def configure_and_run_simulation(
    *,
    case_path: Path,
    nominal_frequency_hz: float,
    tds_cfg: dict[str, Any],
    base_scale: float,
    scale_pv: bool,
    m_values: np.ndarray,
    d_values: np.ndarray,
    load_step_scale: float | None = None,
    load_step_time: float | None = None,
    load_step_owners: list[str] | None = None,
    line_uid: int | None = None,
    line_trip_time: float | None = None,
) -> andes.System:
    ss = andes.load(str(case_path), setup=False)
    ss.config.freq = float(nominal_frequency_hz)
    add_measurement_devices(ss)
    define_operating_point(
        ss,
        base_scale=float(base_scale),
        M_vec=np.asarray(m_values, dtype=float),
        D_vec=np.asarray(d_values, dtype=float),
        ed_cfg={"enable": False},
        scale_pv=bool(scale_pv),
    )

    if load_step_scale is not None and load_step_time is not None and not math.isclose(float(load_step_scale), 1.0):
        load_cfg = {"owners": list(load_step_owners or [])}
        for dev in select_step_targets(ss, load_cfg, rng=None):
            ss.add(
                model="Alter",
                param_dict={
                    "t": float(load_step_time),
                    "model": "PQ",
                    "dev": dev,
                    "src": "Ppf",
                    "attr": "v",
                    "method": "*",
                    "amount": float(load_step_scale),
                },
            )
            ss.add(
                model="Alter",
                param_dict={
                    "t": float(load_step_time),
                    "model": "PQ",
                    "dev": dev,
                    "src": "Qpf",
                    "attr": "v",
                    "method": "*",
                    "amount": float(load_step_scale),
                },
            )

    if line_uid is not None:
        ss_pick = andes.load(str(case_path), setup=False)
        picked = pick_line_contingencies(
            ss_pick,
            {"line_ids": [str(int(line_uid))], "max_lines": 1},
            np.random.default_rng(0),
        )
        if not picked:
            raise ValueError(f"Could not resolve line_uid={line_uid} into a valid contingency.")
        ss.add(
            model="Toggle",
            param_dict={
                "t": float(line_trip_time if line_trip_time is not None else 1.0),
                "model": "Line",
                "dev": picked[0]["idx"],
            },
        )

    ss.setup()
    ss.PFlow.run()
    configure_tds(ss, tds_cfg)
    ss.TDS.init()
    success = bool(ss.TDS.run())
    ss.TDS.load_plotter()
    if not success:
        raise RuntimeError("Dynamic simulation did not converge.")
    return ss


def extract_response_frame(
    ss: andes.System,
    *,
    schedule_id: str,
    schedule_label: str,
    nominal_frequency_hz: float,
    system_base_mva: float,
    device_bases: dict[int, float],
) -> pd.DataFrame:
    plotter = ss.TDS.plotter
    time = np.asarray(plotter.get_values(0), dtype=float).reshape(-1)
    rows: list[dict[str, Any]] = []

    coi_idx = list(plotter.find("omega COI", idx_only=True))
    if not coi_idx:
        raise RuntimeError("No COI channel found in plotter output.")
    f_coi_hz = np.asarray(plotter.get_values([int(coi_idx[0])]), dtype=float).reshape(-1) * float(nominal_frequency_hz)
    rocof_hz_s = np.gradient(f_coi_hz, time, edge_order=2 if time.size > 2 else 1)
    for t, f_hz, r_hz_s in zip(time, f_coi_hz, rocof_hz_s):
        rows.append(
            {
                "schedule_id": schedule_id,
                "schedule_label": schedule_label,
                "time_s": float(t),
                "series_type": "frequency_dev_hz",
                "series_id": "coi",
                "unit": "Hz",
                "value": float(f_hz - nominal_frequency_hz),
            }
        )
        rows.append(
            {
                "schedule_id": schedule_id,
                "schedule_label": schedule_label,
                "time_s": float(t),
                "series_type": "rocof_hz_s",
                "series_id": "coi",
                "unit": "Hz/s",
                "value": float(r_hz_s),
            }
        )

    ibr_indices = list(plotter.find("Pe REGCV1", idx_only=True))
    for unit_id, idx in enumerate(ibr_indices, start=1):
        series = np.asarray(plotter.get_values([int(idx)]), dtype=float).reshape(-1)
        if series.size != time.size or series.size == 0:
            continue
        delta_system_base = series - float(series[0])
        unit_base = float(device_bases.get(unit_id, system_base_mva))
        delta_ibr_base = delta_system_base * (float(system_base_mva) / unit_base)
        for t, value in zip(time, delta_ibr_base):
            rows.append(
                {
                    "schedule_id": schedule_id,
                    "schedule_label": schedule_label,
                    "time_s": float(t),
                    "series_type": "delta_p_ibr_pu",
                    "series_id": f"IBR_{unit_id}",
                    "unit": "p.u. on IBR base",
                    "value": float(value),
                }
            )

    return pd.DataFrame(rows)


def plot_figure_1_2(
    cfg: dict[str, Any],
    *,
    style_cfg: dict[str, Any],
    case_path: Path,
    figures_dir: Path,
    plot_data_dir: Path,
    tables_dir: Path,
    device_bases: dict[int, float],
) -> None:
    fig_cfg = dict(cfg.get("figure_1_2", {}) or {})
    nominal_frequency_hz = float(cfg["system"]["nominal_frequency_hz"])
    system_base_mva = float(cfg["system"]["system_base_mva"])
    regcv1 = pd.read_excel(case_path, sheet_name="REGCV1")
    n_ibr = int(regcv1.shape[0])

    responses: list[pd.DataFrame] = []
    schedule_rows: list[dict[str, Any]] = []
    for schedule in list(fig_cfg.get("schedules") or []):
        m_val = float(schedule["M"])
        d_val = float(schedule["D"])
        schedule_id = str(schedule["schedule_id"])
        label = str(schedule["label"])
        schedule_rows.append({"schedule_id": schedule_id, "schedule_label": label, "M": m_val, "D": d_val})

        ss = configure_and_run_simulation(
            case_path=case_path,
            nominal_frequency_hz=nominal_frequency_hz,
            tds_cfg=dict(cfg.get("tds", {}) or {}),
            base_scale=float(fig_cfg.get("base_scale", 1.0)),
            scale_pv=bool(fig_cfg.get("scale_pv", False)),
            m_values=np.full(n_ibr, m_val, dtype=float),
            d_values=np.full(n_ibr, d_val, dtype=float),
            load_step_scale=float(fig_cfg.get("load_step_scale", 1.0)),
            load_step_time=float(fig_cfg.get("load_step_time", 1.0)),
        )
        response = extract_response_frame(
            ss,
            schedule_id=schedule_id,
            schedule_label=label,
            nominal_frequency_hz=nominal_frequency_hz,
            system_base_mva=system_base_mva,
            device_bases=device_bases,
        )
        response["M"] = m_val
        response["D"] = d_val
        responses.append(response)

    response_df = pd.concat(responses, ignore_index=True, sort=False)
    focus_ibr = int(fig_cfg.get("focus_ibr_unit", 4))
    disturbance_time = float(fig_cfg.get("load_step_time", 1.0))

    # Align traces at disturbance time so pre-disturbance stays flat and
    # post-disturbance differences reflect scheduling/headroom effects.
    aligned_blocks: list[pd.DataFrame] = []
    key_cols = ["schedule_id", "series_type", "series_id"]
    for _, block in response_df.groupby(key_cols, sort=False):
        block = block.sort_values("time_s").copy()
        nearest_idx = (block["time_s"] - disturbance_time).abs().idxmin()
        baseline = float(block.loc[nearest_idx, "value"])
        block["time_rel_s"] = block["time_s"] - disturbance_time
        block["value_aligned"] = block["value"] - baseline
        block.loc[block["time_rel_s"] < 0.0, "value_aligned"] = 0.0
        aligned_blocks.append(block)
    response_df = pd.concat(aligned_blocks, ignore_index=True, sort=False)

    save_plot_data(response_df, "fig_1_2_simulation_intuition", plot_data_dir)
    save_table(pd.DataFrame(schedule_rows), "fig_1_2_schedule_summary", tables_dir)

    schedule_palette = _style_palette(style_cfg, "schedule")
    fallback_schedule_colors = ["#355070", "#6d597a", "#b56576"]
    colors: dict[str, str] = {}
    for idx, row in enumerate(schedule_rows):
        sid = str(row["schedule_id"])
        colors[sid] = schedule_palette.get(sid, fallback_schedule_colors[idx % len(fallback_schedule_colors)])
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 5.9), sharex=True, constrained_layout=True)
    top_df = response_df.loc[
        (response_df["series_type"] == "delta_p_ibr_pu")
        & (response_df["series_id"] == f"IBR_{focus_ibr}")
    ].copy()
    bottom_df = response_df.loc[response_df["series_type"] == "frequency_dev_hz"].copy()

    for schedule in schedule_rows:
        subset_top = top_df.loc[top_df["schedule_id"] == schedule["schedule_id"]]
        subset_bottom = bottom_df.loc[bottom_df["schedule_id"] == schedule["schedule_id"]]
        label = str(schedule["schedule_label"])
        axes[0].plot(
            subset_top["time_rel_s"],
            subset_top["value_aligned"],
            label=label,
            linewidth=2.0,
            color=colors[schedule["schedule_id"]],
        )
        axes[1].plot(
            subset_bottom["time_rel_s"],
            subset_bottom["value_aligned"],
            linewidth=2.0,
            color=colors[schedule["schedule_id"]],
        )

    disturbance_color = _style_accent(style_cfg, "constraint_limit", "#8d0801")
    for ax in axes:
        ax.axvline(0.0, color=disturbance_color, linestyle="--", linewidth=1.3)
    axes[0].set_ylabel(rf"$\Delta P_{{\mathrm{{IBR}},{focus_ibr}}}(t)$ [p.u.]")
    axes[1].set_ylabel(r"$\Delta f_{\mathrm{COI}}(t)$ [Hz]")
    axes[1].set_xlabel("Time since disturbance [s]")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    axes[0].set_title("IBR response under the same disturbance (aligned at $t=0$)")
    axes[1].set_title("COI frequency response under the same disturbance (aligned at $t=0$)")
    save_figure(fig, "fig_1_2_simulation_intuition", figures_dir)


def plot_figure_3_3(
    cfg: dict[str, Any],
    *,
    figures_dir: Path,
    plot_data_dir: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    save_plot_data(build_network_plot_data(nodes, edges, figure_id="fig_3_3"), "fig_3_3_network_line_limits", plot_data_dir)

    fig, ax = plt.subplots(figsize=(11.4, 9.0), constrained_layout=True)
    fig_cfg = dict(cfg.get("figure_3_3", {}) or {})
    limit_unit = str(fig_cfg.get("line_limit_unit", "mva")).strip().lower()
    system_base_mva = float(cfg.get("system", {}).get("system_base_mva", 100.0))
    if limit_unit in {"pu", "p.u.", "per_unit"}:
        line_limit_values = edges["thermal_limit_mva"] / system_base_mva
        cbar_label = "Line limit [p.u.]"
    else:
        line_limit_values = edges["thermal_limit_mva"]
        cbar_label = "Line limit [MVA]"
    norm = mcolors.Normalize(vmin=float(line_limit_values.min()), vmax=float(line_limit_values.max()))
    cmap = cm.get_cmap("YlOrRd")
    for (_, row), line_limit in zip(edges.iterrows(), line_limit_values):
        color = cmap(norm(float(line_limit)))
        width = 1.4 + 3.4 * float(norm(float(line_limit)))
        # Draw a thin dark outline under each line for clearer separation.
        ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color="#111111", linewidth=width + 1.0, alpha=0.9, zorder=0)
        ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color=color, linewidth=width, alpha=0.95, zorder=1)

    category_style = {
        "PQ": dict(color="#ea6a4a", size=280, edge="#4a4a4a"),
        "SynGen": dict(color="#1b5e20", size=280, edge="#4a4a4a"),
        "IBR": dict(color="#8bcf7a", size=280, edge="#4a4a4a"),
        "Slack": dict(color="#111111", size=280, edge="#4a4a4a"),
        "Other bus": dict(color="#b9d6e8", size=280, edge="#4a4a4a"),
    }
    for category, style in category_style.items():
        subset = nodes.loc[nodes["node_category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            s=float(style["size"]),
            c=str(style["color"]),
            edgecolors=str(style["edge"]),
            linewidths=0.9,
            zorder=3 if category != "Slack" else 4,
            label=category,
        )

    for _, row in nodes.iterrows():
        cat = str(row.get("node_category", "Other bus"))
        bus_color = "white" if cat == "Slack" else "#1f2933"
        ax.text(
            row["x"],
            row["y"],
            str(int(row["bus"])),
            fontsize=8.0,
            ha="center",
            va="center",
            color=bus_color,
            fontweight="semibold",
            zorder=5,
        )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label(cbar_label)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=category_style[key]["color"], markeredgecolor="#4a4a4a", markersize=8.5, label=key)
        for key in ["PQ", "SynGen", "IBR", "Slack", "Other bus"]
    ]
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9.6, frameon=False, title="Bus type", title_fontsize=9.8)
    ax.set_axis_off()
    save_figure(fig, "fig_3_3_network_line_limits", figures_dir)


def plot_figure_3_4(
    cfg: dict[str, Any],
    *,
    style_cfg: dict[str, Any],
    figures_dir: Path,
    plot_data_dir: Path,
) -> None:
    fig_cfg = dict(cfg.get("figure_3_4", {}) or {})
    frames: list[pd.DataFrame] = []
    for dataset in list(fig_cfg.get("datasets") or []):
        csv_path = REPO_ROOT / str(dataset["csv"])
        frame = pd.read_csv(
            csv_path,
            usecols=["base_load_scale", "load_step_scale", "total_load_p_prefault", "DELTA_PQ_tot"],
        )
        frame["scenario_family"] = str(dataset["family"])
        frame["scenario_label"] = str(dataset["label"])
        frame["post_step_total_load_p"] = frame["total_load_p_prefault"] + frame["DELTA_PQ_tot"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)

    plot_rows: list[pd.DataFrame] = []
    for metric in ["base_load_scale", "load_step_scale", "post_step_total_load_p"]:
        block = data[["scenario_family", "scenario_label", metric]].copy()
        block = block.rename(columns={metric: "value"})
        block["metric"] = metric
        plot_rows.append(block)
    plot_df = pd.concat(plot_rows, ignore_index=True, sort=False)
    save_plot_data(plot_df, "fig_3_4_sampling_distributions", plot_data_dir)

    metric_labels = {
        "base_load_scale": r"Base-load scale $s_{\mathrm{base}}$ [-]",
        "load_step_scale": r"Load-step scale $s_{\mathrm{step}}$ [-]",
        "post_step_total_load_p": r"Post-disturbance total load $P_{L}^{+}$ [p.u.]",
    }
    sampling_palette = _style_palette(style_cfg, "sampling")
    colors = {
        "Global mismatch": sampling_palette.get("global", "#2f6690"),
        "Zone-based mismatch": sampling_palette.get("zone_based", "#c65d3b"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    metric_order = ["base_load_scale", "load_step_scale", "post_step_total_load_p"]
    label_order = ["Global mismatch", "Zone-based mismatch"]
    for idx, (ax, metric) in enumerate(zip(axes, metric_order)):
        for label in label_order:
            subset = data.loc[data["scenario_label"].astype(str) == label].copy()
            if subset.empty:
                continue
            ax.hist(
                subset[metric],
                bins=24,
                alpha=0.62,
                label=label,
                color=colors.get(label, "#4c6a92"),
                edgecolor="white",
                linewidth=0.40,
            )
        ax.set_xlabel(metric_labels[metric])
        ax.set_ylabel("Count [-]" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.18)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 2),
            bbox_to_anchor=(0.5, 1.05),
            fontsize=8.8,
            frameon=False,
        )
    save_figure(fig, "fig_3_4_sampling_distributions", figures_dir)


def plot_figure_3_5(
    *,
    style_cfg: dict[str, Any],
    figures_dir: Path,
    plot_data_dir: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> None:
    save_plot_data(build_network_plot_data(nodes, edges, figure_id="fig_3_5"), "fig_3_5_zones_proximity_map", plot_data_dir)

    zone_palette_raw = _style_palette(style_cfg, "zone_owner")
    zone_colors = {int(k): v for k, v in zone_palette_raw.items() if str(k).isdigit()}
    if not zone_colors:
        zone_colors = {
            1: "#4f7c5d",
            2: "#2f6690",
            3: "#c65d3b",
            4: "#805e73",
        }
    fig, ax = plt.subplots(figsize=(10.6, 9.0), constrained_layout=True)
    for _, row in edges.iterrows():
        ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color="#111111", linewidth=1.3, alpha=0.9, zorder=1)

    category_style = {
        "PQ": dict(color="#ea6a4a", size=280, edge="#4a4a4a"),
        "SynGen": dict(color="#1b5e20", size=280, edge="#4a4a4a"),
        "IBR": dict(color="#8bcf7a", size=280, edge="#4a4a4a"),
        "Slack": dict(color="#111111", size=280, edge="#4a4a4a"),
        "Other bus": dict(color="#b9d6e8", size=280, edge="#4a4a4a"),
    }
    for category, style in category_style.items():
        subset = nodes.loc[nodes["node_category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            s=float(style["size"]),
            c=str(style["color"]),
            edgecolors=str(style["edge"]),
            linewidths=0.9,
            zorder=3 if category != "Slack" else 4,
        )

    # Build transparent zone rectangles around zone load buses, including
    # nearest IBRs so zone-to-IBR association is visible.
    load_nodes = nodes.loc[nodes["is_load_bus"] & ~nodes["is_ibr_bus"]].copy()
    ibr_nodes = nodes.loc[nodes["is_ibr_bus"]].copy()
    zone_centroids: dict[int, tuple[float, float]] = {}
    for zone_owner, subset in load_nodes.groupby("zone_owner"):
        if pd.isna(zone_owner) or subset.empty:
            continue
        zone_centroids[int(zone_owner)] = (float(subset["x"].mean()), float(subset["y"].mean()))

    ibr_zone_rows: list[dict[str, Any]] = []
    for _, row in ibr_nodes.iterrows():
        if not zone_centroids:
            continue
        x = float(row["x"])
        y = float(row["y"])
        zone_assigned = min(
            zone_centroids.items(),
            key=lambda kv: (x - kv[1][0]) ** 2 + (y - kv[1][1]) ** 2,
        )[0]
        ibr_zone_rows.append({"zone_owner": int(zone_assigned), "x": x, "y": y})
    ibr_zone_df = pd.DataFrame(ibr_zone_rows)

    zone_boxes: list[dict[str, float | int]] = []
    for zone_owner, subset in load_nodes.groupby("zone_owner"):
        if pd.isna(zone_owner) or subset.empty:
            continue
        zone_id = int(zone_owner)
        zone_ibr = ibr_zone_df.loc[ibr_zone_df["zone_owner"] == zone_id] if not ibr_zone_df.empty else pd.DataFrame()
        x_vals = subset["x"] if zone_ibr.empty else pd.concat([subset["x"], zone_ibr["x"]], ignore_index=True)
        y_vals = subset["y"] if zone_ibr.empty else pd.concat([subset["y"], zone_ibr["y"]], ignore_index=True)
        x_min = float(x_vals.min()) - 0.045
        x_max = float(x_vals.max()) + 0.045
        y_min = float(y_vals.min()) - 0.045
        y_max = float(y_vals.max()) + 0.045
        zone_boxes.append({"zone": int(zone_owner), "x0": x_min, "x1": x_max, "y0": y_min, "y1": y_max})

    zone_boxes = sorted(zone_boxes, key=lambda b: int(b["zone"]))

    def _overlap(a: dict[str, float | int], b: dict[str, float | int], margin: float = 0.01) -> bool:
        return not (
            float(a["x1"]) + margin < float(b["x0"])
            or float(b["x1"]) + margin < float(a["x0"])
            or float(a["y1"]) + margin < float(b["y0"])
            or float(b["y1"]) + margin < float(a["y0"])
        )

    # Lightweight deterministic separation to reduce rectangle overlap.
    for _ in range(24):
        moved = False
        for i in range(len(zone_boxes)):
            for j in range(i + 1, len(zone_boxes)):
                bi = zone_boxes[i]
                bj = zone_boxes[j]
                if not _overlap(bi, bj):
                    continue
                shift = 0.03 + 0.005 * (j - i)
                bj["y0"] = float(bj["y0"]) - shift
                bj["y1"] = float(bj["y1"]) - shift
                bj["x0"] = float(bj["x0"]) + 0.5 * shift
                bj["x1"] = float(bj["x1"]) + 0.5 * shift
                moved = True
        if not moved:
            break

    for box in zone_boxes:
        zone = int(box["zone"])
        color = zone_colors.get(zone, "#999999")
        rect = Rectangle(
            (float(box["x0"]), float(box["y0"])),
            float(box["x1"]) - float(box["x0"]),
            float(box["y1"]) - float(box["y0"]),
            facecolor=color,
            edgecolor=color,
            linewidth=1.2,
            alpha=0.16,
            zorder=2,
        )
        ax.add_patch(rect)

    for _, row in nodes.iterrows():
        cat = str(row.get("node_category", "Other bus"))
        bus_color = "white" if cat == "Slack" else "#1f2933"
        ax.text(row["x"], row["y"], str(int(row["bus"])), fontsize=8.0, ha="center", va="center", color=bus_color, fontweight="semibold", zorder=5)
    ax.set_axis_off()

    category_labels = {
        "PQ": "Load bus (PQ)",
        "SynGen": "Synchronous generator",
        "IBR": "IBR",
        "Slack": "Slack bus",
        "Other bus": "Other bus",
    }
    category_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=category_style[key]["color"],
            markeredgecolor="#4a4a4a",
            markersize=8.5,
            label=category_labels[key],
        )
        for key in ["PQ", "SynGen", "IBR", "Slack", "Other bus"]
    ]
    zone_handles = [
        Patch(
            facecolor=zone_colors.get(z, "#999999"),
            edgecolor=zone_colors.get(z, "#999999"),
            alpha=0.22,
            label=f"Zone {z}",
        )
        for z in [1, 2, 3, 4]
    ]
    legend_bus = ax.legend(
        handles=category_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        title="Bus type",
        title_fontsize=8.6,
        fontsize=8.1,
        frameon=False,
        handlelength=1.2,
        labelspacing=0.45,
        borderpad=0.2,
    )
    ax.add_artist(legend_bus)
    ax.legend(
        handles=zone_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.64),
        title="Zone region",
        title_fontsize=8.6,
        fontsize=8.1,
        frameon=False,
        handlelength=1.2,
        labelspacing=0.45,
        borderpad=0.2,
    )
    save_figure(fig, "fig_3_5_zones_proximity_map", figures_dir)


def plot_figure_3_6(
    cfg: dict[str, Any],
    *,
    style_cfg: dict[str, Any],
    case_path: Path,
    figures_dir: Path,
    plot_data_dir: Path,
    tables_dir: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    device_bases: dict[int, float],
) -> None:
    fig_cfg = dict(cfg.get("figure_3_6", {}) or {})
    benchmark_cfg = load_yaml(REPO_ROOT / str(fig_cfg["benchmark_config"]))
    severity_df = resolve_outage_severity_table(benchmark_cfg)
    screened = select_screened_outages(severity_df, benchmark_cfg)
    save_table(screened, "fig_3_6_screened_outages", tables_dir)

    representative_mode = str(fig_cfg.get("representative_outage_mode", "highest_selected_severity"))
    explicit_line_uid = fig_cfg.get("representative_line_uid")
    if explicit_line_uid is not None:
        representative_line_uid = int(explicit_line_uid)
        matched = screened.loc[screened["line_uid"] == representative_line_uid]
        if matched.empty:
            representative = pd.Series(
                {
                    "line_uid": representative_line_uid,
                    "severity_bin": "not_screened",
                    "severity_score": np.nan,
                }
            )
            representative_mode = "explicit_line_uid"
        else:
            representative = matched.iloc[0]
            representative_mode = "explicit_line_uid"
    else:
        representative = screened.sort_values(["severity_score", "line_uid"], ascending=[False, True]).iloc[0]
        representative_line_uid = int(representative["line_uid"])
    rep_table = pd.DataFrame(
        [
            {
                "line_uid": representative_line_uid,
                "severity_bin": str(representative["severity_bin"]),
                "severity_score": float(representative["severity_score"]),
                "selection_mode": representative_mode,
            }
        ]
    )
    save_table(rep_table, "fig_3_6_representative_outage", tables_dir)

    edges_panel = edges.copy()
    edges_panel["screened"] = edges_panel["line_uid"].isin(screened["line_uid"])
    severity_map = screened.set_index("line_uid")[["severity_bin", "severity_score"]]
    edges_panel = edges_panel.join(severity_map, on="line_uid")
    not_considered_ids = {str(v).strip() for v in list(fig_cfg.get("n1_not_considered_line_ids") or []) if str(v).strip()}

    nominal_frequency_hz = float(cfg["system"]["nominal_frequency_hz"])
    system_base_mva = float(cfg["system"]["system_base_mva"])
    regcv1 = pd.read_excel(case_path, sheet_name="REGCV1")
    n_ibr = int(regcv1.shape[0])
    ss = configure_and_run_simulation(
        case_path=case_path,
        nominal_frequency_hz=nominal_frequency_hz,
        tds_cfg=dict(cfg.get("tds", {}) or {}),
        base_scale=float(fig_cfg.get("base_scale", 1.0)),
        scale_pv=False,
        m_values=np.full(n_ibr, float(fig_cfg.get("M", 3.0)), dtype=float),
        d_values=np.full(n_ibr, float(fig_cfg.get("D", 2.0)), dtype=float),
        line_uid=representative_line_uid,
        line_trip_time=float(fig_cfg.get("line_trip_time", 1.0)),
    )
    response_df = extract_response_frame(
        ss,
        schedule_id="representative_outage",
        schedule_label=f"Line {representative_line_uid}",
        nominal_frequency_hz=nominal_frequency_hz,
        system_base_mva=system_base_mva,
        device_bases=device_bases,
    )
    response_df["line_uid"] = representative_line_uid
    response_df["severity_bin"] = str(representative["severity_bin"])
    response_df["severity_score"] = float(representative["severity_score"])

    save_plot_data(build_network_plot_data(nodes, edges_panel, figure_id="fig_3_6a"), "fig_3_6a_screened_outages_map", plot_data_dir)
    save_plot_data(response_df, "fig_3_6b_representative_outage_response", plot_data_dir)

    fig = plt.figure(figsize=(12.6, 5.6), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0])

    ax_map = fig.add_subplot(outer[0, 0])
    # Highlight non-critical (not-screened) lines, keep screened lines subdued.
    for _, row in edges_panel.iterrows():
        line_uid = int(row["line_uid"])
        line_idx = str(row.get("line_idx", "")).strip()
        is_not_considered = line_idx in not_considered_ids
        if line_uid == representative_line_uid:
            color = "#c1121f"
            width = 2.8
            alpha = 1.0
            zorder = 1
        elif is_not_considered:
            color = "#b3b3b3"
            width = 1.8
            alpha = 1.0
            zorder = 1
        else:
            color = "#111111"
            width = 2.1
            alpha = 0.95
            zorder = 1
        ax_map.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color=color, linewidth=width, alpha=alpha, zorder=zorder)
    category_style = {
        "PQ": dict(color="#ea6a4a", size=230, edge="#4a4a4a"),
        "SynGen": dict(color="#1b5e20", size=230, edge="#4a4a4a"),
        "IBR": dict(color="#8bcf7a", size=230, edge="#4a4a4a"),
        "Slack": dict(color="#111111", size=230, edge="#4a4a4a"),
        "Other bus": dict(color="#b9d6e8", size=230, edge="#4a4a4a"),
    }
    for category, style in category_style.items():
        subset = nodes.loc[nodes["node_category"] == category]
        if subset.empty:
            continue
        ax_map.scatter(
            subset["x"],
            subset["y"],
            s=float(style["size"]),
            c=str(style["color"]),
            edgecolors=str(style["edge"]),
            linewidths=0.8,
            zorder=3 if category != "Slack" else 5,
        )
    for _, row in nodes.iterrows():
        cat = str(row.get("node_category", "Other bus"))
        bus_color = "white" if cat == "Slack" else "#1f2933"
        ax_map.text(row["x"], row["y"], str(int(row["bus"])), fontsize=7.3, ha="center", va="center", color=bus_color, fontweight="semibold")
    line_handles = [
        Line2D([0], [0], color="#b3b3b3", lw=2.1, label="Not considered for N-1"),
        Line2D([0], [0], color="#111111", lw=2.1, label="Considered for N-1"),
        Line2D([0], [0], color="#c1121f", lw=3.0, label=f"Example tripped line {representative_line_uid}"),
    ]
    bus_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=category_style[key]["color"], markeredgecolor="#4a4a4a", markersize=7.8, label=key)
        for key in ["PQ", "SynGen", "IBR", "Slack", "Other bus"]
    ]
    section_lines = Line2D([], [], linestyle="none", label="Lines")
    section_buses = Line2D([], [], linestyle="none", label="Bus type")
    spacer = Line2D([], [], linestyle="none", label=" ")
    combined_legend = [section_lines] + line_handles + [spacer, section_buses] + bus_handles
    ax_map.legend(handles=combined_legend, loc="upper left", fontsize=7.7, frameon=True, handlelength=1.3, borderpad=0.6, labelspacing=0.4)
    ax_map.set_axis_off()

    inner = outer[0, 1].subgridspec(2, 1, hspace=0.08)
    ax_freq = fig.add_subplot(inner[0, 0])
    ax_rocof = fig.add_subplot(inner[1, 0], sharex=ax_freq)
    freq_df = response_df.loc[response_df["series_type"] == "frequency_dev_hz"]
    rocof_df = response_df.loc[response_df["series_type"] == "rocof_hz_s"]
    ax_freq.plot(freq_df["time_s"], freq_df["value"], color="#8d0801", linewidth=2.0)
    ax_freq.axvline(float(fig_cfg.get("line_trip_time", 1.0)), color="#595959", linestyle="--", linewidth=1.1)
    ax_freq.set_ylabel(r"$\Delta f_{\mathrm{COI}}(t)$ [Hz]")
    ax_rocof.plot(rocof_df["time_s"], rocof_df["value"], color="#355070", linewidth=2.0)
    ax_rocof.axvline(float(fig_cfg.get("line_trip_time", 1.0)), color="#595959", linestyle="--", linewidth=1.1)
    ax_rocof.set_ylabel(r"$\mathrm{RoCoF}_{\mathrm{COI}}(t)$ [Hz/s]")
    ax_rocof.set_xlabel("Time [s]")
    save_figure(fig, "fig_3_6_screened_outages_response", figures_dir)


def build_all_figures(config_path: Path) -> None:
    cfg = load_yaml(config_path)
    style_cfg = _load_style_config(cfg)
    output_root = REPO_ROOT / str(cfg.get("outputs", {}).get("root", "results/thesis_figure_results/outputs"))
    figures_dir = output_root / "figures"
    plot_data_dir = output_root / "plot_data"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_data_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    case_path = REPO_ROOT / str(cfg["system"]["case"])
    case_tables = load_case_tables(case_path)
    device_bases = load_regcv1_device_bases(case_tables)
    set_thesis_style(style_cfg)

    node_df, edge_df = network_frames(case_path, str(cfg["system"].get("network_layout", "kamada")))
    nodes, edges = base_network_metadata(case_tables, node_df, edge_df)
    save_table(nodes, "network_nodes", tables_dir)
    save_table(edges, "network_edges", tables_dir)

    plot_figure_1_2(
        cfg,
        style_cfg=style_cfg,
        case_path=case_path,
        figures_dir=figures_dir,
        plot_data_dir=plot_data_dir,
        tables_dir=tables_dir,
        device_bases=device_bases,
    )
    plot_figure_3_3(cfg, figures_dir=figures_dir, plot_data_dir=plot_data_dir, nodes=nodes, edges=edges)
    plot_figure_3_4(cfg, style_cfg=style_cfg, figures_dir=figures_dir, plot_data_dir=plot_data_dir)
    plot_figure_3_5(style_cfg=style_cfg, figures_dir=figures_dir, plot_data_dir=plot_data_dir, nodes=nodes, edges=edges)
    plot_figure_3_6(
        cfg,
        style_cfg=style_cfg,
        case_path=case_path,
        figures_dir=figures_dir,
        plot_data_dir=plot_data_dir,
        tables_dir=tables_dir,
        nodes=nodes,
        edges=edges,
        device_bases=device_bases,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build thesis context figures and plot-data exports.")
    parser.add_argument(
        "--config",
        default="configs/figure/context_figures.yaml",
        help="Path to the context-figure config YAML.",
    )
    args = parser.parse_args(argv)
    build_all_figures((REPO_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config))


if __name__ == "__main__":
    main()
