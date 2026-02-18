import argparse
import csv
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import yaml
import andes
import cvxpy as cp

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_metrics import export_plotter_all, extract_simulation_row


def _resolve_case_path(case: str) -> str:
    if os.path.exists(case):
        return case
    case_path = andes.get_case(case)
    if os.path.exists(case_path):
        return case_path
    raise FileNotFoundError(f"Case not found: {case}")


def _as_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _sanitize_label(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return safe or "owner"


def _sample_from_bins(bins: Sequence[Dict], rng: np.random.Generator) -> Tuple[float, str]:
    probs = np.asarray([b.get("prob", 1.0) for b in bins], dtype=float)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(bins), p=probs))
    bin_cfg = bins[idx]
    low, high = float(bin_cfg["low"]), float(bin_cfg["high"])
    val = float(rng.uniform(low, high))
    label = str(bin_cfg.get("label", f"bin{idx}"))
    return val, label


def _sample_scalar(range_cfg: Dict, rng: np.random.Generator, *, default_low: float, default_high: float, log_uniform: bool = False) -> float:
    low = float(range_cfg.get("low", default_low))
    high = float(range_cfg.get("high", default_high))
    if log_uniform:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return float(rng.uniform(low, high))


def _sample_value(value_cfg: Dict, rng: np.random.Generator, *, default_low: float, default_high: float) -> Tuple[float, str]:
    if isinstance(value_cfg, (list, tuple)) and len(value_cfg) >= 2:
        value_cfg = {"low": value_cfg[0], "high": value_cfg[1]}
    if isinstance(value_cfg, (int, float)):
        value_cfg = {"low": value_cfg, "high": value_cfg}
    if "bins" in value_cfg and value_cfg["bins"]:
        return _sample_from_bins(value_cfg["bins"], rng)
    log_u = bool(value_cfg.get("log_uniform", False))
    val = _sample_scalar(value_cfg, rng, default_low=default_low, default_high=default_high, log_uniform=log_u)
    return val, "uniform_log" if log_u else "uniform"


def _select_step_targets(ss, load_cfg: Dict) -> List[str]:
    """Return PQ device names to apply the step scale to.

    Priority rules:
    - if load_cfg["pq_names"] is provided and non-empty, start from that list;
      otherwise default to all PQ names in the case.
    - if load_cfg["owners"] is provided, keep only PQs whose owner is in that
      list (ownership is read from ss.PQ.owner).
    """

    pq_names_cfg = _as_list(load_cfg.get("pq_names"))
    if not pq_names_cfg:
        pq_names_cfg = list(ss.PQ.name.v) if getattr(ss, "PQ", None) and ss.PQ.n else []

    owner_filter = {str(o) for o in _as_list(load_cfg.get("owners"))}
    if owner_filter and getattr(ss.PQ, "n", 0):
        name_to_owner = {str(name): str(owner) for name, owner in zip(ss.PQ.name.v, ss.PQ.owner.v)}
        pq_names_cfg = [name for name in pq_names_cfg if name_to_owner.get(str(name)) in owner_filter]

    return pq_names_cfg


def _build_fieldnames(
    pq_names: Sequence[str],
    owner_labels: Sequence[str],
    n_ibr: int,
    n_genrou: int,
    include_plotter: bool,
) -> List[str]:
    fieldnames = [
        "sim_id",
        "seed",
        "success",
        "cont_type",
        "contingency_id",
        "contingency_time",
        "load_step_enabled",
        "line_uid",
        "line_idx",
        "line_name",
        "line_from_bus",
        "line_to_bus",
        "line_rating",
        "pre_fault_flow",
        "pre_fault_loading",
        "base_load_scale",
        "load_step_scale",
        "load_step_time",
        "base_load_p_total",
        "base_load_q_total",
        "DELTA_PQ_tot",
        "M_agg",
        "D_agg",
        "time_max_dev",
        "rocof_COI",
        "dev_COI",
    ]
    for i in range(1, n_ibr + 1):
        fieldnames.append(f"M_{i}")
        fieldnames.append(f"D_{i}")
    for name in pq_names:
        fieldnames.append(f"DELTA_P_{name}")
    for owner in owner_labels:
        fieldnames.append(f"DELTA_P_OWNER_{owner}")
    for i in range(1, n_genrou + 1):
        fieldnames.append(f"P_GENROU_{i}")
    for i in range(1, n_ibr + 1):
        fieldnames.append(f"P_REGCV1_{i}")
    for i in range(1, n_ibr + 1):
        fieldnames.append(f"Delta_P_IBR_{i}")
    fieldnames.extend(
        [
            "step_bin_label",
            "M_bin_label",
            "D_bin_label",
            "ed_cost_genrou_a",
            "ed_cost_genrou_b",
            "ed_cost_genrou_c",
            "ed_cost_ibr_a",
            "ed_cost_ibr_b",
            "ed_cost_ibr_c",
        ]
    )
    if include_plotter:
        fieldnames.append("plotter_csv")
    return fieldnames


def _sample_ed_costs(ed_cfg: Dict, rng: np.random.Generator) -> Dict[str, Tuple[float, float, float]]:
    return {
        "gen": _sample_cost_triple(ed_cfg.get("genrou_costs", {}), rng, default=(1.0, 0.1, 0.01)),
        "ibr": _sample_cost_triple(ed_cfg.get("regcv1_costs", {}), rng, default=(1.0, 0.1, 0.01)),
    }


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rng_for_sim(seed: int, sim_id: int) -> np.random.Generator:
    sim_seed = np.random.SeedSequence([seed, sim_id])
    return np.random.default_rng(sim_seed)


def _chunk_sim_ids(n_sims: int, workers: int) -> List[List[int]]:
    if workers <= 1:
        return [list(range(n_sims))]
    chunk_size = (n_sims + workers - 1) // workers
    return [list(range(i, min(i + chunk_size, n_sims))) for i in range(0, n_sims, chunk_size)]


def _worker_output_path(output_dir: Path, output_csv: str, worker_id: int) -> Path:
    base = Path(output_csv)
    suffix = base.suffix if base.suffix else ".csv"
    return output_dir / f"{base.stem}_worker_{worker_id:02d}{suffix}"


def _sample_cost_triple(cfg: Dict, rng: np.random.Generator, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a = _sample_scalar(cfg.get("a", {}), rng, default_low=default[0], default_high=default[0])
    b = _sample_scalar(cfg.get("b", {}), rng, default_low=default[1], default_high=default[1])
    c = _sample_scalar(cfg.get("c", {}), rng, default_low=default[2], default_high=default[2])
    return a, b, c


def _run_ed_dispatch(
    ss,
    ed_cfg: Dict,
    ibr_idx: Sequence[int],
    *,
    sampled_costs: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve economic dispatch with two shared cost triples (GENROU-like vs REGCV1-like)."""
    ng = ss.PV.n + ss.Slack.n
    if ng == 0:
        return np.zeros(0), {}

    Pd = float(np.sum(ss.PQ.p0.v))
    Pg_min = np.asarray(ss.PV.pmin.v + ss.Slack.pmin.v, dtype=float)
    Pg_max = np.asarray(ss.PV.pmax.v + ss.Slack.pmax.v, dtype=float)

    if sampled_costs is None:
        gen_cost = (1.0, 0.1, 0.01)
        ibr_cost = (1.0, 0.1, 0.01)
    else:
        gen_cost = sampled_costs["gen"]
        ibr_cost = sampled_costs["ibr"]

    a = np.full(ng, gen_cost[0], dtype=float)
    b = np.full(ng, gen_cost[1], dtype=float)
    c = np.full(ng, gen_cost[2], dtype=float)
    for idx in ibr_idx:
        if 0 <= idx < ng:
            a[idx] = ibr_cost[0]
            b[idx] = ibr_cost[1]
            c[idx] = ibr_cost[2]

    Pg = cp.Variable(ng)
    constraints = [cp.sum(Pg) == Pd, Pg >= Pg_min, Pg <= Pg_max]
    objective = cp.Minimize(cp.sum(a + cp.multiply(b, Pg) + cp.multiply(c, cp.square(Pg))))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=ed_cfg.get("solver", "OSQP"), verbose=bool(ed_cfg.get("verbose", False)))
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"ED solve failed with status={prob.status}")
    Pg_val = np.asarray(Pg.value, dtype=float).reshape(-1)

    ss.PV.p0.v = Pg_val[: ss.PV.n]
    ss.Slack.p0.v = Pg_val[ss.PV.n :]
    if hasattr(ss, "REGCV1") and ss.REGCV1.n:
        for local_idx, gen_idx in enumerate(ibr_idx):
            if 0 <= gen_idx < Pg_val.size and local_idx < ss.REGCV1.n:
                try:
                    ss.REGCV1.pref.v[local_idx] = Pg_val[gen_idx]
                except Exception:
                    pass

    return Pg_val, {
        "ed_cost_genrou_a": float(gen_cost[0]),
        "ed_cost_genrou_b": float(gen_cost[1]),
        "ed_cost_genrou_c": float(gen_cost[2]),
        "ed_cost_ibr_a": float(ibr_cost[0]),
        "ed_cost_ibr_b": float(ibr_cost[1]),
        "ed_cost_ibr_c": float(ibr_cost[2]),
    }


def _to_float_or_nan(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _line_rating_from_ss(ss, uid: int) -> float:
    for attr in ("rate_a", "rateA", "RATE_A"):
        obj = getattr(ss.Line, attr, None)
        if obj is None:
            continue
        vals = getattr(obj, "v", None)
        if vals is None or uid >= len(vals):
            continue
        return _to_float_or_nan(vals[uid])
    return float("nan")


def _extract_line_records(ss) -> List[Dict[str, float | int | str]]:
    records: List[Dict[str, float | int | str]] = []
    n_line = int(getattr(ss.Line, "n", 0))
    idx_vals = list(getattr(getattr(ss.Line, "idx", None), "v", []))
    name_vals = list(getattr(getattr(ss.Line, "name", None), "v", []))
    bus1_vals = list(getattr(getattr(ss.Line, "bus1", None), "v", []))
    bus2_vals = list(getattr(getattr(ss.Line, "bus2", None), "v", []))
    u_vals = list(getattr(getattr(ss.Line, "u", None), "v", []))

    for uid in range(n_line):
        idx_val = idx_vals[uid] if uid < len(idx_vals) else uid + 1
        name_val = name_vals[uid] if uid < len(name_vals) else f"Line_{idx_val}"
        bus1 = bus1_vals[uid] if uid < len(bus1_vals) else np.nan
        bus2 = bus2_vals[uid] if uid < len(bus2_vals) else np.nan
        in_service = bool(u_vals[uid]) if uid < len(u_vals) else True
        records.append(
            {
                "uid": uid,
                "idx": idx_val,
                "name": str(name_val),
                "bus1": _to_float_or_nan(bus1),
                "bus2": _to_float_or_nan(bus2),
                "rating": _line_rating_from_ss(ss, uid),
                "in_service": in_service,
            }
        )
    return records


def _pick_line_contingencies(ss, cont_cfg: Dict, rng: np.random.Generator) -> List[Dict[str, float | int | str]]:
    line_records = _extract_line_records(ss)
    if not line_records:
        return []

    active = [r for r in line_records if bool(r.get("in_service", True))]
    line_ids_cfg = _as_list(cont_cfg.get("line_ids"))
    if line_ids_cfg:
        wanted = {str(v) for v in line_ids_cfg}
        selected = [
            r
            for r in active
            if (str(r["idx"]) in wanted) or (str(r["name"]) in wanted) or (str(r["uid"]) in wanted)
        ]
    else:
        selected = active

    max_lines = int(cont_cfg.get("max_lines", 0) or 0)
    if max_lines > 0 and len(selected) > max_lines:
        pick = rng.choice(len(selected), size=max_lines, replace=False)
        selected = [selected[int(i)] for i in np.sort(pick)]

    return selected


def _line_flow_from_ss(ss, uid: int) -> float:
    # Best effort across different ANDES line variable names.
    candidates = [("a1", "e"), ("Pij", "v"), ("p1", "v"), ("P1", "v")]
    for attr, sub in candidates:
        obj = getattr(ss.Line, attr, None)
        if obj is None:
            continue
        vals = getattr(obj, sub, None)
        if vals is None or uid >= len(vals):
            continue
        return _to_float_or_nan(vals[uid])
    return float("nan")


def _line_ratings_from_pandapower(ss) -> Optional[np.ndarray]:
    """
    Best-effort fallback to extract RATE_A per branch from pandapower.
    Returns None if conversion/extraction fails.
    """
    try:
        from andes.interop import pandapower as ap
        from pandapower.pd2ppc import _pd2ppc
        from pandapower import auxiliary as aux
    except Exception:
        return None

    try:
        pp_net = ap.to_pandapower(ss, verify=False)
        if not hasattr(pp_net, "_options") or not isinstance(pp_net._options, dict):
            pp_net._options = {}
        if "mode" not in pp_net._options:
            aux._add_ppc_options(
                pp_net,
                calculate_voltage_angles=True,
                trafo_model="pi",
                check_connectivity=False,
                mode="opf",
                switch_rx_ratio=2,
                enforce_q_lims=False,
                recycle=None,
            )
        _, ppci = _pd2ppc(pp_net)
        branch = ppci["branch"]
        return np.asarray(branch[:, 5], dtype=float)  # RATE_A
    except Exception:
        return None


def _run_worker(
    worker_id: int,
    sim_ids: Sequence[int],
    cfg: Dict,
    case_path: str,
    pq_names: Sequence[str],
    owner_map: Dict[str, str],
    regcv1_ids: Sequence[str],
    plotter_dir: Optional[Path],
    fieldnames: Sequence[str],
    output_dir: Path,
) -> Optional[str]:
    if not sim_ids:
        return None
    if plotter_dir is not None:
        plotter_dir.mkdir(parents=True, exist_ok=True)
    worker_csv = _worker_output_path(output_dir, cfg.get("output_csv", "simulation_results.csv"), worker_id)
    with open(worker_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sim_id in sim_ids:
            rng = _rng_for_sim(int(cfg["seed"]), sim_id)
            rows = run_single_sim(cfg, sim_id, rng, case_path, pq_names, owner_map, regcv1_ids, plotter_dir)
            for row in rows:
                for name in fieldnames:
                    row.setdefault(name, np.nan)
                writer.writerow(row)
    return str(worker_csv)


def _merge_worker_csvs(csv_path: Path, worker_paths: Iterable[Path], fieldnames: Sequence[str]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for worker_path in worker_paths:
            with open(worker_path, "r", newline="", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    writer.writerow(row)


def run_single_sim(
    cfg: Dict,
    sim_id: int,
    rng: np.random.Generator,
    case_path: str,
    pq_names: Sequence[str],
    owner_map: Dict[str, str],
    regcv1_ids: Sequence[str],
    plotter_dir: Path,
):
    base_scale = rng.uniform(cfg["base_load_scale"]["low"], cfg["base_load_scale"]["high"])

    cont_cfg = cfg.get("contingency", {})
    load_step_cfg = dict(cont_cfg.get("load_step", {}) or {})
    line_n1_cfg = dict(cont_cfg.get("line_n1", {}) or {})

    # Backward compatibility with older contingency.mode schema.
    if ("mode" in cont_cfg) and (not load_step_cfg) and (not line_n1_cfg):
        cont_mode = str(cont_cfg.get("mode", "none")).lower()
        valid_modes = {"none", "load_step", "line_n1"}
        if cont_mode not in valid_modes:
            raise ValueError(
                f"Unsupported contingency.mode={cont_mode!r}. Expected one of {sorted(valid_modes)}."
            )
        load_step_cfg["enable"] = bool(cont_mode == "load_step" or cont_cfg.get("include_load_step", False))
        line_n1_cfg["enable"] = bool(cont_mode == "line_n1")
        line_n1_cfg["trip_time"] = cont_cfg.get("trip_time")
        line_n1_cfg["line_ids"] = cont_cfg.get("line_ids")
        line_n1_cfg["max_lines"] = cont_cfg.get("max_lines")

    load_step_enabled = bool(load_step_cfg.get("enable", False))
    line_n1_enabled = bool(line_n1_cfg.get("enable", False))

    load_step_time = float(load_step_cfg.get("time", cfg.get("tds", {}).get("load_step_time", 0.1)))
    load_step_scale_cfg = load_step_cfg.get("scale", cfg.get("load_step_scale", {}))
    default_low = float(load_step_scale_cfg.get("low", 1.0))
    default_high = float(load_step_scale_cfg.get("high", default_low))

    if load_step_enabled:
        step_scale, step_bin = _sample_value(
            load_step_scale_cfg,
            rng,
            default_low=default_low,
            default_high=default_high,
        )
    else:
        step_scale, step_bin = 1.0, "disabled"

    M_samples = [
        _sample_value(
            cfg.get("ibr", {}).get("M_range", {}),
            rng,
            default_low=cfg["ibr"]["M_range"][0],
            default_high=cfg["ibr"]["M_range"][1],
        )
        for _ in range(len(regcv1_ids))
    ]
    D_samples = [
        _sample_value(
            cfg.get("ibr", {}).get("D_range", {}),
            rng,
            default_low=cfg["ibr"]["D_range"][0],
            default_high=cfg["ibr"]["D_range"][1],
        )
        for _ in range(len(regcv1_ids))
    ]
    M_vec = np.asarray([v for v, _ in M_samples], dtype=float)
    D_vec = np.asarray([v for v, _ in D_samples], dtype=float)
    M_bin_label = M_samples[0][1] if M_samples else "none"
    D_bin_label = D_samples[0][1] if D_samples else "none"

    ed_cfg = cfg.get("ed", {})
    sampled_ed_costs = _sample_ed_costs(ed_cfg, rng) if ed_cfg.get("enable", False) else None

    # Use a dry system load to determine which line contingencies to run.
    if line_n1_enabled:
        ss_pick = andes.load(case_path, setup=False)
        line_records = _pick_line_contingencies(ss_pick, line_n1_cfg, rng)
        if not line_records:
            raise ValueError(
                "contingency.line_n1.enable=true, but no valid in-service lines were found "
                "for contingency.line_n1.line_ids/max_lines."
            )
        trip_time = float(line_n1_cfg.get("trip_time", load_step_time))
        contingencies = line_records
    else:
        trip_time = float("nan")
        contingencies = [None]

    rows: List[Dict] = []
    for cont in contingencies:
        ss = andes.load(case_path, setup=False)
        ss.config.freq = float(50)

        # Apply base load scale
        for uid in range(ss.PQ.n):
            ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
            ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
        for uid in range(ss.PV.n):
            ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
            ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale

        ss.REGCV1.M.v, ss.REGCV1.D.v = M_vec, D_vec

        # Optional economic dispatch before disturbance (same sampled costs for all contingencies in this sim_id)
        ed_meta = {}
        pg_dispatch = np.array([], dtype=float)
        if ed_cfg.get("enable", False):
            ibr_idx = _as_list(ed_cfg.get("ibr_idx")) or []
            pg_dispatch, ed_meta = _run_ed_dispatch(
                ss,
                ed_cfg,
                ibr_idx=ibr_idx,
                sampled_costs=sampled_ed_costs,
            )

        ss.PQ.config.p2p = 1
        ss.PQ.config.q2q = 1
        ss.PQ.config.p2z = 0
        ss.PQ.config.q2z = 0
        ss.PQ.config.p2i = 0
        ss.PQ.config.q2i = 0
        ss.PQ.config.pq2z = 0

        pq_p_before, pq_q_before = ss.PQ.p0.v, ss.PQ.q0.v
        base_load_q_total = float(np.sum(pq_q_before)) if len(pq_q_before) else 0.0
        pq_owner_list = [owner_map.get(str(o), _sanitize_label(o)) for o in ss.PQ.owner.v]

        if load_step_enabled:
            step_targets = _select_step_targets(ss, cfg.get("load", {}))
            for dev in step_targets:
                ss.add(model="Alter", param_dict=dict(t=load_step_time, model="PQ", dev=dev, src="Ppf", attr="v", method="*", amount=step_scale))
                ss.add(model="Alter", param_dict=dict(t=load_step_time, model="PQ", dev=dev, src="Qpf", attr="v", method="*", amount=step_scale))

        if cont is not None:
            # N-1 line outage via Toggle at contingency time.
            ss.add(
                model="Toggle",
                param_dict={
                    "t": trip_time,
                    "model": "Line",
                    "dev": cont["idx"],
                },
            )

        ss.setup()
        ss.PFlow.run()
        pp_fmax = _line_ratings_from_pandapower(ss)

        pre_fault_flow = float("nan")
        pre_fault_loading = float("nan")
        line_rating = float("nan")
        if cont is not None:
            uid = int(cont["uid"])
            pre_fault_flow = _line_flow_from_ss(ss, uid)
            rating = _to_float_or_nan(cont.get("rating"))
            if (not np.isfinite(rating) or rating <= 0.0) and pp_fmax is not None and uid < len(pp_fmax):
                rating = _to_float_or_nan(pp_fmax[uid])
            line_rating = rating
            if np.isfinite(pre_fault_flow) and np.isfinite(line_rating) and line_rating > 0:
                pre_fault_loading = abs(pre_fault_flow) / line_rating

        ss.TDS.config.no_tqdm = bool(cfg["tds"].get("no_tqdm", True))
        ss.TDS.config.criteria = int(cfg["tds"].get("criteria", 0))
        ss.TDS.config.tol = float(cfg["tds"].get("tol", 1e-3))
        ss.TDS.config.tf = float(cfg["tds"]["t_end"])
        ss.TDS.config.tstep = float(cfg["tds"]["t_step"])
        ss.TDS.config.fixt = int(cfg["tds"].get("fixt", 0))
        ss.TDS.config.method = str(cfg["tds"].get("method", "backeuler"))
        ss.TDS.config.honest = int(cfg["tds"].get("honest", 1))
        ss.TDS.config.max_iter = int(cfg["tds"].get("max_iter", 35))
        ss.TDS.config.shrinkt = int(cfg["tds"].get("shrinkt", 1))

        ss.TDS.init()
        success = bool(ss.TDS.run())
        ss.TDS.load_plotter()

        pq_p_after, pq_q_after = ss.PQ.Ppf.v, ss.PQ.Qpf.v

        genrou_pg = np.asarray(getattr(ss.GENROU, "Pg", np.zeros(0)), dtype=float)
        if genrou_pg.size == 0 and hasattr(ss.GENROU, "p0"):
            genrou_pg = np.asarray(ss.GENROU.p0.v, dtype=float)
        if genrou_pg.size == 0 and pg_dispatch.size:
            genrou_pg = pg_dispatch[: ss.GENROU.n] if ss.GENROU.n else np.zeros(0)
        regcv1_pg = np.zeros(ss.REGCV1.n, dtype=float)
        if hasattr(ss.REGCV1, "pref"):
            try:
                regcv1_pg = np.asarray(ss.REGCV1.pref.v, dtype=float)
            except Exception:
                regcv1_pg = np.zeros(ss.REGCV1.n, dtype=float)
        if pg_dispatch.size:
            ibr_idx = _as_list(ed_cfg.get("ibr_idx")) or list(range(ss.REGCV1.n))
            for local_idx, gen_idx in enumerate(ibr_idx):
                if 0 <= gen_idx < pg_dispatch.size and local_idx < regcv1_pg.size:
                    regcv1_pg[local_idx] = pg_dispatch[gen_idx]

        plotter_csv = None
        if plotter_dir is not None:
            if cont is None:
                name_suffix = f"none_{sim_id:05d}"
            else:
                name_suffix = f"line_{sim_id:05d}_{int(cont['uid']):03d}"
            plotter_csv = str(plotter_dir / f"plotter_{name_suffix}.csv")
            export_plotter_all(ss.TDS.plotter, plotter_csv)

        row = extract_simulation_row(
            ss=ss,
            base_load_scale=base_scale,
            load_step_scale=step_scale,
            load_step_time=load_step_time,
            pq_names=pq_names,
            pq_owners=pq_owner_list,
            pq_p_before=pq_p_before,
            pq_p_after=pq_p_after,
            base_load_q_total=base_load_q_total,
            M_vec=M_vec,
            D_vec=D_vec,
            genrou_pg=genrou_pg,
            regcv1_pg=regcv1_pg,
            success=success,
            plotter_csv=plotter_csv,
        )

        row["sim_id"] = sim_id
        row["seed"] = int(cfg["seed"])
        row["step_bin_label"] = step_bin
        row["M_bin_label"] = M_bin_label
        row["D_bin_label"] = D_bin_label
        row.update(ed_meta)

        if cont is None:
            if load_step_enabled:
                row["cont_type"] = "load"
                row["contingency_id"] = "load_step"
                row["contingency_time"] = load_step_time
            else:
                row["cont_type"] = "none"
                row["contingency_id"] = "none"
                row["contingency_time"] = np.nan
            row["load_step_enabled"] = int(bool(load_step_enabled))
            row["line_uid"] = np.nan
            row["line_idx"] = np.nan
            row["line_name"] = ""
            row["line_from_bus"] = np.nan
            row["line_to_bus"] = np.nan
            row["line_rating"] = np.nan
            row["pre_fault_flow"] = np.nan
            row["pre_fault_loading"] = np.nan
        else:
            row["cont_type"] = "line_plus_load" if load_step_enabled else "line"
            row["contingency_id"] = f"line:{cont['idx']}"
            row["contingency_time"] = trip_time
            row["load_step_enabled"] = int(bool(load_step_enabled))
            row["line_uid"] = int(cont["uid"])
            row["line_idx"] = cont["idx"]
            row["line_name"] = str(cont["name"])
            row["line_from_bus"] = _to_float_or_nan(cont["bus1"])
            row["line_to_bus"] = _to_float_or_nan(cont["bus2"])
            row["line_rating"] = line_rating
            row["pre_fault_flow"] = pre_fault_flow
            row["pre_fault_loading"] = pre_fault_loading

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ANDES sims and extract metrics.")
    parser.add_argument("--config", default="data_generation/generation.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    print("Loading config from ", args.config)
    cfg = load_config(args.config)
    andes.config_logger(stream_level=int(cfg.get("stream_level", 30)))

    case_path = _resolve_case_path(cfg["case"])

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg.get("output_csv", "simulation_results.csv")

    plotter_cfg = cfg.get("plotter", {})
    export_plotter = bool(plotter_cfg.get("export", True))
    plotter_dir = None
    if export_plotter:
        plotter_dir = output_dir / plotter_cfg.get("subdir", "plotter")
        plotter_dir.mkdir(parents=True, exist_ok=True)

    base_ss = andes.load(case_path, setup=False)
    pq_names = list(base_ss.PQ.name.v) if base_ss.PQ.n else []
    if cfg["load"].get("pq_names"):
        pq_names = [name for name in pq_names if name in cfg["load"]["pq_names"]]

    owner_map = {str(o): _sanitize_label(o) for o in set(base_ss.PQ.owner.v)} if getattr(base_ss, "PQ", None) else {}
    owner_labels = sorted(owner_map.values())

    regcv1_ids = cfg["ibr"].get("indices") or []
    if not regcv1_ids:
        n_ibr = int(cfg["ibr"].get("n_ibr", 4))
        regcv1_ids = list(base_ss.REGCV1.idx.v)[:n_ibr]
    if not regcv1_ids:
        raise ValueError("No REGCV1 entries found to assign M/D parameters.")

    n_genrou = int(getattr(base_ss, "GENROU", None).n) if getattr(base_ss, "GENROU", None) else 0
    fieldnames = _build_fieldnames(pq_names, owner_labels, len(regcv1_ids), n_genrou, export_plotter)

    n_sims = int(cfg["n_sims"])
    workers = max(1, int(cfg.get("workers", 1)))
    if workers <= 1 or n_sims <= 1:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sim_id in range(n_sims):
                sim_rng = _rng_for_sim(int(cfg["seed"]), sim_id)
                rows = run_single_sim(cfg, sim_id, sim_rng, case_path, pq_names, owner_map, regcv1_ids, plotter_dir)
                for row in rows:
                    for name in fieldnames:
                        row.setdefault(name, np.nan)
                    writer.writerow(row)
        return

    sim_chunks = _chunk_sim_ids(n_sims, workers)
    worker_args = []
    for worker_id, sim_ids in enumerate(sim_chunks):
        if sim_ids:
            worker_args.append(
                (
                    worker_id,
                    sim_ids,
                    cfg,
                    case_path,
                    pq_names,
                    owner_map,
                    regcv1_ids,
                    plotter_dir,
                    fieldnames,
                    output_dir,
                )
            )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(worker_args))) as pool:
        worker_paths = pool.starmap(_run_worker, worker_args)

    merge_paths = [Path(path) for path in worker_paths if path]
    _merge_worker_csvs(csv_path, merge_paths, fieldnames)


if __name__ == "__main__":
    main()
