import argparse
import copy
import csv
import io
import multiprocessing as mp
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import yaml
import andes

from data_generation.debug_tools import save_coi_trace_csv, save_debug_coi_plot
from data_generation.disturbance_dispatch import DisturbanceDispatcher, resolve_disturbance_kind, sample_value
from data_generation.extract_metrics import (
    extract_initial_state_metrics,
    extract_line_metrics,
    extract_operating_point_snapshot,
    extract_simulation_row,
    initial_state_fieldnames_from_plotter,
    simulation_row_fieldnames,
)


DEFAULT_DISPATCH_COST_PATH = "configs/shared/ieee39_regcv1_dispatch_costs.yaml"
DEFAULT_FEATURE_NAMES_PATH = "configs/shared/data_generation_feature_names.yaml"


def _run_with_suppressed_pandapower_noise(fn, *args, **kwargs):
    """Internal helper to run with suppressed pandapower noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"andes\.interop\.pandapower",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"pandas\..*",
        )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return fn(*args, **kwargs)


def _build_fieldnames(
    pq_names: Sequence[str],
    owner_labels: Sequence[str],
    n_ibr: int,
    n_genrou: int,
    line_uids: Sequence[int],
    bus_numbers: Sequence[int],
    initial_state_fields: Sequence[str],
    feature_names_path: Optional[str],
) -> List[str]:
    """Internal helper to build fieldnames."""
    return simulation_row_fieldnames(
        pq_names=pq_names,
        owner_labels=owner_labels,
        n_ibr=n_ibr,
        n_genrou=n_genrou,
        line_uids=line_uids,
        bus_numbers=bus_numbers,
        initial_state_fields=initial_state_fields,
        feature_names_path=feature_names_path,
    )


def load_config(path: str) -> Dict:
    """Load config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_names_path(cfg: Dict) -> str:
    """Internal helper to feature names path."""
    return str(cfg.get("feature_names_path") or DEFAULT_FEATURE_NAMES_PATH)


def _include_initial_state(cfg: Dict) -> bool:
    """Internal helper to resolve initial-state feature inclusion."""
    features_cfg = cfg.get("features", {}) or {}
    return bool(features_cfg.get("include_initial_state", True))


def _assert_line_metrics_dc_ready(cfg: Dict) -> None:
    """Internal helper to assert line metrics dc ready."""
    cont_cfg = cfg.get("contingency", {}) or {}
    line_n1_cfg = cont_cfg.get("line_n1", {}) or {}
    if not bool(line_n1_cfg.get("enable", False)):
        return
    try:
        import pandapower  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "contingency.line_n1.enable=true requires pandapower in the active Python environment. "
            "Run with the project venv (../venv/bin/python ...) or install pandapower."
        ) from e


def _rng_for_sim(seed: int, sim_id: int) -> np.random.Generator:
    """Internal helper to rng for sim."""
    sim_seed = np.random.SeedSequence([seed, sim_id])
    return np.random.default_rng(sim_seed)


def _chunk_sim_ids(n_sims: int, workers: int) -> List[List[int]]:
    """Internal helper to chunk sim ids."""
    if workers <= 1:
        return [list(range(n_sims))]
    chunk_size = (n_sims + workers - 1) // workers
    return [list(range(i, min(i + chunk_size, n_sims))) for i in range(0, n_sims, chunk_size)]


def _worker_output_path(output_dir: Path, output_csv: str, worker_id: int) -> Path:
    """Internal helper to worker output path."""
    base = Path(output_csv)
    suffix = base.suffix if base.suffix else ".csv"
    return output_dir / f"{base.stem}_worker_{worker_id:02d}{suffix}"


def _dispatch_ibr_indices(ss, fallback: Optional[Sequence[int]] = None) -> List[int]:
    """Return 0-based dispatch-vector positions for controllable IBRs."""
    if not getattr(getattr(ss, "REGCV1", None), "n", 0):
        return []
    regcv1_gen = getattr(getattr(ss, "REGCV1", None), "gen", None)
    regcv1_gen_vals = getattr(regcv1_gen, "v", None)
    if regcv1_gen_vals is not None and len(regcv1_gen_vals) > 0:
        out: List[int] = []
        for val in regcv1_gen_vals:
            try:
                idx = int(val) - 1
            except Exception:
                continue
            if idx >= 0:
                out.append(idx)
        if out:
            return out
    return list(fallback or [])


def _resolve_repo_path(path_str: Optional[str], default_rel: str) -> Path:
    """Internal helper to resolve repo path."""
    path = Path(path_str) if path_str else Path(default_rel)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def _midpoint_from_cfg(value_cfg, *, default_low: float, default_high: float) -> float:
    """Internal helper to midpoint from cfg."""
    if isinstance(value_cfg, (list, tuple)) and len(value_cfg) >= 2:
        return 0.5 * (float(value_cfg[0]) + float(value_cfg[1]))
    if isinstance(value_cfg, (int, float)):
        return float(value_cfg)
    if isinstance(value_cfg, dict):
        if value_cfg.get("bins"):
            mids = [0.5 * (float(item["low"]) + float(item["high"])) for item in value_cfg["bins"]]
            return float(np.mean(mids)) if mids else 0.5 * (default_low + default_high)
        low = float(value_cfg.get("low", default_low))
        high = float(value_cfg.get("high", default_high))
        return 0.5 * (low + high)
    return 0.5 * (default_low + default_high)


@lru_cache(maxsize=None)
def _load_dispatch_bus_costs(cost_table_path: str) -> Dict[int, Dict[str, float | str]]:
    """Load generator dispatch-cost coefficients indexed by generator bus."""
    path = Path(cost_table_path)
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    generators = payload.get("generators")
    if not isinstance(generators, list) or not generators:
        raise RuntimeError(f"Invalid dispatch cost file at {path}: missing non-empty 'generators' list.")

    costs: Dict[int, Dict[str, float | str]] = {}
    for row in generators:
        if not isinstance(row, dict):
            raise RuntimeError(f"Invalid dispatch cost row in {path}: expected mapping, got {type(row).__name__}.")
        try:
            bus = int(row["bus"])
            if bus in costs:
                raise RuntimeError(f"Duplicate generator bus entry in dispatch cost file: {path} (bus {bus})")
            label = str(row.get("label", f"bus_{bus}"))
            costs[bus] = {
                "label": label,
                "a": float(row["a"]),
                "b": float(row["b"]),
                "c": float(row["c"]),
                "b_r": float(row["b_r"]),
            }
        except KeyError as exc:
            raise RuntimeError(f"Missing required dispatch cost key {exc!s} in {path}.") from exc
        except Exception as exc:
            raise RuntimeError(f"Invalid dispatch cost row for bus={row.get('bus')} in {path}.") from exc

    return costs


def _dispatch_cost_arrays(ss, ed_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed ED coefficient arrays from generator bus numbers."""
    cost_table_path = _resolve_repo_path(ed_cfg.get("cost_table_path"), DEFAULT_DISPATCH_COST_PATH)
    table_costs = _load_dispatch_bus_costs(str(cost_table_path))
    gen_buses = [int(v) for v in list(ss.PV.bus.v) + list(ss.Slack.bus.v)]
    a = np.zeros(len(gen_buses), dtype=float)
    b = np.zeros(len(gen_buses), dtype=float)
    c = np.zeros(len(gen_buses), dtype=float)
    b_r = np.zeros(len(gen_buses), dtype=float)
    missing: List[int] = []

    for i, bus in enumerate(gen_buses):
        coeffs = table_costs.get(bus)
        if coeffs is None:
            missing.append(bus)
            continue
        a[i] = float(coeffs["a"])
        b[i] = float(coeffs["b"])
        c[i] = float(coeffs["c"])
        b_r[i] = float(coeffs["b_r"])

    if missing:
        raise RuntimeError(
            "Missing ED coefficients for generator buses: "
            + ", ".join(str(bus) for bus in missing)
        )

    return a, b, c, b_r


def _build_ed_line_constraints(ss, Pg_var):
    """Build PTDF-based line flow limits for ED in p.u. space."""
    try:
        from andes.interop import pandapower as ap
        from pandapower import auxiliary as aux
        from pandapower.pd2ppc import _pd2ppc
        from pandapower.pypower.makePTDF import makePTDF
        from pandapower.pypower.idx_brch import RATE_A
    except Exception as exc:
        raise RuntimeError(
            "ED with line limits requires pandapower (andes.interop + pypower PTDF)."
        ) from exc

    pp_net = _run_with_suppressed_pandapower_noise(ap.to_pandapower, ss, verify=False)
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
    _, ppci = _run_with_suppressed_pandapower_noise(_pd2ppc, pp_net)
    branch = np.asarray(np.real(ppci["branch"]), dtype=float)
    ptdf = np.asarray(makePTDF(ppci["baseMVA"], ppci["bus"], branch, using_sparse_solver=False), dtype=float)
    base_mva = float(ppci.get("baseMVA", np.nan))
    if not np.isfinite(base_mva) or base_mva <= 0:
        raise RuntimeError("Could not read valid baseMVA from pandapower case for ED line limits.")

    # RATE_A (branch column index from pypower idx_brch)
    fmax_raw = np.asarray(branch[:, RATE_A], dtype=float)

    # 99999/1e10-style RATE_A placeholders mean "unlimited" and should not be
    # accepted silently for constrained ED.
    sentinel_threshold = float(1.0e4)
    valid_direct = np.isfinite(fmax_raw) & (fmax_raw > 0) & (fmax_raw < sentinel_threshold)

    if not np.all(valid_direct):
        # Fallback from ANDES line ratings when branch/line row alignment exists.
        n_line = int(getattr(getattr(ss, "Line", None), "n", 0))
        rate_a_vals = list(getattr(getattr(getattr(ss, "Line", None), "rate_a", None), "v", []))
        sn_vals = list(getattr(getattr(getattr(ss, "Line", None), "Sn", None), "v", []))

        if branch.shape[0] == n_line and n_line > 0:
            for i in np.where(~valid_direct)[0]:
                ra = np.nan
                if i < len(rate_a_vals):
                    try:
                        ra = float(rate_a_vals[i])
                    except Exception:
                        ra = np.nan
                if not (np.isfinite(ra) and ra > 0 and ra < sentinel_threshold):
                    if i < len(sn_vals):
                        try:
                            ra = float(sn_vals[i])
                        except Exception:
                            ra = np.nan
                if np.isfinite(ra) and ra > 0 and ra < sentinel_threshold:
                    fmax_raw[i] = ra

        valid_direct = np.isfinite(fmax_raw) & (fmax_raw > 0) & (fmax_raw < sentinel_threshold)
        if not np.all(valid_direct):
            bad = np.where(~valid_direct)[0]
            preview = ", ".join(str(int(i)) for i in bad[:10])
            raise RuntimeError(
                "Invalid/unset branch limits for ED line constraints "
                f"(RATE_A <=0, NaN, or >= {sentinel_threshold:g}) on {bad.size} branches. "
                f"Example indices: [{preview}]"
            )

    fmax_pu = fmax_raw / base_mva
    valid = np.isfinite(fmax_pu) & (fmax_pu > 0)
    if not np.any(valid):
        raise RuntimeError("No valid branch limits remain after RATE_A sanitization for ED line constraints.")
    ptdf = ptdf[valid, :]
    fmax_pu = fmax_pu[valid]

    bus_ids = pp_net.bus.index.to_numpy(dtype=int)
    bus_pos = {int(bus): i for i, bus in enumerate(bus_ids)}
    bus_df = ss.Bus.as_df()[["idx"]]
    bus_num_to_uid = {int(row.idx): int(uid) for uid, row in bus_df.iterrows()}

    gen_buses = np.asarray(ss.PV.bus.v + ss.Slack.bus.v, dtype=int)
    load_buses = np.asarray(ss.PQ.bus.v, dtype=int)
    pd_vec = np.asarray(ss.PQ.p0.v, dtype=float)

    n_bus = int(bus_ids.size)
    ng = int(gen_buses.size)
    nd = int(load_buses.size)
    Cg = np.zeros((n_bus, ng), dtype=float)
    Cd = np.zeros((n_bus, nd), dtype=float)

    for j, bus_num in enumerate(gen_buses):
        uid = bus_num_to_uid.get(int(bus_num))
        if uid is None or uid not in bus_pos:
            raise RuntimeError(f"Could not map generator bus {int(bus_num)} to pandapower bus index.")
        Cg[bus_pos[uid], j] += 1.0
    for j, bus_num in enumerate(load_buses):
        uid = bus_num_to_uid.get(int(bus_num))
        if uid is None or uid not in bus_pos:
            raise RuntimeError(f"Could not map load bus {int(bus_num)} to pandapower bus index.")
        Cd[bus_pos[uid], j] += 1.0

    injections = Cg @ Pg_var - Cd @ pd_vec
    flows = ptdf @ injections
    return [flows <= fmax_pu, flows >= -fmax_pu]


def run_ed_dispatch(
    ss,
    ed_cfg: Dict,
    ibr_idx: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    """Solve ED with fixed cost coefficients on the PV+Slack dispatch vector."""
    import cvxpy as cp  # lazy import so runs without ED dependency when disabled
    _ = ibr_idx
    ng = ss.PV.n + ss.Slack.n
    if ng == 0:
        return np.zeros(0), {
            "ed_enabled": 1,
            "ed_solver": str(ed_cfg.get("solver", "OSQP")),
            "ed_status": "empty",
            "ed_total_cost": 0.0,
            "ed_constant_cost": 0.0,
            "ed_energy_cost": 0.0,
            "ed_reserve_cost": 0.0,
            "ed_quadratic_cost": 0.0,
        }

    Pd = float(np.sum(ss.PQ.p0.v))
    Pg_min = np.asarray(ss.PV.pmin.v + ss.Slack.pmin.v, dtype=float)
    Pg_max = np.asarray(ss.PV.pmax.v + ss.Slack.pmax.v, dtype=float)
    a, b, c, b_r = _dispatch_cost_arrays(ss, ed_cfg)
    if a.size != ng:
        raise RuntimeError(
            f"Dispatch-cost coefficient size mismatch: expected {ng}, got {a.size}."
        )

    Pg = cp.Variable(ng)
    constraints = [cp.sum(Pg) == Pd, Pg >= Pg_min, Pg <= Pg_max]
    if bool(ed_cfg.get("line_limits_enable", True)):
        constraints += _build_ed_line_constraints(ss, Pg)
    objective = cp.Minimize(
        cp.sum(
            c
            + cp.multiply(b, Pg)
            # + cp.multiply(b_r, Pg_max - Pg) # Reserve cost is not considered in the data generation pipeline.
            + cp.multiply(a, cp.square(Pg))
        )
    )
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=ed_cfg.get("solver", "OSQP"), verbose=bool(ed_cfg.get("verbose", False)))
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"ED solve failed with status={prob.status}")
    Pg_val = np.asarray(Pg.value, dtype=float).reshape(-1)

    constant_cost = float(np.sum(a))
    energy_cost = float(np.sum(b * Pg_val))
    reserve_cost = float(np.sum(b_r * Pg_val))
    quadratic_cost = float(np.sum(c * (Pg_val**2)))
    total_cost = constant_cost + energy_cost + reserve_cost + quadratic_cost
    
    ss.PV.p0.v = Pg_val[: ss.PV.n]
    ss.Slack.p0.v = Pg_val[ss.PV.n :]

    # genrou_dispatch_by_name: Dict[str, float] = {}
    # if hasattr(ss, "GENROU") and ss.GENROU.n:
    #     genrou_names = [str(value) for value in list(getattr(ss.GENROU.name, "v", []))]
    #     genrou_static_idx = list(getattr(getattr(ss.GENROU, "gen", None), "v", []))
    #     for local_idx, static_idx_raw in enumerate(genrou_static_idx):
    #         try:
    #             static_idx = int(static_idx_raw) - 1
    #         except Exception:
    #             continue
    #         if 0 <= static_idx < Pg_val.size and local_idx < len(genrou_names):
    #             genrou_dispatch_by_name[genrou_names[local_idx]] = float(Pg_val[static_idx])

    # if hasattr(ss, "REGCV1") and ss.REGCV1.n:
    #     for local_idx, gen_idx in enumerate(ibr_idx):
    #         if 0 <= gen_idx < Pg_val.size and local_idx < ss.REGCV1.n:
    #             dispatch_value = float(Pg_val[gen_idx])
    #             try:
    #                 ss.REGCV1.pref.v[local_idx] = dispatch_value
    #             except Exception:
    #                 pass
    #             # Keep any exposed secondary reference aligned with the ED dispatch
    #             # so TDS does not create an artificial post-initialization transient.
    #             for attr_name in ("Pref2", "pref2"):
    #                 try:
    #                     attr = getattr(ss.REGCV1, attr_name, None)
    #                     values = getattr(attr, "v", None)
    #                     if values is not None and local_idx < len(values):
    #                         values[local_idx] = dispatch_value
    #                 except Exception:
    #                     pass

    return Pg_val, {
        "ed_enabled": 1,
        "ed_solver": str(ed_cfg.get("solver", "OSQP")),
        "ed_status": str(prob.status),
        "ed_total_cost": total_cost,
        "ed_constant_cost": constant_cost,
        "ed_energy_cost": energy_cost,
        "ed_reserve_cost": reserve_cost,
        "ed_quadratic_cost": quadratic_cost,
    }


# def _sync_static_generators_from_pflow(ss) -> None:
#     """Sync solved PF injections back into static-generator p0/q0 parameters.

#     ANDES dynamic model initialization uses static-generator p0/q0 values for
#     several linked dynamic devices. After PFlow, PV.q0 and Slack.p0/q0 can be
#     stale unless we copy the solved injections back explicitly.
#     """

#     pv_p = np.asarray(getattr(getattr(ss.PV, "p", None), "v", []), dtype=float)
#     pv_q = np.asarray(getattr(getattr(ss.PV, "q", None), "v", []), dtype=float)
#     if pv_p.size == ss.PV.n:
#         ss.PV.p0.v[:] = pv_p
#     if pv_q.size == ss.PV.n:
#         ss.PV.q0.v[:] = pv_q

#     slack_p = np.asarray(getattr(getattr(ss.Slack, "p", None), "v", []), dtype=float)
#     slack_q = np.asarray(getattr(getattr(ss.Slack, "q", None), "v", []), dtype=float)
#     if slack_p.size == ss.Slack.n:
#         ss.Slack.p0.v[:] = slack_p
#     if slack_q.size == ss.Slack.n:
#         ss.Slack.q0.v[:] = slack_q


# def _sync_dynamic_references_from_static_generators(ss) -> None:
#     """Align dynamic-device references with the current static-generator p0/q0."""

#     static_p = np.asarray(ss.PV.p0.v + ss.Slack.p0.v, dtype=float)

#     if hasattr(ss, "REGCV1") and ss.REGCV1.n:
#         regcv1_gen_vals = list(getattr(getattr(ss.REGCV1, "gen", None), "v", []))
#         for local_idx, gen_idx_raw in enumerate(regcv1_gen_vals):
#             try:
#                 gen_idx = int(gen_idx_raw) - 1
#             except Exception:
#                 continue
#             if not (0 <= gen_idx < static_p.size):
#                 continue
#             dispatch_value = float(static_p[gen_idx])
#             try:
#                 ss.REGCV1.pref.v[local_idx] = dispatch_value
#             except Exception:
#                 pass
#             for attr_name in ("Pref2", "pref2"):
#                 try:
#                     attr = getattr(ss.REGCV1, attr_name, None)
#                     values = getattr(attr, "v", None)
#                     if values is not None and local_idx < len(values):
#                         values[local_idx] = dispatch_value
#                 except Exception:
#                     pass


def _apply_base_operating_point_scale(ss, base_scale: float, *, scale_pv: bool) -> None:
    """
    Apply the sampled base operating-point scale before ED/TDS.

    The operating-point definition always rescales PQ demand. PV scaling is optional and
    controlled separately to keep the behavior explicit.
    """
    for uid in range(ss.PQ.n):
        ss.PQ.p0.v[uid] = ss.PQ.p0.v[uid] * base_scale
        ss.PQ.q0.v[uid] = ss.PQ.q0.v[uid] * base_scale
    if not scale_pv:
        return
    for uid in range(ss.PV.n):
        ss.PV.p0.v[uid] = ss.PV.p0.v[uid] * base_scale
        ss.PV.q0.v[uid] = ss.PV.q0.v[uid] * base_scale
        
    ss.Slack.p0.v[0] = ss.Slack.p0.v[0] * base_scale
    ss.Slack.q0.v[0] = ss.Slack.q0.v[0] * base_scale


def _assign_vis_coefficients(ss, M_vec: np.ndarray, D_vec: np.ndarray) -> None:
    """Internal helper to assign vis coefficients."""
    if not getattr(getattr(ss, "REGCV1", None), "n", 0):
        return
    ss.REGCV1.M.v, ss.REGCV1.D.v = M_vec, D_vec


def _configure_pq_model(ss) -> None:
    """Internal helper to configure pq model."""
    ss.PQ.config.p2p = 1
    ss.PQ.config.q2q = 1
    ss.PQ.config.p2z = 0
    ss.PQ.config.q2z = 0
    ss.PQ.config.p2i = 0
    ss.PQ.config.q2i = 0
    ss.PQ.config.pq2z = 0
    # TODO: put somewhere where is more logical
    ss.PV.config.allow_adjust = 0


def configure_tds(ss, tds_cfg: Dict) -> None:
    """Configure ANDES TDS settings from the generation config."""
    ss.TDS.config.no_tqdm = bool(tds_cfg.get("no_tqdm", True))
    ss.TDS.config.criteria = int(tds_cfg.get("criteria", 0))
    ss.TDS.config.tol = float(tds_cfg.get("tol", 1e-3))
    ss.TDS.config.tf = float(tds_cfg["t_end"])
    ss.TDS.config.tstep = float(tds_cfg["t_step"])
    ss.TDS.config.fixt = int(tds_cfg.get("fixt", 0))
    ss.TDS.config.method = str(tds_cfg.get("method", "backeuler"))
    ss.TDS.config.honest = int(tds_cfg.get("honest", 1))
    ss.TDS.config.max_iter = int(tds_cfg.get("max_iter", 35))
    ss.TDS.config.shrinkt = int(tds_cfg.get("shrinkt", 1))


def define_operating_point(
    ss,
    *,
    base_scale: float,
    M_vec: np.ndarray,
    D_vec: np.ndarray,
    ed_cfg: Dict,
    scale_pv: bool,
) -> Dict[str, object]:
    """
    Define the pre-disturbance operating point and optionally solve ED.

    Sequence:
    1. Rescale the base operating point.
    2. Assign sampled VIS coefficients.
    3. Solve the pre-contingency ED over the PV+Slack dispatch vector.
       The ED objective uses the fixed dispatch-cost coefficients and can enforce PTDF-based
       line-flow limits through `_build_ed_line_constraints(...)`.
    4. Freeze the resulting PQ state so the prefault x_op/x_cont features can be extracted later.
    """
    _apply_base_operating_point_scale(ss, base_scale, scale_pv=scale_pv)
    _assign_vis_coefficients(ss, M_vec, D_vec)

    ed_meta: Dict[str, float | str] = {
        "ed_enabled": 1 if bool(ed_cfg.get("enable", False)) else 0,
        "ed_solver": str(ed_cfg.get("solver", "OSQP")) if ed_cfg.get("enable", False) else "",
        "ed_status": "disabled" if not ed_cfg.get("enable", False) else "",
        "ed_total_cost": np.nan,
        "ed_constant_cost": np.nan,
        "ed_energy_cost": np.nan,
        "ed_reserve_cost": np.nan,
        "ed_quadratic_cost": np.nan,
    }
    pg_dispatch = np.array([], dtype=float)
    if ed_cfg.get("enable", False):
        ibr_idx = _dispatch_ibr_indices(ss, fallback=ed_cfg.get("ibr_idx") or [])
        pg_dispatch, ed_meta = run_ed_dispatch(
            ss,
            ed_cfg,
            ibr_idx=ibr_idx,
        )

    _configure_pq_model(ss)

    pq_p_before = np.asarray(ss.PQ.p0.v, dtype=float).copy()
    pq_q_before = np.asarray(ss.PQ.q0.v, dtype=float).copy()
    return {
        "pq_p_before": pq_p_before,
        "pq_q_before": pq_q_before,
        "base_load_q_total": float(np.sum(pq_q_before)) if pq_q_before.size else 0.0,
        "pg_dispatch": pg_dispatch,
        "ed_meta": ed_meta,
    }


def _discover_initial_state_fieldnames(
    *,
    cfg: Dict,
    case_path: str,
    regcv1_ids: Sequence[str],
    andes_opts: Dict,
) -> List[str]:
    """Internal helper to discover initial state fieldnames."""
    ss = andes.load(case_path, setup=False, **andes_opts)
    ss.config.freq = float(50)

    base_scale = _midpoint_from_cfg(
        cfg.get("base_load_scale", {}),
        default_low=0.3,
        default_high=1.0,
    )
    n_regcv1 = len(regcv1_ids)
    m_mid = _midpoint_from_cfg(cfg.get("ibr", {}).get("M_range", {}), default_low=0.0, default_high=8.0)
    d_mid = _midpoint_from_cfg(cfg.get("ibr", {}).get("D_range", {}), default_low=0.0, default_high=6.0)
    M_vec = np.full(n_regcv1, m_mid, dtype=float)
    D_vec = np.full(n_regcv1, d_mid, dtype=float)

    dry_ed_cfg = dict(cfg.get("ed", {}))
    dry_ed_cfg["enable"] = False

    define_operating_point(
        ss,
        base_scale=base_scale,
        M_vec=M_vec,
        D_vec=D_vec,
        ed_cfg=dry_ed_cfg,
        scale_pv=bool(cfg.get("load", {}).get("scale_pv", False)),
    )
    ss.setup()
    ss.PFlow.run()
    configure_tds(ss, cfg["tds"])
    ss.TDS.init()
    ss.TDS.load_plotter()
    fieldnames = initial_state_fieldnames_from_plotter(
        ss.TDS.plotter,
        feature_names_path=_feature_names_path(cfg),
    )
    if fieldnames:
        return fieldnames

    ss.TDS.config.tf = float(ss.TDS.config.tstep)
    ss.TDS.run()
    ss.TDS.load_plotter()
    return initial_state_fieldnames_from_plotter(
        ss.TDS.plotter,
        feature_names_path=_feature_names_path(cfg),
    )

def _run_worker(
    worker_id: int,
    sim_ids: Sequence[int],
    cfg: Dict,
    case_path: str,
    pq_names: Sequence[str],
    owner_map: Dict[str, str],
    regcv1_ids: Sequence[str],
    line_uids: Sequence[int],
    feature_names_path: str,
    fieldnames: Sequence[str],
    output_dir: Path,
) -> Optional[str]:
    """Internal helper to run worker."""
    if not sim_ids:
        return None
    worker_csv = _worker_output_path(output_dir, cfg.get("output_csv", "simulation_results.csv"), worker_id)
    with open(worker_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sim_id in sim_ids:
            rng = _rng_for_sim(int(cfg["seed"]), sim_id)
            rows = run_single_sim(
                cfg,
                sim_id,
                rng,
                case_path,
                pq_names,
                owner_map,
                regcv1_ids,
                line_uids,
                feature_names_path,
                output_dir,
            )
            for row in rows:
                for name in fieldnames:
                    row.setdefault(name, np.nan)
                writer.writerow(row)
    return str(worker_csv)


def _merge_worker_csvs(csv_path: Path, worker_paths: Iterable[Path], fieldnames: Sequence[str]) -> None:
    """Internal helper to merge worker csvs."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for worker_path in worker_paths:
            with open(worker_path, "r", newline="", encoding="utf-8") as f_in:
                reader = csv.DictReader(f_in)
                for row in reader:
                    writer.writerow(row)


def _log_row_nan_health(row: Dict, *, sim_id: int) -> None:
    """Internal helper to log row nan health."""
    key_cols = [
        "line_rating",
        "pre_fault_flow",
        "pre_fault_loading",
        "pre_p_from",
        "system_max_loading_prefault",
        "system_mean_loading_prefault",
        "system_top5_loading_mean_prefault",
        "ptdf_l1_norm_outaged_line",
        "max_abs_lodf_row",
        "predicted_max_post_cont_loading_dc",
    ]
    nan_cols = [c for c in key_cols if c in row and (row[c] is None or not np.isfinite(float(row[c])))]
    if nan_cols:
        cont = row.get("line_uid", "unknown")
        print(f"[nan-health] sim_id={sim_id} cont={cont} NaN in: {', '.join(nan_cols)}")


def run_single_sim(
    cfg: Dict,
    sim_id: int,
    rng: np.random.Generator,
    case_path: str,
    pq_names: Sequence[str],
    owner_map: Dict[str, str],
    regcv1_ids: Sequence[str],
    line_uids: Sequence[int],
    feature_names_path: str,
    output_dir: Optional[Path] = None,
):
    """Run single sim."""
    base_scale = rng.uniform(cfg["base_load_scale"]["low"], cfg["base_load_scale"]["high"])

    M_samples = [
        sample_value(
            cfg.get("ibr", {}).get("M_range", {}),
            rng,
            default_low=cfg["ibr"]["M_range"][0],
            default_high=cfg["ibr"]["M_range"][1],
        )
        for _ in range(len(regcv1_ids))
    ]
    D_samples = [
        sample_value(
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
    # Use a dry system load to determine which line contingencies to run.
    andes_opts = {}
    pycode_env = os.environ.get("ANDES_PYCODE_PATH")
    if pycode_env:
        andes_opts["options"] = {"pycode_path": pycode_env}

    disturbance_ss = None
    disturbance_kind = resolve_disturbance_kind(cfg)
    if disturbance_kind in {"line_n1", "line_plus_load"}:
        disturbance_ss = andes.load(case_path, setup=False, **andes_opts)
    disturbance_dispatcher = DisturbanceDispatcher()
    disturbance_specs = disturbance_dispatcher.plan(disturbance_ss, cfg, rng)

    rows: List[Dict] = []
    include_initial_state = _include_initial_state(cfg)
    for spec in disturbance_specs:
        cont = spec.contingency()
        load_step_enabled = spec.kind in {"load_step", "line_plus_load"}
        load_step_time = float(spec.load_step_time)
        step_scale = float(spec.load_step_scale)
        step_bin = str(spec.meta.get("step_bin_label", "disabled"))
        trip_time = float(spec.trip_time)

        ss = andes.load(case_path, setup=False, **andes_opts)
        ss.config.freq = float(50)
        operating_point = define_operating_point(
            ss,
            base_scale=base_scale,
            M_vec=M_vec,
            D_vec=D_vec,
            ed_cfg=ed_cfg,
            scale_pv=bool(cfg.get("load", {}).get("scale_pv", False)),
        )
        pq_p_before = np.asarray(operating_point["pq_p_before"], dtype=float)
        pq_q_before = np.asarray(operating_point["pq_q_before"], dtype=float)
        base_load_q_total = float(operating_point["base_load_q_total"])
        pg_dispatch = np.asarray(operating_point["pg_dispatch"], dtype=float)
        ed_meta = dict(operating_point["ed_meta"])
        pq_owner_list = [owner_map.get(str(o), str(o)) for o in ss.PQ.owner.v]

        disturbance_dispatcher.apply(ss, spec, cfg, rng)

        ss.setup()
        ss.PFlow.run()
        # _sync_static_generators_from_pflow(ss)
        # _sync_dynamic_references_from_static_generators(ss)
        operating_point_snapshot = extract_operating_point_snapshot(ss)
        line_metrics_snapshot = extract_line_metrics(
            ss=ss,
            contingency=cont,
            line_uids=line_uids,
            feature_names_path=feature_names_path,
        )

        configure_tds(ss, cfg["tds"])

        ss.TDS.init()
        success = bool(ss.TDS.run())
        ss.TDS.load_plotter()
        if bool(cfg.get("debug", {}).get("save_coi_plots", False)) and output_dir is not None:
            save_debug_coi_plot(
                ss=ss,
                plotter=ss.TDS.plotter,
                output_dir=output_dir,
                sim_id=sim_id,
                contingency=cont,
                step_scale=step_scale,
            )
        if bool(cfg.get("debug", {}).get("save_coi_traces", False)) and output_dir is not None:
            trace_dir = Path(cfg.get("debug", {}).get("coi_trace_dir") or output_dir)
            save_coi_trace_csv(
                ss=ss,
                plotter=ss.TDS.plotter,
                output_dir=trace_dir,
                sim_id=sim_id,
            )
        initial_state_snapshot = (
            extract_initial_state_metrics(
                ss.TDS.plotter,
                feature_names_path=feature_names_path,
            )
            if include_initial_state
            else None
        )

        pq_p_after = ss.PQ.Ppf.v

        genrou_pg = np.asarray(getattr(ss.GENROU, "Pg", np.zeros(0)), dtype=float)
        if genrou_pg.size == 0 and hasattr(ss.GENROU, "p0"):
            genrou_pg = np.asarray(ss.GENROU.p0.v, dtype=float)
        if genrou_pg.size == 0 and pg_dispatch.size:
            genrou_pg = pg_dispatch[: ss.GENROU.n] if ss.GENROU.n else np.zeros(0)
        regcv1_pg = np.zeros(ss.REGCV1.n, dtype=float)
        if hasattr(ss.REGCV1, "Pref"):
            try:
                regcv1_pg = np.asarray(ss.REGCV1.Pref.v, dtype=float)
            except Exception:
                regcv1_pg = np.zeros(ss.REGCV1.n, dtype=float)
        if pg_dispatch.size:
            ibr_idx = _dispatch_ibr_indices(ss, fallback=ed_cfg.get("ibr_idx") or []) or list(range(ss.REGCV1.n))
            for local_idx, gen_idx in enumerate(ibr_idx):
                if 0 <= gen_idx < pg_dispatch.size and local_idx < regcv1_pg.size:
                    regcv1_pg[local_idx] = pg_dispatch[gen_idx]

        row = extract_simulation_row(
            ss=ss,
            base_load_scale=base_scale,
            load_step_scale=step_scale,
            load_step_time=load_step_time,
            pq_names=pq_names,
            pq_owners=pq_owner_list,
            pq_p_before=pq_p_before,
            pq_q_before=pq_q_before,
            pq_p_after=pq_p_after,
            base_load_q_total=base_load_q_total,
            M_vec=M_vec,
            D_vec=D_vec,
            genrou_pg=genrou_pg,
            regcv1_pg=regcv1_pg,
            success=success,
            contingency=cont,
            load_step_enabled=load_step_enabled,
            trip_time=trip_time,
            line_uids=line_uids,
            line_metrics_snapshot=line_metrics_snapshot,
            operating_point_snapshot=operating_point_snapshot,
            initial_state_snapshot=initial_state_snapshot,
            include_initial_state=include_initial_state,
            ed_meta=ed_meta,
            plotter=ss.TDS.plotter,
            feature_names_path=feature_names_path,
        )

        row["sim_id"] = sim_id
        row["seed"] = int(cfg["seed"])
        row["step_bin_label"] = step_bin
        row["M_bin_label"] = M_bin_label
        row["D_bin_label"] = D_bin_label
        _log_row_nan_health(row, sim_id=sim_id)

        rows.append(row)

    return rows


def _andes_options_from_env() -> Dict:
    """Internal helper to andes options from env."""
    andes_opts: Dict = {}
    pycode_env = os.environ.get("ANDES_PYCODE_PATH")
    if pycode_env:
        andes_opts["options"] = {"pycode_path": pycode_env}
    return andes_opts


def _resolve_simulation_inputs(
    cfg: Dict,
    *,
    case_path: str,
    andes_opts: Dict,
) -> Dict[str, object]:
    """Internal helper to resolve simulation inputs."""
    base_ss = andes.load(case_path, setup=False, **andes_opts)
    pq_names = list(base_ss.PQ.name.v) if base_ss.PQ.n else []
    if cfg["load"].get("pq_names"):
        pq_names = [name for name in pq_names if name in cfg["load"]["pq_names"]]

    owner_map = {str(o): str(o) for o in set(base_ss.PQ.owner.v)} if getattr(base_ss, "PQ", None) else {}
    owner_labels = sorted(owner_map.values())

    regcv1_model = getattr(base_ss, "REGCV1", None)
    n_regcv1 = int(getattr(regcv1_model, "n", 0) or 0)
    regcv1_ids = list(cfg.get("ibr", {}).get("indices") or [])
    if n_regcv1 <= 0:
        cfg.setdefault("ibr", {})
        cfg["ibr"]["indices"] = []
        cfg["ibr"]["n_ibr"] = 0
        regcv1_ids = []
    else:
        if not regcv1_ids:
            n_ibr = int(cfg.get("ibr", {}).get("n_ibr", n_regcv1))
            regcv1_ids = list(getattr(getattr(base_ss.REGCV1, "idx", None), "v", []) or [])[:n_ibr]
        regcv1_ids = regcv1_ids[:n_regcv1]
    if n_regcv1 > 0 and not regcv1_ids:
        raise ValueError("No REGCV1 entries found to assign M/D parameters.")

    n_genrou = int(getattr(base_ss, "GENROU", None).n) if getattr(base_ss, "GENROU", None) else 0
    n_line = int(getattr(base_ss, "Line", None).n) if getattr(base_ss, "Line", None) else 0
    bus_idx_vals = list(getattr(getattr(getattr(base_ss, "Bus", None), "idx", None), "v", []))
    bus_numbers = [int(v) for v in bus_idx_vals]
    line_uids = list(range(n_line))
    feature_names_path = _feature_names_path(cfg)
    include_initial_state = _include_initial_state(cfg)
    initial_state_fields = (
        _discover_initial_state_fieldnames(
            cfg=cfg,
            case_path=case_path,
            regcv1_ids=regcv1_ids,
            andes_opts=andes_opts,
        )
        if include_initial_state
        else []
    )
    fieldnames = _build_fieldnames(
        pq_names,
        owner_labels,
        len(regcv1_ids),
        n_genrou,
        line_uids,
        bus_numbers,
        initial_state_fields,
        feature_names_path,
    )
    return {
        "pq_names": pq_names,
        "owner_map": owner_map,
        "regcv1_ids": regcv1_ids,
        "line_uids": line_uids,
        "feature_names_path": feature_names_path,
        "fieldnames": fieldnames,
        "include_initial_state": include_initial_state,
    }


def run_generation(config_path: str) -> Path:
    """Run generation."""
    print("Loading config from ", config_path)
    cfg = load_config(config_path)
    _assert_line_metrics_dc_ready(cfg)
    andes.config_logger(stream_level=int(cfg.get("stream_level", 30)))

    case_path = str(cfg["case"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / cfg.get("output_csv", "simulation_results.csv")
    andes_opts = _andes_options_from_env()

    sim_inputs = _resolve_simulation_inputs(cfg, case_path=case_path, andes_opts=andes_opts)
    pq_names = sim_inputs["pq_names"]
    owner_map = sim_inputs["owner_map"]
    regcv1_ids = sim_inputs["regcv1_ids"]
    line_uids = sim_inputs["line_uids"]
    feature_names_path = sim_inputs["feature_names_path"]
    fieldnames = sim_inputs["fieldnames"]

    n_sims = int(cfg["n_sims"])
    workers = max(1, int(cfg.get("workers", 1)))
    if workers <= 1 or n_sims <= 1:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sim_id in range(n_sims):
                sim_rng = _rng_for_sim(int(cfg["seed"]), sim_id)
                rows = run_single_sim(
                    cfg,
                    sim_id,
                    sim_rng,
                    case_path,
                    pq_names,
                    owner_map,
                    regcv1_ids,
                    line_uids,
                    feature_names_path,
                    output_dir,
                )
                for row in rows:
                    for name in fieldnames:
                        row.setdefault(name, np.nan)
                    writer.writerow(row)
        return csv_path

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
                    line_uids,
                    feature_names_path,
                    fieldnames,
                    output_dir,
                )
            )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(worker_args))) as pool:
        worker_paths = pool.starmap(_run_worker, worker_args)

    merge_paths = [Path(path) for path in worker_paths if path]
    _merge_worker_csvs(csv_path, merge_paths, fieldnames)
    return csv_path


def run_one_sim(
    config: Dict,
    sim_id: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Run one sim."""
    cfg = copy.deepcopy(config)
    case_path = str(cfg["case"])
    andes_opts = _andes_options_from_env()
    sim_inputs = _resolve_simulation_inputs(cfg, case_path=case_path, andes_opts=andes_opts)
    sim_rng = rng if rng is not None else _rng_for_sim(int(cfg["seed"]), sim_id)
    output_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    return run_single_sim(
        cfg,
        sim_id,
        sim_rng,
        case_path,
        sim_inputs["pq_names"],
        sim_inputs["owner_map"],
        sim_inputs["regcv1_ids"],
        sim_inputs["line_uids"],
        sim_inputs["feature_names_path"],
        output_dir,
    )


def main() -> None:
    """Run the CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run ANDES sims and extract metrics.")
    parser.add_argument("--config", default="configs/data_generation/generation.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    run_generation(args.config)


if __name__ == "__main__":
    main()
