from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import andes
import cvxpy as cp
import numpy as np
from andes.interop.pandapower import to_pandapower
from pandapower import auxiliary as aux
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeLODF import makeLODF
from pandapower.pypower.makePTDF import makePTDF


@dataclass(frozen=True)
class LineSecurityArtifacts:
    ppci: dict[str, Any]
    ptdf: np.ndarray
    fmax: np.ndarray
    Cg: np.ndarray
    Cd: np.ndarray
    critical_bus_nums: np.ndarray


def sanitize_branch_fmax(
    ss: andes.System,
    branch: np.ndarray,
    *,
    sentinel_threshold: float = 1.0e4,
) -> np.ndarray:
    """
    Sanitize RATE_A and convert to per-unit on ss.config.mva.
    """
    fmax_raw = np.asarray(np.real(branch[:, 5]), dtype=float).copy()
    valid_direct = np.isfinite(fmax_raw) & (fmax_raw > 0) & (fmax_raw < float(sentinel_threshold))

    if not np.all(valid_direct):
        n_line = int(getattr(getattr(ss, "Line", None), "n", 0))
        rate_a_vals = list(getattr(getattr(getattr(ss, "Line", None), "rate_a", None), "v", []))
        sn_vals = list(getattr(getattr(getattr(ss, "Line", None), "Sn", None), "v", []))

        if branch.shape[0] == n_line and n_line > 0:
            bad_idx = np.where(~valid_direct)[0]
            for i in bad_idx:
                ra = np.nan
                if i < len(rate_a_vals):
                    try:
                        ra = float(rate_a_vals[i])
                    except Exception:
                        ra = np.nan
                if not (np.isfinite(ra) and ra > 0 and ra < float(sentinel_threshold)):
                    if i < len(sn_vals):
                        try:
                            ra = float(sn_vals[i])
                        except Exception:
                            ra = np.nan
                if np.isfinite(ra) and ra > 0 and ra < float(sentinel_threshold):
                    fmax_raw[i] = ra

        valid_direct = np.isfinite(fmax_raw) & (fmax_raw > 0) & (fmax_raw < float(sentinel_threshold))
        if not np.all(valid_direct):
            bad = np.where(~valid_direct)[0]
            preview = ", ".join(str(int(i)) for i in bad[:10])
            raise RuntimeError(
                "Invalid branch limits (RATE_A <= 0, NaN, or sentinel). "
                f"Bad indices sample: [{preview}]"
            )

    return fmax_raw / ss.config.mva


def build_injection_matrices(ss: andes.System, *, bus_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    bus_pos = {int(bus): i for i, bus in enumerate(bus_ids)}
    bus_df = ss.Bus.as_df()[["idx"]]
    bus_num_to_uid = {int(row.idx): int(uid) for uid, row in bus_df.iterrows()}

    gen_buses = np.concatenate(
        [np.asarray(ss.PV.bus.v, dtype=int), np.asarray(ss.Slack.bus.v, dtype=int)]
    )
    load_buses = np.asarray(ss.PQ.bus.v, dtype=int)

    Cg = np.zeros((len(bus_ids), len(gen_buses)), dtype=float)
    for j, bus in enumerate(gen_buses):
        uid = bus_num_to_uid.get(int(bus))
        if uid is None or uid not in bus_pos:
            raise KeyError(f"Generator bus mapping failed for bus={bus}.")
        Cg[bus_pos[uid], j] += 1.0

    Cd = np.zeros((len(bus_ids), len(load_buses)), dtype=float)
    for j, bus in enumerate(load_buses):
        uid = bus_num_to_uid.get(int(bus))
        if uid is None or uid not in bus_pos:
            raise KeyError(f"Load bus mapping failed for bus={bus}.")
        Cd[bus_pos[uid], j] += 1.0

    return Cg, Cd


def build_input_feature_constraints(
    *,
    x: cp.Variable,
    x_seed_sc: np.ndarray,
    m_idx: Sequence[int],
    d_idx: Sequence[int],
    m_min_sc: np.ndarray,
    m_max_sc: np.ndarray,
    d_min_sc: np.ndarray,
    d_max_sc: np.ndarray,
    free_feature_idx: Sequence[int] | None = None,
    pg: cp.Variable | None = None,
    pg_feature_idx: Sequence[int] | None = None,
    x_scaler: Any | None = None,
    pg_link_order: Sequence[int] | None = None,
    x_feature_names: Sequence[str] | None = None,
    n_genrou: int | None = None,
    n_regcv1: int | None = None,
    genrou_m_fixed: Sequence[float] | None = None,
    genrou_d_fixed: Sequence[float] | None = None,
    pg_max: np.ndarray | None = None,
    dispatch_total_constant: float | None = None,
) -> list[cp.Constraint]:
    """
    Input block: fixed exogenous descriptors + bounded controllable features.
    """
    n_feat = int(x_seed_sc.size)
    free = set(int(i) for i in m_idx) | set(int(i) for i in d_idx)

    if free_feature_idx:
        free.update(int(i) for i in free_feature_idx)

    constraints: list[cp.Constraint] = []
    pg_link_idx: list[int] = []
    pg_n = int(pg.shape[0]) if pg is not None else 0

    if pg is not None and pg_feature_idx:
        pg_link_idx = [int(i) for i in pg_feature_idx]
        if len(pg_link_idx) != pg_n:
            raise ValueError(
                "Dispatch feature link mismatch: number of x dispatch features does not match number of pg variables "
                f"({len(pg_link_idx)} vs {pg_n})."
            )
        if min(pg_link_idx, default=0) < 0 or max(pg_link_idx, default=-1) >= n_feat:
            raise ValueError("pg_feature_idx contains out-of-range x indices.")
        if len(set(pg_link_idx)) != len(pg_link_idx):
            raise ValueError("pg_feature_idx contains duplicate x indices.")
        free.update(pg_link_idx)

    constraints += [x[list(m_idx)] >= m_min_sc, x[list(m_idx)] <= m_max_sc]
    constraints += [x[list(d_idx)] >= d_min_sc, x[list(d_idx)] <= d_max_sc]

    if pg is not None and pg_link_idx:
        if x_scaler is None:
            raise ValueError("x_scaler is required when linking x dispatch features to pg.")
        order = tuple(pg_link_order) if pg_link_order is not None else tuple(range(pg_n))
        if len(order) != pg_n:
            raise ValueError(f"pg_link_order length mismatch: expected {pg_n}, got {len(order)}")
        if min(order, default=0) < 0 or max(order, default=-1) >= pg_n:
            raise ValueError("pg_link_order contains index beyond pg length.")
        if len(set(order)) != len(order):
            raise ValueError("pg_link_order contains duplicate pg indices.")

        constraints += [
            x[pg_link_idx]
            == cp.multiply(cp.hstack([pg[i] for i in order]), x_scaler.scale_[pg_link_idx])
            + x_scaler.min_[pg_link_idx]
        ]

    derived_constraints: list[cp.Constraint] = []
    derived_linked_idx: set[int] = set()

    # Always set up helper functions and name mapping if we have the necessary info
    if x_scaler is not None and x_feature_names is not None:
        name_to_idx = {str(name): i for i, name in enumerate(x_feature_names)}

        def _scale_expr(idx: int, expr_unscaled):
            scale = float(x_scaler.scale_[idx])
            offset = float(x_scaler.min_[idx])
            if abs(scale) <= 1e-12:
                return None
            return expr_unscaled * scale + offset

        def _unscale_expr(idx: int, expr_scaled):
            scale = float(x_scaler.scale_[idx])
            offset = float(x_scaler.min_[idx])
            if abs(scale) <= 1e-12:
                raise ValueError(f"Cannot unscale feature at idx={idx}: scaler scale is ~0.")
            return (expr_scaled - offset) / scale

        def _link_derived(name: str, expr_unscaled) -> None:
            idx = name_to_idx.get(name)
            if idx is None:
                return
            scaled_expr = _scale_expr(idx, expr_unscaled)
            if scaled_expr is None:
                derived_constraints.append(x[idx] == float(x_seed_sc[idx]))
            else:
                derived_constraints.append(x[idx] == scaled_expr)
            derived_linked_idx.add(int(idx))

        # Link M_agg and D_agg based on M_i and D_i (no dispatch needed)
        n_gen = int(n_genrou if n_genrou is not None else 0)
        n_ibr = int(n_regcv1 if n_regcv1 is not None else len(m_idx))

        m_expr = []
        for i in range(n_ibr):
            name = f"M_{i + 1}"
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            m_expr.append(_unscale_expr(idx, x[idx]))

        d_expr = []
        for i in range(n_ibr):
            name = f"D_{i + 1}"
            idx = name_to_idx.get(name)
            if idx is None:
                continue
            d_expr.append(_unscale_expr(idx, x[idx]))

        sg_m_sum = float(np.sum(np.asarray(genrou_m_fixed, dtype=float))) if genrou_m_fixed is not None else 0.0
        sg_d_sum = float(np.sum(np.asarray(genrou_d_fixed, dtype=float))) if genrou_d_fixed is not None else 0.0
        n_total_md = int(n_gen + len(m_expr))
        if n_total_md > 0 and m_expr:
            _link_derived("M_agg", (sg_m_sum + cp.sum(cp.hstack(m_expr))) / float(n_total_md))
        if n_total_md > 0 and d_expr:
            _link_derived("D_agg", (sg_d_sum + cp.sum(cp.hstack(d_expr))) / float(n_total_md))

    # Dispatch-dependent derived features (P_GENROU_TOTAL, P_REGCV1_TOTAL, reserves, etc.)
    if (
        x_scaler is not None
        and x_feature_names is not None
        and pg is not None
        and pg_link_idx
    ):
        n_gen = int(n_genrou if n_genrou is not None else 0)
        n_ibr = int(n_regcv1 if n_regcv1 is not None else len(m_idx))
        dispatch_names = [f"P_GENROU_{i + 1}" for i in range(n_gen)] + [
            f"P_REGCV1_{i + 1}" for i in range(n_ibr)
        ]

        dispatch_expr_by_name: dict[str, cp.Expression] = {}
        dispatch_pmax_by_name: dict[str, float] = {}
        for local_idx, feat_name in enumerate(dispatch_names):
            if local_idx >= len(order):
                break
            pg_idx = int(order[local_idx])
            dispatch_expr_by_name[feat_name] = pg[pg_idx]
            if pg_max is not None and 0 <= pg_idx < int(np.asarray(pg_max).size):
                dispatch_pmax_by_name[feat_name] = float(np.asarray(pg_max, dtype=float)[pg_idx])

        p_genrou_expr = [dispatch_expr_by_name[f"P_GENROU_{i + 1}"] for i in range(n_gen) if f"P_GENROU_{i + 1}" in dispatch_expr_by_name]
        p_regcv1_expr = [dispatch_expr_by_name[f"P_REGCV1_{i + 1}"] for i in range(n_ibr) if f"P_REGCV1_{i + 1}" in dispatch_expr_by_name]

        if p_genrou_expr:
            p_genrou_total = cp.sum(cp.hstack(p_genrou_expr))
            _link_derived("P_GENROU_TOTAL", p_genrou_total)
        else:
            p_genrou_total = 0.0

        if p_regcv1_expr:
            p_regcv1_total = cp.sum(cp.hstack(p_regcv1_expr))
            _link_derived("P_REGCV1_TOTAL", p_regcv1_total)
        else:
            p_regcv1_total = 0.0

        _link_derived("P_DISPATCH_TOTAL", p_genrou_total + p_regcv1_total)

        if dispatch_total_constant is not None and abs(float(dispatch_total_constant)) > 1e-12:
            _link_derived("P_REGCV1_SHARE", p_regcv1_total / float(dispatch_total_constant))

        reserve_genrou_expr: list[cp.Expression] = []
        for i in range(n_gen):
            p_name = f"P_GENROU_{i + 1}"
            r_name = f"P_GENROU_RESERVE_{i + 1}"
            if p_name not in dispatch_expr_by_name or p_name not in dispatch_pmax_by_name:
                continue
            expr = float(dispatch_pmax_by_name[p_name]) - dispatch_expr_by_name[p_name]
            _link_derived(r_name, expr)
            reserve_genrou_expr.append(expr)

        reserve_regcv1_expr: list[cp.Expression] = []
        for i in range(n_ibr):
            p_name = f"P_REGCV1_{i + 1}"
            r_name = f"P_REGCV1_RESERVE_{i + 1}"
            if p_name not in dispatch_expr_by_name or p_name not in dispatch_pmax_by_name:
                continue
            expr = float(dispatch_pmax_by_name[p_name]) - dispatch_expr_by_name[p_name]
            _link_derived(r_name, expr)
            reserve_regcv1_expr.append(expr)

        if reserve_genrou_expr:
            _link_derived("reserve_p_genrou", cp.sum(cp.hstack(reserve_genrou_expr)))
        if reserve_regcv1_expr:
            _link_derived("reserve_p_ibr", cp.sum(cp.hstack(reserve_regcv1_expr)))
        if reserve_genrou_expr or reserve_regcv1_expr:
            _link_derived(
                "reserve_p_total_prefault",
                cp.sum(cp.hstack(reserve_genrou_expr + reserve_regcv1_expr)),
            )

    if x_feature_names is not None:
        name_to_idx = {str(name): i for i, name in enumerate(x_feature_names)}

        def _is_sched_feature(name: str) -> bool:
            """Only flag features that are actual scheduling decision variables or their
            direct derived aggregates.  The x_op reserve snapshots (reserve_p_genrou,
            reserve_p_ibr, reserve_p_total_prefault) are operational measurements and
            should be treated as fixed exogenous descriptors."""
            s = str(name)
            return (
                s.startswith("M_")
                or s.startswith("D_")
                or s.startswith("P_GENROU_")
                or s.startswith("P_REGCV1_")
                or s.startswith("P_DISPATCH_")
            )

        unresolved_sched: list[str] = []
        for name, idx in name_to_idx.items():
            if not _is_sched_feature(name):
                continue
            if int(idx) in free or int(idx) in derived_linked_idx:
                continue
            unresolved_sched.append(str(name))
        if unresolved_sched:
            preview = ", ".join(unresolved_sched[:12])
            if len(unresolved_sched) > 12:
                preview += ", ..."
            raise ValueError(
                "Some scheduling features are not decision-linked and would be fixed, "
                f"which violates the scheduling contract. Examples: {preview}"
            )

    eq_idx = np.asarray([i for i in range(n_feat) if i not in free and i not in derived_linked_idx], dtype=int)
    if eq_idx.size:
        constraints += [x[eq_idx] == x_seed_sc[eq_idx]]
    constraints += derived_constraints

    return constraints


def build_output_feature_constraints(
    *,
    y: cp.Variable,
    y_min_sc: np.ndarray,
    y_max_sc: np.ndarray,
    enforce_dispatch_output_link: bool,
    y_ibr_idx: Sequence[int],
    pg: cp.Variable,
    pg_base: np.ndarray,
    pg_min: np.ndarray,
    pg_max: np.ndarray,
    ibr_idx: Sequence[int],
    y_scaler: Any,
) -> list[cp.Constraint]:
    """
    Output block: dynamic-security bounds + optional Delta-P coupling.
    """
    constraints: list[cp.Constraint] = [y >= y_min_sc, y <= y_max_sc]
    if not enforce_dispatch_output_link:
        return constraints

    idx_y = np.asarray(y_ibr_idx, dtype=int)
    idx_ibr = np.asarray(ibr_idx, dtype=int)
    if idx_y.size == 0 or idx_ibr.size == 0:
        return constraints
    if idx_y.size != idx_ibr.size:
        raise ValueError("y_ibr_idx and ibr_idx must have same length.")

    dp_ibr = pg[idx_ibr] - pg_base[idx_ibr]
    p_min_sc = cp.multiply(pg_min[idx_ibr] - pg_base[idx_ibr], y_scaler.scale_[idx_y]) + y_scaler.min_[idx_y]
    p_max_sc = cp.multiply(pg_max[idx_ibr] - pg_base[idx_ibr], y_scaler.scale_[idx_y]) + y_scaler.min_[idx_y]
    dp_sc = cp.multiply(dp_ibr, y_scaler.scale_[idx_y]) + y_scaler.min_[idx_y]
    constraints += [y[idx_y] <= dp_sc, y[idx_y] >= p_min_sc, y[idx_y] <= p_max_sc]
    return constraints


def build_ed_constraints(
    *,
    pg: cp.Variable,
    pg_min: np.ndarray,
    pg_max: np.ndarray,
    pd: np.ndarray,
    step_scale: float,
) -> list[cp.Constraint]:
    """
    ED block from thesis Eq. (2.19b)-(2.19d).
    """
    return [
        pg >= pg_min,
        pg <= pg_max,
        cp.sum(pg) == float(np.sum(pd)) / float(step_scale),
    ]


def build_basecase_line_constraints(
    ss: andes.System,
    *,
    pg: cp.Variable,
    pd: np.ndarray,
) -> tuple[cp.Expression, list[cp.Constraint], LineSecurityArtifacts]:
    """
    Base-case DC line flow constraints from Eq. (2.20) + (2.19e).
    """
    pp_net = to_pandapower(ss, verify=False)
    pp_net._options = {}
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
    ptdf = np.asarray(makePTDF(ppci["baseMVA"], ppci["bus"], branch, using_sparse_solver=False), dtype=float)
    fmax = sanitize_branch_fmax(ss, branch)
    bus_ids = pp_net.bus.index.to_numpy()
    pp_bus_nums = np.asarray(ppci["bus"][:, 0], dtype=int)
    uid_to_pp_bus_num = {int(uid): int(pp_bus_nums[i]) for i, uid in enumerate(bus_ids)}
    Cg, Cd = build_injection_matrices(ss, bus_ids=bus_ids)

    injections = Cg @ pg - Cd @ pd
    flows = ptdf @ injections
    constraints = [flows <= fmax, flows >= -fmax]

    bus_df = ss.Bus.as_df()[["idx"]]
    bus_num_to_uid = {int(row.idx): int(uid) for uid, row in bus_df.iterrows()}
    gen_buses = np.concatenate(
        [np.asarray(ss.PV.bus.v, dtype=int), np.asarray(ss.Slack.bus.v, dtype=int)]
    )
    load_buses = np.asarray(ss.PQ.bus.v, dtype=int)
    critical_bus_nums: set[int] = set()
    for bus in np.concatenate([gen_buses, load_buses]):
        uid = bus_num_to_uid.get(int(bus))
        if uid is None:
            continue
        mapped = uid_to_pp_bus_num.get(int(uid))
        if mapped is not None:
            critical_bus_nums.add(int(mapped))

    artifacts = LineSecurityArtifacts(
        ppci=ppci,
        ptdf=ptdf,
        fmax=fmax,
        Cg=Cg,
        Cd=Cd,
        critical_bus_nums=np.asarray(sorted(critical_bus_nums), dtype=int),
    )
    return flows, constraints, artifacts


def _critical_outage_line_mask(branch_arr: np.ndarray, critical_buses: set[int]) -> np.ndarray:
    n_branch = int(branch_arr.shape[0])
    mask = np.zeros(n_branch, dtype=bool)
    if n_branch == 0 or not critical_buses:
        return mask

    in_service = np.ones(n_branch, dtype=bool)
    if branch_arr.shape[1] > 10:
        in_service = np.asarray(branch_arr[:, 10], dtype=float) > 0.5

    adjacency: dict[int, list[tuple[int, int]]] = {}
    for eidx in np.flatnonzero(in_service):
        i = int(eidx)
        u = int(branch_arr[i, 0])
        v = int(branch_arr[i, 1])
        if u == v:
            continue
        adjacency.setdefault(u, []).append((v, i))
        adjacency.setdefault(v, []).append((u, i))

    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    parent_bus: dict[int, int] = {}
    parent_edge: dict[int, int] = {}
    bridges: list[int] = []
    t = 0

    def _dfs(u: int) -> None:
        nonlocal t
        t += 1
        disc[u] = t
        low[u] = t
        for v, eidx in adjacency.get(u, []):
            if v not in disc:
                parent_bus[v] = u
                parent_edge[v] = eidx
                _dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append(int(eidx))
            elif not (parent_bus.get(u) == v and parent_edge.get(u) == eidx):
                low[u] = min(low[u], disc[v])

    for u in adjacency:
        if u in disc:
            continue
        parent_bus[u] = -1
        parent_edge[u] = -1
        _dfs(u)

    if not bridges:
        return mask

    def _reachable(start: int, blocked_edge: int) -> set[int]:
        seen: set[int] = set()
        stack = [int(start)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for nbr, eidx in adjacency.get(node, []):
                if eidx == blocked_edge:
                    continue
                if not in_service[eidx]:
                    continue
                if nbr not in seen:
                    stack.append(int(nbr))
        return seen

    for eidx in bridges:
        u = int(branch_arr[eidx, 0])
        v = int(branch_arr[eidx, 1])
        side_u = _reachable(u, eidx)
        side_v = _reachable(v, eidx)
        if side_u and side_v:
            if any(bus in critical_buses for bus in side_u) and any(bus in critical_buses for bus in side_v):
                mask[eidx] = True
    return mask


def build_n1_constraints(
    flows: cp.Expression,
    ppci: dict[str, Any],
    ptdf: np.ndarray,
    fmax: np.ndarray,
    *,
    critical_bus_nums: np.ndarray | None = None,
    exclude_islanding_critical: bool = True,
    collect_stats: bool = False,
) -> list[cp.Constraint] | tuple[list[cp.Constraint], dict[str, int | bool]]:
    """
    Preventive N-1 with LODF from thesis Eq. (4.18)-(4.19), without recourse.
    """
    branch = ppci["branch"]
    try:
        lodf = np.asarray(makeLODF(branch, ptdf), dtype=float)
    except TypeError:
        lodf = np.asarray(makeLODF(ptdf, branch), dtype=float)

    critical_set = set(
        np.asarray(
            critical_bus_nums if critical_bus_nums is not None else np.zeros(0, dtype=int),
            dtype=int,
        ).tolist()
    )
    skip_mask = (
        _critical_outage_line_mask(branch, critical_set)
        if exclude_islanding_critical
        else np.zeros(int(len(fmax)), dtype=bool)
    )

    constraints: list[cp.Constraint] = []
    skipped_nonfinite = 0
    skipped_critical = 0
    active_outages = 0

    for outage in range(len(fmax)):
        if skip_mask[outage]:
            skipped_critical += 1
            continue
        col = lodf[:, outage]
        if not np.all(np.isfinite(col)):
            skipped_nonfinite += 1
            continue
        f_post = flows + cp.multiply(col, flows[outage])
        constraints += [f_post <= fmax, f_post >= -fmax]
        active_outages += 1

    if not collect_stats:
        return constraints

    stats = {
        "n_total_outages": int(len(fmax)),
        "n_active_outages": int(active_outages),
        "n_skipped_nonfinite_lodf": int(skipped_nonfinite),
        "n_skipped_islanding_critical": int(skipped_critical),
        "exclude_islanding_critical": bool(exclude_islanding_critical),
    }
    return constraints, stats


def build_n1_redispatch_constraints(
    *,
    pg: cp.Variable,
    pd: np.ndarray,
    Cg: np.ndarray,
    Cd: np.ndarray,
    ptdf: np.ndarray,
    ppci: dict[str, Any],
    fmax: np.ndarray,
    pg_min: np.ndarray | None = None,
    pg_max: np.ndarray | None = None,
    redispatch_up: np.ndarray | None = None,
    redispatch_down: np.ndarray | None = None,
    critical_bus_nums: np.ndarray | None = None,
    exclude_islanding_critical: bool = True,
    collect_stats: bool = False,
) -> tuple[list[cp.Constraint], cp.Variable] | tuple[list[cp.Constraint], dict[str, int | bool], cp.Variable]:
    """
    Optional N-1 recourse-like redispatch block.

    For each outage c, a redispatch vector delta_pg[c] is introduced:
    - sum(delta_pg[c]) = 0
    - bounded by redispatch limits and/or generator bounds
    - post-outage flows computed with LODF on the redispatched pre-outage base flow.
    """
    branch = ppci["branch"]
    try:
        lodf = np.asarray(makeLODF(branch, ptdf), dtype=float)
    except TypeError:
        lodf = np.asarray(makeLODF(ptdf, branch), dtype=float)

    critical_set = set(
        np.asarray(
            critical_bus_nums if critical_bus_nums is not None else np.zeros(0, dtype=int),
            dtype=int,
        ).tolist()
    )
    skip_mask = (
        _critical_outage_line_mask(branch, critical_set)
        if exclude_islanding_critical
        else np.zeros(int(len(fmax)), dtype=bool)
    )

    active_outages = [i for i in range(len(fmax)) if not skip_mask[i]]
    n_active = len(active_outages)
    n_gen = int(pg.shape[0])
    delta_pg = cp.Variable((n_active, n_gen), name="delta_pg_n1")

    constraints: list[cp.Constraint] = []
    skipped_nonfinite = 0
    applied_outages = 0

    for row, outage in enumerate(active_outages):
        col = lodf[:, outage]
        if not np.all(np.isfinite(col)):
            skipped_nonfinite += 1
            continue

        delta = delta_pg[row, :]
        pg_post = pg + delta
        constraints += [cp.sum(delta) == 0]

        if redispatch_up is not None:
            constraints += [delta <= np.asarray(redispatch_up, dtype=float)]
        if redispatch_down is not None:
            constraints += [delta >= -np.asarray(redispatch_down, dtype=float)]

        if pg_min is not None:
            constraints += [pg_post >= np.asarray(pg_min, dtype=float)]
        if pg_max is not None:
            constraints += [pg_post <= np.asarray(pg_max, dtype=float)]

        injections_pre_outage = Cg @ pg_post - Cd @ pd
        flows_pre_outage = ptdf @ injections_pre_outage
        f_post = flows_pre_outage + cp.multiply(col, flows_pre_outage[outage])

        monitored = np.ones(len(fmax), dtype=bool)
        monitored[outage] = False
        if np.any(monitored):
            constraints += [f_post[monitored] <= fmax[monitored], f_post[monitored] >= -fmax[monitored]]
        applied_outages += 1

    if not collect_stats:
        return constraints, delta_pg

    stats = {
        "n_total_outages": int(len(fmax)),
        "n_active_outages": int(n_active),
        "n_outages_with_constraints": int(applied_outages),
        "n_skipped_nonfinite_lodf": int(skipped_nonfinite),
        "n_skipped_islanding_critical": int(np.sum(skip_mask)),
        "exclude_islanding_critical": bool(exclude_islanding_critical),
    }
    return constraints, stats, delta_pg


def evaluate_line_security_metrics(
    *,
    pg_value: np.ndarray,
    pd: np.ndarray,
    artifacts: LineSecurityArtifacts,
    include_n1: bool,
    exclude_islanding_critical: bool = True,
    tolerance: float = 1e-8,
) -> dict[str, float | int | bool]:
    """
    Evaluate base-case and preventive N-1 DC line loading metrics from a solved dispatch.
    """
    pg_arr = np.asarray(pg_value, dtype=float).reshape(-1)
    pd_arr = np.asarray(pd, dtype=float).reshape(-1)
    injections = artifacts.Cg @ pg_arr - artifacts.Cd @ pd_arr
    flows_base = artifacts.ptdf @ injections

    denom = np.where(np.abs(artifacts.fmax) > 1e-12, np.abs(artifacts.fmax), np.nan)
    loading_base = np.abs(flows_base) / denom
    loading_base_pct = 100.0 * loading_base
    base_violation = loading_base > (1.0 + float(tolerance))
    base_excess = np.maximum(loading_base - 1.0, 0.0)

    metrics: dict[str, float | int | bool] = {
        "base_max_loading_ratio": float(np.nanmax(loading_base)) if loading_base.size else np.nan,
        "base_max_loading_pct": float(np.nanmax(loading_base_pct)) if loading_base_pct.size else np.nan,
        "base_mean_loading_pct": float(np.nanmean(loading_base_pct)) if loading_base_pct.size else np.nan,
        "base_n_violations": int(np.sum(base_violation)) if base_violation.size else 0,
        "base_max_violation_ratio": float(np.nanmax(base_excess)) if base_excess.size else 0.0,
        "base_max_violation_pct": float(np.nanmax(100.0 * base_excess)) if base_excess.size else 0.0,
    }

    if not include_n1:
        metrics.update(
            {
                "n1_considered": False,
                "n1_n_total_outages": 0,
                "n1_n_active_outages": 0,
                "n1_n_skipped_nonfinite_lodf": 0,
                "n1_n_skipped_islanding_critical": 0,
                "n1_n_outages_with_violation": 0,
                "n1_total_line_violations": 0,
                "n1_max_loading_ratio": np.nan,
                "n1_max_loading_pct": np.nan,
                "n1_max_violation_ratio": np.nan,
                "n1_max_violation_pct": np.nan,
                "n1_worst_outage_index": -1,
            }
        )
        return metrics

    branch = artifacts.ppci["branch"]
    try:
        lodf = np.asarray(makeLODF(branch, artifacts.ptdf), dtype=float)
    except TypeError:
        lodf = np.asarray(makeLODF(artifacts.ptdf, branch), dtype=float)

    critical_set = set(np.asarray(artifacts.critical_bus_nums, dtype=int).tolist())
    skip_mask = (
        _critical_outage_line_mask(branch, critical_set)
        if exclude_islanding_critical
        else np.zeros(int(len(artifacts.fmax)), dtype=bool)
    )

    max_loading_ratio = -np.inf
    max_violation_ratio = 0.0
    worst_outage = -1
    n_outage_violation = 0
    total_line_viol = 0
    skipped_nonfinite = 0
    active_outages = 0

    for outage in range(len(artifacts.fmax)):
        if skip_mask[outage]:
            continue

        col = lodf[:, outage]
        if not np.all(np.isfinite(col)):
            skipped_nonfinite += 1
            continue

        active_outages += 1
        f_post = flows_base + col * flows_base[outage]
        monitored = np.ones(len(artifacts.fmax), dtype=bool)
        monitored[outage] = False
        if not np.any(monitored):
            continue

        loading = np.abs(f_post[monitored]) / denom[monitored]
        excess = np.maximum(loading - 1.0, 0.0)
        if loading.size:
            local_max_loading = float(np.nanmax(loading))
            if local_max_loading > max_loading_ratio:
                max_loading_ratio = local_max_loading
                worst_outage = int(outage)
        n_viol = int(np.sum(loading > (1.0 + float(tolerance))))
        if n_viol > 0:
            n_outage_violation += 1
            total_line_viol += n_viol
            max_violation_ratio = max(max_violation_ratio, float(np.nanmax(excess)))

    if max_loading_ratio == -np.inf:
        max_loading_ratio = np.nan

    metrics.update(
        {
            "n1_considered": True,
            "n1_n_total_outages": int(len(artifacts.fmax)),
            "n1_n_active_outages": int(active_outages),
            "n1_n_skipped_nonfinite_lodf": int(skipped_nonfinite),
            "n1_n_skipped_islanding_critical": int(np.sum(skip_mask)),
            "n1_n_outages_with_violation": int(n_outage_violation),
            "n1_total_line_violations": int(total_line_viol),
            "n1_max_loading_ratio": float(max_loading_ratio) if np.isfinite(max_loading_ratio) else np.nan,
            "n1_max_loading_pct": float(100.0 * max_loading_ratio) if np.isfinite(max_loading_ratio) else np.nan,
            "n1_max_violation_ratio": float(max_violation_ratio),
            "n1_max_violation_pct": float(100.0 * max_violation_ratio),
            "n1_worst_outage_index": int(worst_outage),
        }
    )
    return metrics
