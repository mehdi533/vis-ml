# TODO: prepare proof it works 

from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import cvxpy as cp
import andes


from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
from pandapower import auxiliary as aux


def build_pandapower_net(ss: andes.System):
    """Convert an ANDES System to a pandapower net."""
    try:
        from andes.interop import pandapower as ap
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("pandapower is required for PTDF extraction.") from exc
    return ap.to_pandapower(ss, verify=False)


def compute_ptdf(pp_net, *, use_sparse: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PTDF from a pandapower net.

    Returns
    -------
    ptdf : (n_line, n_bus) np.ndarray
    bus_ids : bus indices in PP order
    line_ids : line indices in PP order
    """

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
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"], using_sparse_solver=use_sparse)
    bus_ids = pp_net.bus.index.to_numpy()
    line_ids = pp_net.line.index.to_numpy()
    return np.asarray(ptdf, dtype=float), bus_ids, line_ids


def extract_fmax_from_pandapower(pp_net) -> np.ndarray:
    """
    Extract branch flow limits (fmax) from a pandapower net via the pypower case.

    Returns
    -------
    fmax : np.ndarray
        RATE_A values from ppci branch (MVA or MW-equivalent).
    """

    _, ppci = _pd2ppc(pp_net)
    branch = ppci["branch"]
    fmax = np.asarray(branch[:, 5], dtype=float)  # RATE_A
    return fmax


def build_injection_matrices(
    ss: andes.System,
    *,
    bus_ids: Sequence[int],
    gen_buses: Sequence[int] | None = None,
    load_buses: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build generator and load incidence matrices aligned with `bus_ids`.

    Returns
    -------
    Cg : (n_bus, n_gen) generator incidence
    Cd : (n_bus, n_load) load incidence
    """
    bus_pos = {int(bus): i for i, bus in enumerate(bus_ids)}
    # ANDES bus numbers are in ss.Bus.idx; pandapower uses uids as bus indices.
    bus_df = ss.Bus.as_df()[["idx"]]
    bus_num_to_uid = {int(row.idx): int(uid) for uid, row in bus_df.iterrows()}

    if gen_buses is None:
        gen_buses = np.concatenate(
            [np.asarray(ss.PV.bus.v, dtype=int), np.asarray(ss.Slack.bus.v, dtype=int)]
        )
    if load_buses is None:
        load_buses = np.asarray(ss.PQ.bus.v, dtype=int)

    Cg = np.zeros((len(bus_ids), len(gen_buses)), dtype=float)
    for j, bus in enumerate(gen_buses):
        uid = bus_num_to_uid.get(int(bus))
        if uid is None:
            raise KeyError(f"Bus {bus} not found in ANDES Bus.idx.")
        if uid not in bus_pos:
            raise KeyError(f"Bus uid {uid} not found in pandapower bus_ids.")
        Cg[bus_pos[uid], j] += 1.0

    Cd = np.zeros((len(bus_ids), len(load_buses)), dtype=float)
    for j, bus in enumerate(load_buses):
        uid = bus_num_to_uid.get(int(bus))
        if uid is None:
            raise KeyError(f"Bus {bus} not found in ANDES Bus.idx.")
        if uid not in bus_pos:
            raise KeyError(f"Bus uid {uid} not found in pandapower bus_ids.")
        Cd[bus_pos[uid], j] += 1.0

    return Cg, Cd


def compute_net_injections(
    Cg: np.ndarray,
    Pg: cp.Expression | np.ndarray,
    Cd: np.ndarray,
    Pd: cp.Expression | np.ndarray,
    *,
    shunt: cp.Expression | np.ndarray | None = None,
    bus_inj: cp.Expression | np.ndarray | None = None,
) -> cp.Expression:
    """
    Build net injections p = Cg*Pg - Cd*Pd - shunt - bus_inj.
    """
    p = Cg @ Pg - Cd @ Pd
    if shunt is not None:
        p = p - shunt
    if bus_inj is not None:
        p = p - bus_inj
    return p


def compute_line_flows(
    ptdf: np.ndarray,
    injections: cp.Expression | np.ndarray,
) -> cp.Expression:
    """Compute line flows f = PTDF @ injections."""
    return ptdf @ injections


def build_line_flow_constraints(
    flows: cp.Expression,
    *,
    fmax: Sequence[float],
    fmin: Sequence[float] | None = None,
) -> list[cp.Constraint]:
    """
    Build line flow constraints fmin <= f <= fmax.
    If fmin is None, symmetric limits -fmax are used.
    """
    fmax = np.asarray(fmax, dtype=float)
    if fmin is None:
        return [flows <= fmax, flows >= -fmax]
    fmin = np.asarray(fmin, dtype=float)
    if fmin.shape != fmax.shape:
        raise ValueError("fmin and fmax must have the same shape.")
    return [flows <= fmax, flows >= fmin]


def build_power_balance_constraint(
    injections: cp.Expression,
    *,
    per_bus: bool = False,
) -> list[cp.Constraint]:
    """
    Power-balance constraint: sum(injections)=0 or per-bus injections=0.
    """
    if per_bus:
        return [injections == 0]
    return [cp.sum(injections) == 0]
