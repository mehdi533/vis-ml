"""Turn a power-flow-only case (e.g. IEEE 118/300) into a TDS-ready dynamic case.

IEEE 118 and 300 ship with ANDES as MATPOWER power-flow cases: buses, lines and
static generators, but no dynamic machine models -- so they cannot run the
time-domain simulations VIS depends on. This utility assigns a synchronous
machine (GENROU) plus a governor (TGOV1N) to each static generator, using the
same model pair that is active in the thesis's working IEEE 39-bus case
(exciters are disabled there -> constant field, so none are added).

Parameters are standard round-rotor per-unit values on each machine's own MVA
base; inertia is set from a target H and each machine's MVA base is sized from
its dispatch with headroom. This is a *starting* dynamic dataset intended to make
the system integrable and frequency-meaningful -- not a validated manufacturer
model. Tuning H / governor droop per study is expected.
"""

from __future__ import annotations

from typing import Dict, List

# Standard round-rotor GENROU parameters (per-unit on machine base).
STANDARD_GENROU: Dict[str, float] = {
    "fn": 50.0,
    "D": 0.0,
    "ra": 0.0,
    "xl": 0.15,
    "xd": 1.9, "xq": 1.7,
    "xd1": 0.30, "xq1": 0.50,
    "xd2": 0.23, "xq2": 0.23,
    "Td10": 8.0, "Td20": 0.04,
    "Tq10": 0.9, "Tq20": 0.06,
    "S10": 0.0, "S12": 0.0,
    "gammap": 1.0, "gammaq": 1.0,
}

# TGOV1N governor (per-unit on machine base).
STANDARD_TGOV1N: Dict[str, float] = {
    "R": 0.05, "T1": 0.5, "T2": 1.0, "T3": 5.0,
    "VMAX": 1.2, "VMIN": 0.0, "Dt": 0.0, "wref0": 1.0,
}


def _base_mva(ss) -> float:
    for attr in ("mva",):
        v = getattr(getattr(ss, "config", None), attr, None)
        if v:
            return float(v)
    return float(getattr(ss, "mva", 100.0) or 100.0)


def dynamify_case(
    ss,
    target_H: float = 4.0,
    sn_floor: float = 100.0,
    sn_headroom: float = 1.3,
    exclude_gen_idxs=None,
    coi_idx=None,
) -> Dict[str, List]:
    """Attach GENROU + TGOV1N to every static generator of ``ss`` (setup=False).

    Call before ``ss.setup()``. Machine MVA base is ``max(sn_floor, existing Sn,
    sn_headroom * |Pg|_MW)``; inertia constant ``M = 2 * target_H``.

    ``exclude_gen_idxs`` lists static-generator idxs to skip -- use it for buses
    that will instead carry a grid-forming REGCV1 (a bus cannot sensibly hold both
    a synchronous GENROU and a grid-forming converter).

    Returns ``{"GENROU": [...idx], "TGOV1N": [...idx]}``.
    """
    base_mva = _base_mva(ss)
    exclude = set(exclude_gen_idxs or [])
    created: Dict[str, List] = {"GENROU": [], "TGOV1N": []}

    gens = []
    for mdl in ("PV", "Slack"):
        m = getattr(ss, mdl, None)
        if m is None or not getattr(m, "n", 0):
            continue
        idxs = list(m.idx.v)
        buses = list(m.bus.v)
        p0 = list(getattr(m, "p0").v) if hasattr(m, "p0") else [0.0] * len(idxs)
        sn_ex = list(getattr(m, "Sn").v) if hasattr(m, "Sn") else [sn_floor] * len(idxs)
        for k in range(len(idxs)):
            if idxs[k] in exclude:
                continue
            pg_mw = abs(float(p0[k])) * base_mva
            sn = max(float(sn_floor), float(sn_ex[k]) if sn_ex[k] else 0.0, sn_headroom * pg_mw)
            gens.append((idxs[k], buses[k], sn))

    for i, (gidx, bus, sn) in enumerate(gens, start=1):
        genrou_params = dict(
            STANDARD_GENROU, idx=f"GENROU_D{i}", name=f"GENROU_D{i}",
            gen=gidx, bus=bus, Sn=sn, M=2.0 * float(target_H),
        )
        if coi_idx is not None:
            genrou_params["coi"] = coi_idx
        g_idx = ss.add("GENROU", param_dict=genrou_params)
        created["GENROU"].append(g_idx)
        t_idx = ss.add("TGOV1N", param_dict=dict(
            STANDARD_TGOV1N, idx=f"TGOV1N_D{i}", name=f"TGOV1N_D{i}", syn=g_idx,
        ))
        created["TGOV1N"].append(t_idx)

    return created
