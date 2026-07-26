"""Declarative test-system registry + ANDES helpers for multi-system VIS.

Pure-Python parts (SystemSpec, registry, path resolution, REGCV1 template) import
with no heavy dependencies and are unit-tested directly. ANDES is imported lazily
inside the functions that need it, so importing this module stays cheap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]

# REGCV1 grid-forming converter parameter template, extracted from the thesis's
# hand-built ieee39_full_ibrs.xlsx (control gains left at 0 as in that case; M/D
# are the schedulable quantities, Sn/xs are per-device). Keys here are the fixed
# defaults; idx/name/bus/gen/Sn/M/D are supplied per device at augmentation time.
REGCV1_TEMPLATE: Dict[str, float] = {
    "fn": 50.0,
    "Tc": 0.01,
    "kw": 0.0,
    "kv": 0.0,
    "ra": 0.0,
    "gammap": 1.0,
    "gammaq": 1.0,
    "Kpvd": 0.0, "Kivd": 0.0, "Kpvq": 0.0, "Kivq": 0.0,
    "KpId": 0.0, "KiId": 0.0, "KpIq": 0.0, "KiIq": 0.0,
}


@dataclass(frozen=True)
class SystemSpec:
    """Declarative description of a VIS test system."""

    name: str
    case_path: str                 # repo-relative path, or "andes:<sub>/<file>" for a bundled case
    n_buses: int
    description: str
    ibr_gen_idxs: tuple = ()        # StaticGen idxs to run as grid-forming IBRs
    m_range: tuple = (0.0, 8.0)     # schedulable virtual-inertia range
    d_range: tuple = (0.0, 6.0)     # schedulable damping range
    default_ibr_Sn: float = 1040.0  # per-device MVA base if not read from the gen
    notes: str = ""

    @property
    def n_ibrs(self) -> int:
        return len(self.ibr_gen_idxs)


SYSTEM_REGISTRY: Dict[str, SystemSpec] = {
    "ieee39": SystemSpec(
        name="ieee39",
        case_path="data_generation/andes_cases/ieee39_full_ibrs.xlsx",
        n_buses=39,
        description="Modified IEEE 39-bus (10-machine) with 4 REGCV1 grid-forming IBRs. Thesis baseline.",
        ibr_gen_idxs=(1, 6, 8, 9),
        notes="Ships with REGCV1 already configured.",
    ),
    "npcc140": SystemSpec(
        name="npcc140",
        case_path="andes:npcc/npcc.xlsx",
        n_buses=140,
        description="NPCC 140-bus, 48-machine dynamic case (ANDES-bundled). 3.6x the IEEE 39-bus scale.",
        ibr_gen_idxs=(),  # to be chosen; augment with augment_with_grid_forming_ibrs
        notes="No REGCV1 in the base case; needs grid-forming augmentation to run VIS.",
    ),
}


def resolve_case_path(spec_or_path) -> Path:
    """Resolve a SystemSpec (or path string) to an absolute case file path.

    Supports the "andes:<subdir>/<file>" scheme for cases bundled with ANDES.
    """
    raw = spec_or_path.case_path if isinstance(spec_or_path, SystemSpec) else str(spec_or_path)
    if raw.startswith("andes:"):
        import andes  # lazy
        root = Path(os.path.dirname(andes.__file__)) / "cases"
        return root / raw[len("andes:"):]
    p = Path(raw)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def describe_system(spec_or_path, run_pflow: bool = True) -> Dict[str, object]:
    """Load a case and report a light structural summary (+ optional power flow).

    A single ANDES load; no time-domain simulation. Safe to run on a laptop.
    """
    import warnings
    import andes

    warnings.filterwarnings("ignore")
    andes.config_logger(stream_level=40)
    path = resolve_case_path(spec_or_path)
    ss = andes.load(str(path), setup=True, no_output=True)

    def _count(model: str) -> int:
        m = getattr(ss, model, None)
        return int(getattr(m, "n", 0)) if m is not None else 0

    summary: Dict[str, object] = {
        "case_path": str(path),
        "n_buses": _count("Bus"),
        "n_lines": _count("Line"),
        "n_genrou": _count("GENROU"),
        "n_gencls": _count("GENCLS"),
        "n_synchronous_machines": _count("GENROU") + _count("GENCLS"),
        "n_regcv1": _count("REGCV1"),
        "n_pq": _count("PQ"),
        "n_pv": _count("PV"),
        "n_slack": _count("Slack"),
    }
    if run_pflow:
        summary["pflow_converged"] = bool(ss.PFlow.run())
    return summary


def augment_with_grid_forming_ibrs(
    ss,
    gen_idxs: Sequence,
    m_values: Sequence[float],
    d_values: Sequence[float],
    sn_values: Optional[Sequence[float]] = None,
) -> list:
    """Attach REGCV1 grid-forming converters to the given static generators.

    Call on a system loaded with ``setup=False`` (devices must be added before
    setup); the caller then runs ``ss.setup()``. Each REGCV1 references its
    StaticGen and that generator's bus, using REGCV1_TEMPLATE for the fixed
    control parameters and the supplied M/D as the schedulable quantities.

    Returns the list of created REGCV1 idx strings.

    NOTE: for a valid *dynamic* (TDS) run the referenced generators should be
    grid-forming rather than also carrying a synchronous dynamic model; handling
    that model swap per system is the documented next step. Structural/power-flow
    use is supported directly.
    """
    if not (len(gen_idxs) == len(m_values) == len(d_values)):
        raise ValueError("gen_idxs, m_values, d_values must have equal length.")
    if sn_values is not None and len(sn_values) != len(gen_idxs):
        raise ValueError("sn_values, if given, must match gen_idxs length.")

    created = []
    for i, gen in enumerate(gen_idxs):
        # Look up the static generator's bus.
        try:
            row = ss.StaticGen.idx2uid(gen)
            bus = ss.StaticGen.get(src="bus", idx=gen, attr="v")
        except Exception:
            bus = None
        sn = float(sn_values[i]) if sn_values is not None else None
        params = dict(REGCV1_TEMPLATE)
        params.update(
            idx=f"REGCV1_AUG_{i + 1}",
            name=f"REGCV1_AUG_{i + 1}",
            gen=gen,
            M=float(m_values[i]),
            D=float(d_values[i]),
        )
        if bus is not None:
            params["bus"] = bus
        if sn is not None:
            params["Sn"] = sn
        idx = ss.add("REGCV1", param_dict=params)
        created.append(idx)
    return created
