"""Lightweight validation for scheduling optimization configs.

Catches the common mistakes (missing sections, mismatched bound/output lengths,
bad solver name) with clear, aggregated error messages before a run spends
minutes in ANDES/CVXPY setup. Deliberately lenient: it checks the critical
contract, not every optional key, so it never rejects a valid config.
"""

from __future__ import annotations

from typing import Any, List, Mapping

_KNOWN_SOLVERS = {"SCIP", "GUROBI", "OSQP", "CLARABEL", "SCS", "CBC", "GLPK_MI", "MOSEK"}


def validate_optimization_config(cfg: Mapping[str, Any]) -> None:
    """Raise ValueError (with all problems listed) if the config is malformed."""
    errs: List[str] = []

    def req(section: str, key: str | None = None):
        if section not in cfg or cfg[section] is None:
            errs.append(f"missing top-level section '{section}'")
            return None
        node = cfg[section]
        if key is not None:
            if not isinstance(node, Mapping) or key not in node or node[key] is None:
                errs.append(f"missing '{section}.{key}'")
                return None
            return node[key]
        return node

    req("system", "case")
    y_names = req("outputs", "y_names")
    bounds = req("bounds")
    if isinstance(bounds, Mapping):
        for k in ("y_min", "y_max"):
            v = bounds.get(k)
            if not isinstance(v, list):
                errs.append(f"bounds.{k} must be a list")
            elif isinstance(y_names, list) and len(v) != len(y_names):
                errs.append(f"bounds.{k} length {len(v)} != len(outputs.y_names) {len(y_names)}")
        for k in ("M_bounds", "D_bounds"):
            v = bounds.get(k)
            if not (isinstance(v, list) and len(v) == 2):
                errs.append(f"bounds.{k} must be a 2-element [lo, hi] list")

    constraints = req("constraints")
    if isinstance(constraints, Mapping):
        for sw in ("use_nn", "use_ed", "use_line"):
            if sw in constraints and not isinstance(constraints[sw], bool):
                errs.append(f"constraints.{sw} must be a boolean")

    name = req("solver", "name")
    if isinstance(name, str) and name.upper() not in _KNOWN_SOLVERS:
        errs.append(f"solver.name '{name}' not in known solvers {sorted(_KNOWN_SOLVERS)}")

    model = req("model")
    if isinstance(model, Mapping) and not (model.get("model_dir") or model.get("type")):
        errs.append("model needs 'model_dir' or 'type'")

    if errs:
        raise ValueError("Invalid optimization config:\n  - " + "\n  - ".join(errs))
