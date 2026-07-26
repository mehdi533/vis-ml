"""Solver setup and problem-size/statistics helpers for the scheduler.

Extracted verbatim from problem.py to shrink that module (see README refactor note).
"""

from __future__ import annotations

from typing import Any, Mapping

import cvxpy as cp
import numpy as np


def _problem_stats(prob: cp.Problem, solver_name: str) -> tuple[int, int | None]:
    metrics = prob.size_metrics
    n_constraints = int(metrics.num_scalar_eq_constr + metrics.num_scalar_leq_constr)

    nnz_total: int | None = None
    try:
        data, _, _ = prob.get_problem_data(solver_name)
        nnz = 0
        for value in data.values():
            if hasattr(value, "nnz"):
                nnz += int(value.nnz)
            elif isinstance(value, np.ndarray):
                nnz += int(np.count_nonzero(value))
        nnz_total = nnz
    except Exception:
        nnz_total = None
    return n_constraints, nnz_total

def _scalar_constraint_count(constraints: list[cp.Constraint]) -> int:
    total = 0
    for cons in constraints:
        size = getattr(cons, "size", None)
        if size is None:
            shape = getattr(cons, "shape", ())
            size = int(np.prod(shape)) if shape else 1
        total += int(size)
    return int(total)

def _variable_type_counts(prob: cp.Problem) -> dict[str, int]:
    n_binary = 0
    n_integer = 0
    n_total = 0
    for var in prob.variables():
        n = int(np.prod(var.shape))
        n_total += n
        attrs = getattr(var, "attributes", {})
        if bool(attrs.get("boolean", False)):
            n_binary += n
        elif bool(attrs.get("integer", False)):
            n_integer += n
    n_cont = n_total - n_binary - n_integer
    return {
        "n_variables_total": int(n_total),
        "n_variables_continuous": int(n_cont),
        "n_variables_binary": int(n_binary),
        "n_variables_integer_nonbinary": int(n_integer),
    }

def _build_solver_kwargs(cfg: Mapping[str, Any], solver_name: str, verbose: bool, reoptimize: bool) -> dict[str, Any]:
    solver_cfg = cfg.get("solver", {}) if isinstance(cfg, Mapping) else {}
    solver_name_u = str(solver_name).upper()
    kwargs: dict[str, Any] = {
        "solver": solver_name,
        "verbose": bool(verbose),
    }
    if solver_name_u == "GUROBI":
        kwargs["reoptimize"] = bool(reoptimize)

    extra = solver_cfg.get("extra_kwargs", {})
    if isinstance(extra, Mapping):
        kwargs.update(dict(extra))

    alias_map = {
        "time_limit": "TimeLimit",
        "mip_gap": "MIPGap",
        "threads": "Threads",
    }
    for key, target in alias_map.items():
        if key in solver_cfg and target not in kwargs:
            kwargs[target] = solver_cfg[key]

    # Keep GUROBI-specific knobs out of non-GUROBI solver calls (e.g. OSQP).
    if solver_name_u != "GUROBI":
        for bad_key in ("reoptimize", "TimeLimit", "MIPGap", "Threads"):
            kwargs.pop(bad_key, None)
    return kwargs
