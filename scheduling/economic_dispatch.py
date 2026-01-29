import andes
import numpy as np
import cvxpy as cp
from typing import Optional, Sequence



def economic_dispatch_quad(
    Pd: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    Pg_min: np.ndarray,
    Pg_max: np.ndarray,
    *,
    extra_constraints: Optional[Sequence[cp.Constraint]] = None,
    solver: Optional[str] = "GUROBI",
    solver_kwargs: Optional[dict] = None,
):
    """
    Solve a standard quadratic economic dispatch problem:

        minimize   sum_i ( a_i + b_i * Pg_i + c_i * Pg_i^2 )
        s.t.       sum_i Pg_i = Pd
                   Pg_min_i <= Pg_i <= Pg_max_i

    Parameters
    ----------
    Pd : float
        Total active power demand (pu or MW, consistent with coefficients).
    a, b, c : np.ndarray
        Cost coefficients for each generator (same length).
    Pg_min, Pg_max : np.ndarray
        Min and max active power outputs for each generator.

    extra_constraints : Sequence[cp.Constraint], optional
        Additional convex (or MIP) constraints that may include extra variables.
        Use this to couple Pg to other models (e.g., NN constraints).
    solver : str, optional
        CVXPY solver name. Defaults to OSQP.
    solver_kwargs : dict, optional
        Extra keyword arguments passed to cvxpy.Problem.solve().

    Returns
    -------
    Pg_opt : np.ndarray
        Optimal generator outputs.
    total_cost : float
        Optimal total generation cost.
    lam : float or None
        Lagrange multiplier (system marginal cost) on the power balance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    Pg_min = np.asarray(Pg_min, dtype=float)
    Pg_max = np.asarray(Pg_max, dtype=float)

    ng = len(a)
    assert all(len(x) == ng for x in (b, c, Pg_min, Pg_max)), \
        "Cost and limit arrays must have the same length."

    # Decision variable
    Pg = cp.Variable(ng)

    # Objective
    cost_expr = a + cp.multiply(b, Pg) + cp.multiply(c, cp.square(Pg))
    objective = cp.Minimize(cp.sum(cost_expr))

    # Constraints
    constraints = [
        cp.sum(Pg) == Pd,
        Pg >= Pg_min,
        Pg <= Pg_max,
    ]
    if extra_constraints:
        constraints += list(extra_constraints)
    
    prob = cp.Problem(objective, constraints)
    solve_kwargs = solver_kwargs or {}
    if solver is None:
        prob.solve(**solve_kwargs)
    else:
        prob.solve(solver=solver, **solve_kwargs)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Economic dispatch failed with status: {prob.status}")

    Pg_opt = Pg.value
    total_cost = prob.value

    lam = None
    try:
        lam = -float(constraints[0].dual_value)
    except Exception:
        pass

    return Pg_opt, total_cost, lam


def ed_calculation(
    ss: andes.System,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    extra_constraints: Optional[Sequence[cp.Constraint]] = None,
    solver: Optional[str] = "GUROBI",
    solver_kwargs: Optional[dict] = None,
):
    """
    Solve a *lossless* quadratic ED using total PQ demand and apply
    the optimal Pg setpoints to Slack/PV generators *before* ss.setup().

    This allows dynamic models (TGOV1, GENROU, etc.) to be initialized
    consistently with the ED dispatch.

    Parameters
    ----------
    ss : andes.System (loaded with setup=False)
    a, b, c : np.ndarray
        Cost coefficients, aligned with ss.GENROU rows.
    Pg_min, Pg_max : np.ndarray
        Min/max Pg for each GENROU, same length as a,b,c.

    extra_constraints : Sequence[cp.Constraint], optional
        Additional convex (or MIP) constraints that may include extra variables.
        Use this to couple Pg to other models (e.g., NN constraints).
    solver : str, optional
        CVXPY solver name. Defaults to OSQP.
    solver_kwargs : dict, optional
        Extra keyword arguments passed to cvxpy.Problem.solve().

    Returns
    -------
    Pg_opt : np.ndarray
    total_cost : float
    lam : float
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    Pg_min, Pg_max = get_pg_limits_from_static(ss)

    ng = ss.PV.n + ss.Slack.n
    assert ng == len(a) == len(b) == len(c) == len(Pg_min) == len(Pg_max), \
        "Cost and limit arrays must match number of GENROU units."

    # Total active demand (lossless ED)
    Pd = float(np.sum(ss.PQ.p0.v))

    # Solve ED for ALL generators (including the one at the slack bus)
    Pg_opt, total_cost, lam = economic_dispatch_quad(
        Pd,
        a,
        b,
        c,
        Pg_min,
        Pg_max,
        extra_constraints=extra_constraints,
        solver=solver,
        solver_kwargs=solver_kwargs,
    )

    # Map GENROU units to static generators (Slack / PV) by bus
    gen_buses = np.asarray(ss.PV.bus.v, dtype=int)
    slack_map = {bus: i for i, bus in enumerate(ss.Slack.bus.v)} if ss.Slack.n > 0 else {}
    pv_map    = {bus: i for i, bus in enumerate(ss.PV.bus.v)}    if ss.PV.n > 0    else {}

    for Pg_star, bus in zip(Pg_opt, gen_buses):
        if bus in slack_map:
            j = slack_map[bus]
            ss.Slack.p0.v[j] = Pg_star
        elif bus in pv_map:
            j = pv_map[bus]
            ss.PV.p0.v[j] = Pg_star
        else:
            raise RuntimeError(
                f"No Slack or PV generator at bus {bus} for GENROU; cannot apply ED."
            )

    return Pg_opt, total_cost, lam


def get_pg_limits_from_static(ss):
    """
    Build Pg_min and Pg_max arrays aligned with ss.GENROU order,
    using PV.pmin/pmax and Slack.pmin/pmax.
    """
    Pg_min = [ss.PV.pmin.v[i] for i in range(ss.PV.n)]
    Pg_max = [ss.PV.pmax.v[i] for i in range(ss.PV.n)]
    
    Pg_min.extend([ss.Slack.pmin.v[i] for i in range(ss.Slack.n)])
    Pg_max.extend([ss.Slack.pmax.v[i] for i in range(ss.Slack.n)])

    return Pg_min, Pg_max
