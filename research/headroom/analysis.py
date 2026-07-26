"""Headroom / reserve reallocation metrics for targeted-vs-uniform VIS.

All functions operate on tidy per-unit tables with (at least) the columns:
``unit_id``, ``unit_type`` ("SG" or "IBR"), and the quantities being compared
(e.g. ``headroom_up``, ``reserve_up``, ``M``, ``D``). This mirrors the per-unit
rows in scheduling/problem.py's dispatch-impact export while staying decoupled
from its exact column names, so the metrics are unit-testable in isolation.

Sign conventions are stated explicitly: a *positive* "freed" value means the
targeted schedule leaves more upward headroom available than the uniform
baseline. Nothing here asserts an energy saving -- the thesis §6.2 flags the
headroom as an inexact proxy, so we report reallocation, not certified savings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


def total_headroom(schedule: pd.DataFrame, column: str = "headroom_up") -> float:
    """Total headroom (or any additive reserve column) across all units."""
    if column not in schedule.columns:
        raise KeyError(f"schedule is missing column {column!r}.")
    return float(np.nansum(schedule[column].to_numpy(dtype=float)))


def headroom_freed(
    baseline: pd.DataFrame,
    targeted: pd.DataFrame,
    column: str = "headroom_up",
) -> float:
    """Headroom freed by the targeted schedule vs. the uniform baseline.

    Positive => the targeted (surrogate) schedule leaves more upward headroom
    available than the uniform (M, D) baseline.
    """
    return total_headroom(targeted, column) - total_headroom(baseline, column)


def reserve_by_class(
    schedule: pd.DataFrame,
    column: str = "reserve_up",
    type_column: str = "unit_type",
) -> dict[str, float]:
    """Sum a reserve column split by unit class (SG vs IBR)."""
    for c in (column, type_column):
        if c not in schedule.columns:
            raise KeyError(f"schedule is missing column {c!r}.")
    out: dict[str, float] = {}
    for cls, grp in schedule.groupby(type_column):
        out[str(cls)] = float(np.nansum(grp[column].to_numpy(dtype=float)))
    return out


def reserve_shift(
    baseline: pd.DataFrame,
    targeted: pd.DataFrame,
    column: str = "reserve_up",
    type_column: str = "unit_type",
) -> dict[str, float]:
    """Change in reserve by class, and the net SG->IBR shift.

    ``sg_to_ibr_shift`` > 0 means reserve moved off synchronous generators onto
    IBRs (the efficiency signal: conventional headroom freed for clean capacity).
    """
    b = reserve_by_class(baseline, column, type_column)
    t = reserve_by_class(targeted, column, type_column)
    d_sg = t.get("SG", 0.0) - b.get("SG", 0.0)
    d_ibr = t.get("IBR", 0.0) - b.get("IBR", 0.0)
    return {
        "delta_reserve_sg": d_sg,
        "delta_reserve_ibr": d_ibr,
        "sg_to_ibr_shift": -d_sg,  # reserve removed from SG (positive == freed)
    }


def allocation_nonuniformity(values: np.ndarray) -> dict[str, float]:
    """How non-uniform an M or D allocation is across controllable units.

    Returns the spread (max-min), coefficient of variation, and a normalised
    range. A uniform baseline yields ~0 everywhere; a targeted schedule does not.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"spread": 0.0, "cv": 0.0, "range_over_mean": 0.0}
    mean = float(np.mean(v))
    spread = float(np.max(v) - np.min(v))
    std = float(np.std(v))
    cv = std / mean if mean != 0 else 0.0
    return {
        "spread": spread,
        "cv": float(cv),
        "range_over_mean": float(spread / mean) if mean != 0 else 0.0,
    }


@dataclass
class HeadroomReport:
    """Bundle of targeted-vs-uniform reallocation metrics."""

    headroom_freed: float
    reserve_shift: dict[str, float]
    m_nonuniformity: dict[str, float]
    d_nonuniformity: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "headroom_freed": self.headroom_freed,
            "reserve_shift": self.reserve_shift,
            "m_nonuniformity": self.m_nonuniformity,
            "d_nonuniformity": self.d_nonuniformity,
        }


def compare_schedules(
    baseline: pd.DataFrame,
    targeted: pd.DataFrame,
    *,
    headroom_column: str = "headroom_up",
    reserve_column: str = "reserve_up",
    type_column: str = "unit_type",
    m_column: str = "M",
    d_column: str = "D",
    ibr_only_for_md: bool = True,
) -> HeadroomReport:
    """Full targeted-vs-uniform comparison → :class:`HeadroomReport`.

    M/D non-uniformity is computed over IBR rows only by default (SGs have no
    virtual inertia/damping to schedule).
    """
    md_rows = targeted
    if ibr_only_for_md and type_column in targeted.columns:
        md_rows = targeted.loc[targeted[type_column] == "IBR"]
    m_vals = md_rows[m_column].to_numpy(dtype=float) if m_column in md_rows.columns else np.array([])
    d_vals = md_rows[d_column].to_numpy(dtype=float) if d_column in md_rows.columns else np.array([])
    return HeadroomReport(
        headroom_freed=headroom_freed(baseline, targeted, headroom_column),
        reserve_shift=reserve_shift(baseline, targeted, reserve_column, type_column),
        m_nonuniformity=allocation_nonuniformity(m_vals),
        d_nonuniformity=allocation_nonuniformity(d_vals),
    )
