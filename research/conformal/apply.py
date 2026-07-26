"""Adapter: replay-validation outputs -> conformal margins -> tightened bounds.

Reads the per-metric detail CSV written by ``scheduling/replay_validation.py``
(columns ``metric_name, predicted_value, replayed_value, limit_low,
limit_high``), calibrates a conformal margin per metric, and produces a
tightened dynamic-security envelope the optimizer can consume via
``bounds.y_min`` / ``bounds.y_max``.

Pipeline role
-------------
    optimize -> replay (ANDES) -> [this module] -> tightened bounds -> re-optimize

By tightening ``|predicted| <= L - margin`` the *replayed* metric is kept inside
``L`` with probability >= 1 - alpha (see ``calibration`` module), directly
targeting the thesis Ch. 6.2 replay gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from research.conformal.calibration import ConformalMargins, conformal_margin

# Relative tolerance for deciding a metric's limits are a symmetric envelope.
_SYMMETRY_RTOL = 1e-6


def infer_mode(limit_low: float, limit_high: float) -> Optional[str]:
    """Infer the conformal score mode from a metric's limits.

    Returns ``"abs"`` for a symmetric envelope (``limit_low ≈ -limit_high``),
    ``"upper"`` when only an upper limit is finite, or ``None`` when no usable
    limit is present (metric skipped).
    """
    lo_fin, hi_fin = np.isfinite(limit_low), np.isfinite(limit_high)
    if lo_fin and hi_fin:
        scale = max(abs(limit_high), abs(limit_low), 1e-12)
        if abs(limit_low + limit_high) <= _SYMMETRY_RTOL * scale:
            return "abs"
        return "upper"  # asymmetric -> conservative upper-side calibration
    if hi_fin:
        return "upper"
    return None


def margins_from_replay_detail(
    detail: "str | Path | pd.DataFrame",
    alpha: float = 0.05,
    metrics: Optional[list[str]] = None,
) -> ConformalMargins:
    """Calibrate conformal margins per metric from a replay detail table.

    Parameters
    ----------
    detail : path or DataFrame
        Replay detail CSV (or loaded frame) from ``replay_validation``.
    alpha : float
        Target miscoverage; replayed metric stays inside its limit w.p. >= 1-alpha.
    metrics : list of str, optional
        Restrict calibration to these ``metric_name`` values.
    """
    df = detail if isinstance(detail, pd.DataFrame) else pd.read_csv(detail)
    required = {"metric_name", "predicted_value", "replayed_value", "limit_low", "limit_high"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"replay detail is missing required columns: {sorted(missing)}")

    cm = ConformalMargins(alpha=alpha)
    names = metrics if metrics is not None else list(dict.fromkeys(df["metric_name"].tolist()))
    for name in names:
        sub = df.loc[df["metric_name"] == name]
        if sub.empty:
            continue
        lo = float(sub["limit_low"].iloc[0])
        hi = float(sub["limit_high"].iloc[0])
        mode = infer_mode(lo, hi)
        if mode is None:
            continue  # metric has no finite security limit -> nothing to tighten
        cm.margins[name] = conformal_margin(
            sub["predicted_value"].to_numpy(dtype=float),
            sub["replayed_value"].to_numpy(dtype=float),
            alpha=alpha,
            mode=mode,
        )
        cm.modes[name] = mode
    return cm


def tightened_envelope(
    margins: ConformalMargins,
    metric: str,
    limit_low: float,
    limit_high: float,
) -> tuple[float, float]:
    """Return ``(low', high')`` for a metric after applying its conformal margin.

    - ``abs`` mode: pull both sides in by the margin -> ``(low + m, high - m)``.
    - ``upper`` mode: pull only the upper limit in -> ``(low, high - m)``.
    """
    m = margins.margins.get(metric)
    mode = margins.modes.get(metric)
    if m is None or mode is None:
        return (limit_low, limit_high)
    if not np.isfinite(m):
        raise ValueError(
            f"Margin for {metric!r} is not finite (too few calibration samples)."
        )
    if mode == "abs":
        return (limit_low + m, limit_high - m)
    return (limit_low, limit_high - m)


def build_tightened_bounds(
    margins: ConformalMargins,
    y_names: list[str],
    y_min: list[float],
    y_max: list[float],
) -> Dict[str, list[float]]:
    """Produce tightened ``y_min``/``y_max`` lists aligned to ``y_names``.

    Metrics without a calibrated margin keep their original bounds. Returns a
    dict ready to splice into an optimization config's ``bounds`` block.
    """
    if not (len(y_names) == len(y_min) == len(y_max)):
        raise ValueError("y_names, y_min, y_max must have equal length.")
    new_lo, new_hi = list(map(float, y_min)), list(map(float, y_max))
    for i, name in enumerate(y_names):
        lo, hi = tightened_envelope(margins, name, new_lo[i], new_hi[i])
        new_lo[i], new_hi[i] = lo, hi
    return {"y_min": new_lo, "y_max": new_hi}
