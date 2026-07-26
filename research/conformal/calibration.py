"""Split-conformal one-sided margins.

Given calibration residuals between a surrogate's *predicted* security metric
and the *replayed* (ground-truth ANDES) metric, compute a margin that, added to
the predicted value, upper-bounds the replayed value with a finite-sample
coverage guarantee.

Scoring conventions
-------------------
Let ``predicted`` be the surrogate output and ``replayed`` the ANDES-replay
ground truth for the same schedule.

- ``mode="upper"`` (one-directional metric with an upper limit ``value <= L``):
  nonconformity score ``s = replayed - predicted``. The margin ``m`` satisfies
  ``P(replayed - predicted <= m) >= 1 - alpha``. Tightening the embedded bound
  to ``predicted <= L - m`` then makes ``replayed <= L`` hold w.p. >= 1 - alpha.

- ``mode="abs"`` (symmetric magnitude envelope ``|value| <= L``, e.g. RoCoF,
  COI frequency deviation): score ``s = |replayed| - |predicted|``. Tightening
  ``|predicted| <= L - m`` makes ``|replayed| <= L`` hold w.p. >= 1 - alpha,
  since ``|replayed| = |predicted| + s <= (L - m) + m``.

Both guarantees are *marginal* over an exchangeable calibration+test draw and
require no distributional assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np

_VALID_MODES = ("upper", "abs")


def min_calibration_size(alpha: float) -> int:
    """Smallest calibration set size for which a finite margin exists at ``alpha``.

    The conformal rank is ``k = ceil((n + 1) * (1 - alpha))``; a finite margin
    requires ``k <= n``, i.e. ``n >= ceil(1 / alpha) - 1``. Below this the margin
    is ``+inf`` (the guarantee cannot be met with so few points).
    """
    _check_alpha(alpha)
    return int(np.ceil(1.0 / alpha)) - 1


def conformal_margin(
    predicted: np.ndarray,
    replayed: np.ndarray,
    alpha: float = 0.05,
    mode: str = "abs",
) -> float:
    """Split-conformal one-sided margin at miscoverage level ``alpha``.

    Parameters
    ----------
    predicted, replayed : array-like
        Paired surrogate predictions and ANDES-replay ground-truth values.
    alpha : float in (0, 1)
        Target miscoverage; coverage is ``>= 1 - alpha``.
    mode : {"abs", "upper"}
        Nonconformity score convention (see module docstring).

    Returns
    -------
    float
        The margin ``m``. ``+inf`` if the calibration set is too small for the
        requested ``alpha`` (see :func:`min_calibration_size`).
    """
    _check_alpha(alpha)
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}.")

    pred = np.asarray(predicted, dtype=float).ravel()
    rep = np.asarray(replayed, dtype=float).ravel()
    if pred.shape != rep.shape:
        raise ValueError(
            f"predicted and replayed must have equal length; "
            f"got {pred.shape} and {rep.shape}."
        )

    if mode == "abs":
        scores = np.abs(rep) - np.abs(pred)
    else:  # "upper"
        scores = rep - pred

    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        raise ValueError("No finite residuals available for calibration.")

    # Conformal rank: k-th smallest score (1-indexed).
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def empirical_coverage(
    predicted: np.ndarray,
    replayed: np.ndarray,
    margin: float,
    mode: str = "abs",
) -> float:
    """Fraction of samples whose replayed value is covered by ``predicted + margin``.

    For ``mode="abs"`` this is ``mean(|replayed| <= |predicted| + margin)``;
    for ``mode="upper"`` it is ``mean(replayed <= predicted + margin)``.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}.")
    pred = np.asarray(predicted, dtype=float).ravel()
    rep = np.asarray(replayed, dtype=float).ravel()
    if mode == "abs":
        scores = np.abs(rep) - np.abs(pred)
    else:
        scores = rep - pred
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores <= margin))


@dataclass
class ConformalMargins:
    """Per-metric conformal margins fit from paired predicted/replayed columns.

    Example
    -------
    >>> cm = ConformalMargins(alpha=0.05)
    >>> cm.fit({"rocof_COI": ("abs", pred_rocof, rep_rocof)})
    >>> cm.margins["rocof_COI"]           # tighten |RoCoF| bound by this
    >>> cm.tighten_symmetric_bound("rocof_COI", limit=1.0)   # -> 1.0 - margin
    """

    alpha: float = 0.05
    margins: Dict[str, float] = field(default_factory=dict)
    modes: Dict[str, str] = field(default_factory=dict)

    def fit(self, columns: Mapping[str, tuple]) -> "ConformalMargins":
        """Fit margins for each metric.

        ``columns`` maps a metric name to ``(mode, predicted, replayed)``.
        """
        _check_alpha(self.alpha)
        for name, spec in columns.items():
            mode, predicted, replayed = spec
            self.margins[name] = conformal_margin(
                predicted, replayed, alpha=self.alpha, mode=mode
            )
            self.modes[name] = mode
        return self

    def tighten_symmetric_bound(self, name: str, limit: float) -> float:
        """Return the tightened magnitude limit ``limit - margin`` for a metric.

        Raises if the margin is not finite (insufficient calibration data).
        """
        m = self.margins[name]
        if not np.isfinite(m):
            raise ValueError(
                f"Margin for {name!r} is not finite; need at least "
                f"{min_calibration_size(self.alpha)} calibration samples."
            )
        return float(limit - m)


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
