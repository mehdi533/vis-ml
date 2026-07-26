"""Interval Bound Propagation (IBP) for ReLU MLP surrogates.

Given an input box ``[x_lo, x_hi]``, propagate per-neuron pre-activation bounds
``[z_lo, z_hi]`` through affine + ReLU layers. For an affine map ``z = W h + b``
with ``h in [h_lo, h_hi]`` the tightest interval image is

    z_lo = W+ @ h_lo + W- @ h_hi + b
    z_hi = W+ @ h_hi + W- @ h_lo + b

with ``W+ = max(W, 0)`` and ``W- = min(W, 0)``; ReLU maps ``[z_lo, z_hi]`` to
``[max(z_lo, 0), max(z_hi, 0)]``.

A ReLU neuron needs a binary in the MILP encoding only when its interval spans
zero (``z_lo < 0 < z_hi``); otherwise it is stably active (``z_lo >= 0``) or
stably inactive (``z_hi <= 0``) and can be fixed. The interval endpoints are
exactly the big-M constants used by the exact encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LinearLayer:
    """A single affine layer ``z = W @ h + b`` (``W`` shape ``(out, in)``)."""

    W: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        W = np.asarray(self.W, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if W.ndim != 2:
            raise ValueError(f"W must be 2D (out, in); got shape {W.shape}.")
        if b.shape != (W.shape[0],):
            raise ValueError(f"b must have shape ({W.shape[0]},); got {b.shape}.")
        object.__setattr__(self, "W", W)
        object.__setattr__(self, "b", b)


@dataclass(frozen=True)
class ReluStability:
    """Per-layer ReLU stability counts from propagated bounds."""

    n_total: int
    n_active: int      # z_lo >= 0  (relu == identity, no binary)
    n_inactive: int    # z_hi <= 0  (relu == 0, no binary)
    n_unstable: int    # z_lo < 0 < z_hi  (needs a binary)
    max_abs_bigM: float

    @property
    def binary_fraction(self) -> float:
        return self.n_unstable / self.n_total if self.n_total else 0.0


def propagate_interval_bounds(
    layers: Sequence[LinearLayer],
    x_lo: np.ndarray,
    x_hi: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Propagate an input box through ``layers`` (affine + ReLU between them).

    ReLU is applied after every layer except the last (a standard MLP with a
    linear output head). Returns the list of pre-activation ``(z_lo, z_hi)``
    intervals, one per linear layer.
    """
    x_lo = np.asarray(x_lo, dtype=float).ravel()
    x_hi = np.asarray(x_hi, dtype=float).ravel()
    if x_lo.shape != x_hi.shape:
        raise ValueError("x_lo and x_hi must have the same shape.")
    if np.any(x_lo > x_hi):
        raise ValueError("x_lo must be elementwise <= x_hi.")

    h_lo, h_hi = x_lo, x_hi
    preacts: List[Tuple[np.ndarray, np.ndarray]] = []
    for k, layer in enumerate(layers):
        Wp = np.maximum(layer.W, 0.0)
        Wm = np.minimum(layer.W, 0.0)
        z_lo = Wp @ h_lo + Wm @ h_hi + layer.b
        z_hi = Wp @ h_hi + Wm @ h_lo + layer.b
        preacts.append((z_lo, z_hi))
        if k < len(layers) - 1:  # ReLU on hidden layers only
            h_lo, h_hi = np.maximum(z_lo, 0.0), np.maximum(z_hi, 0.0)
    return preacts


def relu_stability(
    preacts: Sequence[Tuple[np.ndarray, np.ndarray]],
    include_output_layer: bool = False,
) -> List[ReluStability]:
    """Summarise ReLU stability from propagated pre-activation bounds.

    By default the final (output) layer is excluded, since it carries no ReLU.
    """
    end = len(preacts) if include_output_layer else max(len(preacts) - 1, 0)
    out: List[ReluStability] = []
    for z_lo, z_hi in preacts[:end]:
        active = int(np.sum(z_lo >= 0))
        inactive = int(np.sum(z_hi <= 0))
        total = int(z_lo.size)
        unstable = total - active - inactive
        max_bigM = float(np.max(np.abs(np.concatenate([z_lo, z_hi])))) if total else 0.0
        out.append(
            ReluStability(
                n_total=total,
                n_active=active,
                n_inactive=inactive,
                n_unstable=unstable,
                max_abs_bigM=max_bigM,
            )
        )
    return out
