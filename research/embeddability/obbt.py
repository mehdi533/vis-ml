"""Optimization-based bound tightening (OBBT) for ReLU MLP embeddings.

Interval bound propagation (`bounds.py`) is cheap but loose: it ignores
correlations between neurons, so it over-counts how many ReLUs are 'unstable'
(need a binary) and inflates the big-M constants. OBBT tightens each neuron's
pre-activation bounds by solving a small LP over the network's convex (triangle)
relaxation of all preceding layers, using the input box as the only hard
constraint. Tighter bounds => fewer binaries and smaller big-M in the exact
encoding, which is how a *larger* surrogate embeds at the same solve cost.

For each hidden ReLU y = relu(z) with z in [l, u] the triangle relaxation is
    y >= 0,  y >= z,  y <= u (z - l) / (u - l).
OBBT for a neuron minimises / maximises its pre-activation subject to the input
box plus the triangle relaxations of earlier layers (with their already-tightened
bounds), layer by layer.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from research.embeddability.bounds import (
    LinearLayer,
    ReluStability,
    propagate_interval_bounds,
    relu_stability,
)


def obbt_bounds(
    layers: Sequence[LinearLayer],
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    solver: str = "CLARABEL",
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return OBBT-tightened per-layer pre-activation bounds ``(z_lo, z_hi)``.

    Falls back to the interval bound for any neuron whose LP does not solve.
    """
    import cvxpy as cp

    x_lo = np.asarray(x_lo, float).ravel()
    x_hi = np.asarray(x_hi, float).ravel()

    # Interval bounds as initialisation / fallback.
    ibp = propagate_interval_bounds(layers, x_lo, x_hi)
    out: List[Tuple[np.ndarray, np.ndarray]] = []

    x = cp.Variable(x_lo.size)
    cons = [x >= x_lo, x <= x_hi]
    h_prev = x

    for k, layer in enumerate(layers):
        W = np.asarray(layer.W, float)
        b = np.asarray(layer.b, float)
        z_expr = W @ h_prev + b
        nk = W.shape[0]
        z_lo = ibp[k][0].copy()
        z_hi = ibp[k][1].copy()

        for j in range(nk):
            try:
                pmin = cp.Problem(cp.Minimize(z_expr[j]), cons)
                pmin.solve(solver=solver)
                if pmin.status in ("optimal", "optimal_inaccurate") and pmin.value is not None:
                    z_lo[j] = max(z_lo[j], float(pmin.value))
                pmax = cp.Problem(cp.Maximize(z_expr[j]), cons)
                pmax.solve(solver=solver)
                if pmax.status in ("optimal", "optimal_inaccurate") and pmax.value is not None:
                    z_hi[j] = min(z_hi[j], float(pmax.value))
            except Exception:
                pass  # keep IBP bound for this neuron
        # Guard against tiny numerical inversions.
        bad = z_lo > z_hi
        if np.any(bad):
            mid = 0.5 * (z_lo[bad] + z_hi[bad])
            z_lo[bad] = mid
            z_hi[bad] = mid
        out.append((z_lo, z_hi))

        if k < len(layers) - 1:  # add this layer's post-activation + triangle relaxation
            h = cp.Variable(nk)
            for j in range(nk):
                l, u = float(z_lo[j]), float(z_hi[j])
                if l >= 0:
                    cons.append(h[j] == z_expr[j])
                elif u <= 0:
                    cons.append(h[j] == 0)
                else:
                    cons += [h[j] >= 0, h[j] >= z_expr[j], h[j] <= (u / (u - l)) * (z_expr[j] - l)]
            h_prev = h

    return out


def compare_ibp_obbt(
    layers: Sequence[LinearLayer],
    x_lo: np.ndarray,
    x_hi: np.ndarray,
) -> dict:
    """Summarise the binary-count / big-M reduction of OBBT vs IBP (hidden layers)."""
    ibp = relu_stability(propagate_interval_bounds(layers, x_lo, x_hi))
    obbt = relu_stability(obbt_bounds(layers, x_lo, x_hi))
    ibp_bin = sum(s.n_unstable for s in ibp)
    obbt_bin = sum(s.n_unstable for s in obbt)
    ibp_M = max((s.max_abs_bigM for s in ibp), default=0.0)
    obbt_M = max((s.max_abs_bigM for s in obbt), default=0.0)
    return {
        "ibp_binaries": ibp_bin,
        "obbt_binaries": obbt_bin,
        "binary_reduction_pct": round(100.0 * (ibp_bin - obbt_bin) / ibp_bin, 1) if ibp_bin else 0.0,
        "ibp_max_bigM": round(ibp_M, 3),
        "obbt_max_bigM": round(obbt_M, 3),
    }
