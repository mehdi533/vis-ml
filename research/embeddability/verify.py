"""Worst-case verification of a ReLU surrogate over an input box.

Complements the conformal (statistical) margin with a *deterministic* guarantee:
solve a small MILP that maximises / minimises a chosen surrogate output over the
entire input box, using the exact ReLU big-M encoding with per-neuron bounds from
interval propagation. The returned interval provably contains every output the
surrogate can produce on the box -- so if the certified maximum stays within the
security limit, the surrogate can never predict a violation there (Venzke &
Chatzivasileiadis-style verification).

Pairs with `research.conformal`: conformal bounds the surrogate-vs-truth gap
statistically; this bounds the surrogate's own output deterministically.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from research.embeddability.bounds import LinearLayer, propagate_interval_bounds


def certified_output_range(
    layers: Sequence[LinearLayer],
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    output_index: int = 0,
    solver: str = "SCIP",
) -> Tuple[float, float]:
    """Certified ``[min, max]`` of output ``output_index`` over the input box.

    Exact for the given big-M relaxation (bounds from interval propagation);
    the interval provably contains all achievable surrogate outputs on the box.
    """
    import cvxpy as cp

    x_lo = np.asarray(x_lo, float).ravel()
    x_hi = np.asarray(x_hi, float).ravel()
    pre = propagate_interval_bounds(layers, x_lo, x_hi)

    x = cp.Variable(x_lo.size)
    cons = [x >= x_lo, x <= x_hi]
    h_prev = x
    for k, layer in enumerate(layers):
        W = np.asarray(layer.W, float)
        b = np.asarray(layer.b, float)
        z = W @ h_prev + b
        if k == len(layers) - 1:            # linear output layer
            out_expr = z
            break
        z_lo, z_hi = pre[k]
        nk = W.shape[0]
        h = cp.Variable(nk)
        for j in range(nk):
            l, u = float(z_lo[j]), float(z_hi[j])
            if l >= 0:
                cons.append(h[j] == z[j])
            elif u <= 0:
                cons.append(h[j] == 0)
            else:
                a = cp.Variable(boolean=True)
                cons += [h[j] >= 0, h[j] >= z[j], h[j] <= u * a, h[j] <= z[j] - l * (1 - a)]
        h_prev = h

    lo = cp.Problem(cp.Minimize(out_expr[output_index]), cons)
    lo.solve(solver=solver)
    hi = cp.Problem(cp.Maximize(out_expr[output_index]), cons)
    hi.solve(solver=solver)
    return float(lo.value), float(hi.value)
