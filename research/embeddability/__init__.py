"""Embeddability tooling: tighten the ReLU->binary MILP encoding of a surrogate.

The size and solve time of the embedded MILP are governed by (i) how many ReLU
neurons need a binary (the *unstable* ones, whose pre-activation interval spans
zero) and (ii) the tightness of the big-M constants. Both are controlled by the
per-neuron pre-activation bounds.

This module provides the feasibility-based first step -- Interval Bound
Propagation (IBP) -- which is cheap, exact-as-a-relaxation, and already the
information the encoding in `scheduling/constraints_nn.py` relies on. It lets us
quantify, for a trained surrogate over a given input box, how many neurons are
provably stable (no binary needed) and how large the big-M constants must be.

Next step (documented, not yet implemented): optimization-based bound tightening
(OBBT), which solves a small LP/MILP per neuron to shrink these intervals
further -- the standard way to embed a *larger* surrogate at equal solve cost
(Grimstad & Andersson 2019; polyhedral-theory survey arXiv:2305.00241).
"""

from research.embeddability.bounds import (
    LinearLayer,
    ReluStability,
    propagate_interval_bounds,
    relu_stability,
)

__all__ = [
    "LinearLayer",
    "ReluStability",
    "propagate_interval_bounds",
    "relu_stability",
]
