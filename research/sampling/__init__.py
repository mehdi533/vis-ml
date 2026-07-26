"""Boundary-focused active sampling for VIS surrogate data generation.

Directed-Walks-style sampling (Thams et al. 2020): instead of sampling the
operating/decision space uniformly, drive the *schedulable* inputs (M/D) toward
a target output level -- the security boundary -- using the trained surrogate's
gradient. Samples then cluster where the dispatch's active constraints live,
which is where surrogate accuracy matters most for the embedded MILP.

This operates in the surrogate's scaled input/output space; the caller maps a
physical security limit to a scaled target value.
"""

from research.sampling.directed_walks import boundary_samples, directed_walk

__all__ = ["boundary_samples", "directed_walk"]
