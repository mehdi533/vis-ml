"""Targeted-vs-uniform VIS allocation analysis (headroom & reserve).

Quantifies, honestly, the *structural* efficiency story of the thesis: the
optimizer breaks the uniform (M, D) baseline and reallocates inertia/damping and
reserve non-uniformly across converters. This module measures:

- headroom freed relative to a uniform baseline (a proxy quantity -- the thesis
  §6.2 notes the headroom proxy is not exact, so results are reported as
  reallocation, not as a certified energy saving);
- the synchronous-generator -> IBR reserve shift (the cleaner efficiency signal);
- the non-uniformity of the M/D allocation (how "targeted" the schedule is).

Reads the per-unit dispatch-impact schema written by scheduling/problem.py; the
core functions are schema-light and unit-tested on synthetic tables.
"""

from research.headroom.analysis import (
    HeadroomReport,
    allocation_nonuniformity,
    compare_schedules,
    headroom_freed,
    reserve_by_class,
    reserve_shift,
)

__all__ = [
    "HeadroomReport",
    "allocation_nonuniformity",
    "compare_schedules",
    "headroom_freed",
    "reserve_by_class",
    "reserve_shift",
]
