"""Split-conformal robust margins for surrogate-embedded VIS scheduling.

Motivation (thesis Ch. 6.2): the embedded surrogate can satisfy the dynamic
security limits at the optimum while the ANDES *replay* of the same schedule
marginally exceeds them (e.g. RoCoF ~1.007 vs 1.0). This module closes that gap
with a distribution-free, finite-sample guarantee: calibrate a one-sided margin
on predicted-vs-replayed residuals, then tighten the embedded security bound by
that margin so the replayed metric stays inside the limit with probability
>= 1 - alpha.

The calibration core is pure statistics (numpy only) and is fully unit-tested
independently of ANDES/CVXPY. Adapters to `replay_validation` outputs and the
optimizer bound-tightening hook live in `apply.py`.

References (verify before formal citation):
- Vovk, Gammerman & Shafer, Algorithmic Learning in a Random World, 2005.
- Lei, G'Sell, Rinaldo, Tibshirani & Wasserman, Distribution-Free Predictive
  Inference for Regression, JASA 2018.
"""

from research.conformal.adaptive import (
    MondrianConformal,
    conditional_coverage,
    marginal_vs_mondrian,
)
from research.conformal.calibration import (
    ConformalMargins,
    conformal_margin,
    empirical_coverage,
    min_calibration_size,
)

__all__ = [
    "ConformalMargins",
    "MondrianConformal",
    "conditional_coverage",
    "conformal_margin",
    "empirical_coverage",
    "marginal_vs_mondrian",
    "min_calibration_size",
]
