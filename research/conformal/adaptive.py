"""Adaptive (conditional) conformal margins — Mondrian binning by a covariate.

The marginal margin in ``calibration.py`` uses one value per metric, which is
loose in easy regimes and can be optimistic in hard ones. A Mondrian conformal
margin partitions the calibration set by an operating-point covariate (load /
stress, or a stress proxy such as the predicted severity) and calibrates a
separate margin per bin. Each bin then enjoys the finite-sample coverage
guarantee *conditionally*, so the tightening adapts: small where the surrogate
is accurate, large only where it is not.

Reference (verify before formal use): Vovk, "Conditional validity of inductive
conformal predictors," ACML 2012 (Mondrian/label-conditional conformal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from research.conformal.calibration import conformal_margin, empirical_coverage


@dataclass
class MondrianConformal:
    """Per-bin conformal margins conditioned on a covariate."""

    alpha: float = 0.1
    n_bins: int = 3
    mode: str = "abs"
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    margins: Dict[int, float] = field(default_factory=dict)

    def fit(self, predicted, replayed, covariate) -> "MondrianConformal":
        predicted = np.asarray(predicted, float).ravel()
        replayed = np.asarray(replayed, float).ravel()
        covariate = np.asarray(covariate, float).ravel()
        # Interior bin edges from covariate quantiles; open on both ends.
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
        interior = np.quantile(covariate, qs) if len(qs) else np.array([])
        self.edges = np.concatenate([[-np.inf], interior, [np.inf]])
        bin_id = np.clip(np.digitize(covariate, self.edges[1:-1]), 0, self.n_bins - 1)
        for b in range(self.n_bins):
            sel = bin_id == b
            if sel.sum() == 0:
                self.margins[b] = float("inf")
            else:
                self.margins[b] = conformal_margin(predicted[sel], replayed[sel], self.alpha, self.mode)
        return self

    def bin_of(self, covariate) -> np.ndarray:
        covariate = np.asarray(covariate, float).ravel()
        return np.clip(np.digitize(covariate, self.edges[1:-1]), 0, self.n_bins - 1)

    def margin_for(self, covariate) -> np.ndarray:
        """Per-sample margin: the calibrated margin of each sample's bin."""
        bins = self.bin_of(covariate)
        return np.array([self.margins.get(int(b), np.inf) for b in bins])


def conditional_coverage(predicted, replayed, covariate, margins_per_sample, mode="abs") -> float:
    """Coverage using a per-sample (possibly varying) margin."""
    predicted = np.asarray(predicted, float).ravel()
    replayed = np.asarray(replayed, float).ravel()
    m = np.asarray(margins_per_sample, float).ravel()
    if mode == "abs":
        scores = np.abs(replayed) - np.abs(predicted)
    else:
        scores = replayed - predicted
    ok = scores <= m
    return float(np.mean(ok))


def marginal_vs_mondrian(predicted, replayed, covariate, alpha=0.1, n_bins=3, mode="abs", seed=0):
    """Compare a single marginal margin vs Mondrian, on held-out coverage per bin.

    Returns per-bin coverage under the marginal margin vs the Mondrian margin,
    plus the margins. Demonstrates that the marginal margin over/under-covers
    across regimes while Mondrian keeps each bin near the 1-alpha target.
    """
    predicted = np.asarray(predicted, float).ravel()
    replayed = np.asarray(replayed, float).ravel()
    covariate = np.asarray(covariate, float).ravel()
    rng = np.random.default_rng(seed)
    n = len(predicted)
    perm = rng.permutation(n)
    half = n // 2
    cal, val = perm[:half], perm[half:]

    marg = conformal_margin(predicted[cal], replayed[cal], alpha, mode)
    mon = MondrianConformal(alpha=alpha, n_bins=n_bins, mode=mode).fit(
        predicted[cal], replayed[cal], covariate[cal])

    val_bins = mon.bin_of(covariate[val])
    out = {"marginal_margin": marg, "bins": []}
    for b in range(n_bins):
        sel = val_bins == b
        if sel.sum() == 0:
            continue
        p, r = predicted[val][sel], replayed[val][sel]
        cov_marg = empirical_coverage(p, r, marg, mode)
        cov_mon = empirical_coverage(p, r, mon.margins[b], mode)
        out["bins"].append({
            "bin": b, "n": int(sel.sum()),
            "mondrian_margin": round(float(mon.margins[b]), 4),
            "coverage_marginal": round(cov_marg, 3),
            "coverage_mondrian": round(cov_mon, 3),
        })
    return out
