"""Tests for adaptive (Mondrian) conformal margins."""

from __future__ import annotations

import numpy as np

from research.conformal.adaptive import MondrianConformal, marginal_vs_mondrian


def _heteroscedastic(n=4000, seed=0):
    """Error magnitude grows with the covariate -> adaptivity should matter."""
    rng = np.random.default_rng(seed)
    cov = rng.uniform(0.0, 1.0, size=n)
    pred = np.zeros(n)
    replayed = rng.normal(scale=0.05 + 0.5 * cov)  # noise scale rises with cov
    return pred, replayed, cov


def test_mondrian_margins_increase_with_covariate():
    pred, rep, cov = _heteroscedastic()
    mc = MondrianConformal(alpha=0.1, n_bins=3, mode="abs").fit(pred, rep, cov)
    m0, m1, m2 = mc.margins[0], mc.margins[1], mc.margins[2]
    assert m0 < m1 < m2  # harder regimes need larger margins


def test_mondrian_keeps_per_bin_coverage():
    pred, rep, cov = _heteroscedastic(seed=1)
    res = marginal_vs_mondrian(pred, rep, cov, alpha=0.1, n_bins=3, mode="abs")
    # Mondrian should hold each bin near 0.9; the single marginal margin should
    # under-cover the hardest bin (bin 2).
    by_bin = {b["bin"]: b for b in res["bins"]}
    assert by_bin[2]["coverage_mondrian"] >= 0.85
    assert by_bin[2]["coverage_marginal"] < by_bin[2]["coverage_mondrian"] + 1e-9
    # and the marginal margin over-covers the easy bin (wasteful)
    assert by_bin[0]["coverage_marginal"] >= by_bin[0]["coverage_mondrian"] - 1e-9


def test_margin_for_assigns_bins():
    pred, rep, cov = _heteroscedastic(seed=2)
    mc = MondrianConformal(alpha=0.1, n_bins=3).fit(pred, rep, cov)
    m = mc.margin_for([0.05, 0.95])  # low-stress, high-stress
    assert m[0] <= m[1]
