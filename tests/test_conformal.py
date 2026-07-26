"""Correctness tests for the split-conformal margin core.

These validate the finite-sample coverage guarantee empirically (Monte Carlo)
and check the exact order-statistic behaviour -- no ANDES/CVXPY needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from research.conformal.calibration import (
    ConformalMargins,
    conformal_margin,
    empirical_coverage,
    min_calibration_size,
)


def test_exact_order_statistic():
    # scores = replayed - predicted = [1..10]; n=10, alpha=0.2
    # k = ceil((n+1)(1-alpha)) = ceil(11*0.8) = 9 -> 9th smallest = 9
    predicted = np.zeros(10)
    replayed = np.arange(1, 11, dtype=float)
    m = conformal_margin(predicted, replayed, alpha=0.2, mode="upper")
    assert m == pytest.approx(9.0)


def test_margin_monotonic_in_alpha():
    rng = np.random.default_rng(0)
    replayed = rng.normal(size=2000)
    predicted = np.zeros_like(replayed)
    m_strict = conformal_margin(predicted, replayed, alpha=0.01, mode="upper")
    m_loose = conformal_margin(predicted, replayed, alpha=0.20, mode="upper")
    # Tighter miscoverage (smaller alpha) => larger required margin.
    assert m_strict > m_loose


def test_infinite_margin_when_too_few_samples():
    # alpha=0.05 needs n >= ceil(1/0.05)-1 = 19 samples for a finite margin.
    assert min_calibration_size(0.05) == 19
    predicted = np.zeros(10)
    replayed = np.arange(10, dtype=float)
    assert conformal_margin(predicted, replayed, alpha=0.05, mode="upper") == float("inf")
    # One more than the threshold gives a finite margin.
    predicted = np.zeros(19)
    replayed = np.arange(19, dtype=float)
    assert np.isfinite(conformal_margin(predicted, replayed, alpha=0.05, mode="upper"))


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_marginal_coverage_guarantee_upper(alpha):
    """Average coverage across fresh splits must meet the 1-alpha guarantee."""
    rng = np.random.default_rng(42)
    n_cal, n_test, n_trials = 400, 400, 300
    covs = []
    for _ in range(n_trials):
        cal = rng.normal(size=n_cal)
        test = rng.normal(size=n_test)
        m = conformal_margin(np.zeros(n_cal), cal, alpha=alpha, mode="upper")
        covs.append(empirical_coverage(np.zeros(n_test), test, m, mode="upper"))
    mean_cov = float(np.mean(covs))
    # Conformal is marginally valid (slightly conservative): >= 1-alpha,
    # not wildly over-covering. Small MC tolerance on the lower side.
    assert mean_cov >= (1 - alpha) - 0.01
    assert mean_cov <= (1 - alpha) + 0.05


def test_abs_mode_covers_symmetric_envelope():
    """|replayed| <= |predicted| + margin should hold ~1-alpha of the time."""
    rng = np.random.default_rng(7)
    alpha = 0.1
    # Predicted magnitudes well away from zero so abs() doesn't flip sign.
    base = rng.uniform(2.0, 5.0, size=2000)
    noise_cal = rng.normal(scale=0.3, size=2000)
    noise_test = rng.normal(scale=0.3, size=2000)
    pred_cal, rep_cal = base, base + noise_cal
    pred_test, rep_test = base, base + noise_test
    m = conformal_margin(pred_cal, rep_cal, alpha=alpha, mode="abs")
    cov = empirical_coverage(pred_test, rep_test, m, mode="abs")
    assert cov >= (1 - alpha) - 0.03


def test_bound_tightening_helper():
    cm = ConformalMargins(alpha=0.1)
    # Replay exceeds prediction by ~0.05 on a |RoCoF| <= 1.0 metric.
    rng = np.random.default_rng(1)
    pred = rng.uniform(0.5, 0.9, size=500)
    rep = pred + rng.uniform(0.0, 0.08, size=500)
    cm.fit({"rocof_COI": ("abs", pred, rep)})
    tightened = cm.tighten_symmetric_bound("rocof_COI", limit=1.0)
    assert 0.90 <= tightened < 1.0  # bound pulled in by a positive margin


def test_input_validation():
    with pytest.raises(ValueError):
        conformal_margin([0, 1], [0, 1], alpha=0.0)
    with pytest.raises(ValueError):
        conformal_margin([0, 1], [0, 1], alpha=1.0)
    with pytest.raises(ValueError):
        conformal_margin([0, 1], [0, 1], mode="bogus")
    with pytest.raises(ValueError):
        conformal_margin([0, 1, 2], [0, 1], alpha=0.1)  # shape mismatch
