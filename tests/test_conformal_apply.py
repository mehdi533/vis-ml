"""Tests for the replay-detail -> margins -> tightened-bounds adapter.

Uses a synthetic replay detail frame matching replay_validation's schema, so it
runs with no ANDES/CVXPY dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.conformal.apply import (
    build_tightened_bounds,
    infer_mode,
    margins_from_replay_detail,
    tightened_envelope,
)


def _synthetic_detail(n=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # rocof_COI: symmetric envelope |x| <= 1.0; replay slightly exceeds prediction.
    pred = rng.uniform(-0.8, 0.8, size=n)
    rep = pred + rng.normal(scale=0.05, size=n)
    for p, r in zip(pred, rep):
        rows.append({"metric_name": "rocof_COI", "predicted_value": p, "replayed_value": r,
                     "limit_low": -1.0, "limit_high": 1.0})
    # Delta_P_IBR_1: symmetric ±5.
    pred2 = rng.uniform(-4.0, 4.0, size=n)
    rep2 = pred2 + rng.normal(scale=0.2, size=n)
    for p, r in zip(pred2, rep2):
        rows.append({"metric_name": "Delta_P_IBR_1", "predicted_value": p, "replayed_value": r,
                     "limit_low": -5.0, "limit_high": 5.0})
    return pd.DataFrame(rows)


def test_infer_mode():
    assert infer_mode(-1.0, 1.0) == "abs"
    assert infer_mode(-np.inf, 1.0) == "upper"
    assert infer_mode(0.0, 2.0) == "upper"      # asymmetric
    assert infer_mode(-np.inf, np.inf) is None


def test_margins_from_replay_detail_roundtrip(tmp_path):
    df = _synthetic_detail()
    csv = tmp_path / "replay_detail.csv"
    df.to_csv(csv, index=False)
    cm = margins_from_replay_detail(csv, alpha=0.1)
    assert set(cm.margins) == {"rocof_COI", "Delta_P_IBR_1"}
    assert cm.modes["rocof_COI"] == "abs"
    assert np.isfinite(cm.margins["rocof_COI"]) and cm.margins["rocof_COI"] > 0


def test_tightened_envelope_pulls_in_symmetric():
    df = _synthetic_detail()
    cm = margins_from_replay_detail(df, alpha=0.1)
    lo, hi = tightened_envelope(cm, "rocof_COI", -1.0, 1.0)
    assert -1.0 < lo < 0 < hi < 1.0
    assert hi == pytest.approx(-lo)  # symmetric tightening


def test_build_tightened_bounds_aligns_and_preserves_uncalibrated():
    df = _synthetic_detail()
    cm = margins_from_replay_detail(df, alpha=0.1)
    y_names = ["rocof_COI", "dev_COI", "Delta_P_IBR_1"]
    y_min = [-1.0, -0.2, -5.0]
    y_max = [1.0, 0.2, 5.0]
    out = build_tightened_bounds(cm, y_names, y_min, y_max)
    # rocof_COI and Delta_P_IBR_1 tightened; dev_COI (no calibration data) unchanged.
    assert out["y_max"][0] < 1.0
    assert out["y_min"][0] > -1.0
    assert out["y_max"][1] == 0.2 and out["y_min"][1] == -0.2
    assert out["y_max"][2] < 5.0


def test_missing_columns_raise():
    bad = pd.DataFrame({"metric_name": ["x"], "predicted_value": [0.0]})
    with pytest.raises(ValueError):
        margins_from_replay_detail(bad)
