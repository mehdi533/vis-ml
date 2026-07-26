"""Tests for the targeted-vs-uniform headroom/reserve analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.headroom.analysis import (
    allocation_nonuniformity,
    compare_schedules,
    headroom_freed,
    reserve_by_class,
    reserve_shift,
)


def _uniform_schedule():
    # 4 IBRs at uniform (M,D)=(4,2); 2 SGs. Uniform baseline leaves little IBR headroom.
    return pd.DataFrame({
        "unit_id": ["IBR1", "IBR2", "IBR3", "IBR4", "SG1", "SG2"],
        "unit_type": ["IBR", "IBR", "IBR", "IBR", "SG", "SG"],
        "M": [4.0, 4.0, 4.0, 4.0, np.nan, np.nan],
        "D": [2.0, 2.0, 2.0, 2.0, np.nan, np.nan],
        "headroom_up": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        "reserve_up": [0.5, 0.5, 0.5, 0.5, 3.0, 3.0],
    })


def _targeted_schedule():
    # Targeted: non-uniform M/D, more IBR headroom freed, reserve shifted SG->IBR.
    return pd.DataFrame({
        "unit_id": ["IBR1", "IBR2", "IBR3", "IBR4", "SG1", "SG2"],
        "unit_type": ["IBR", "IBR", "IBR", "IBR", "SG", "SG"],
        "M": [7.0, 1.0, 5.0, 2.0, np.nan, np.nan],
        "D": [3.5, 0.5, 2.5, 1.0, np.nan, np.nan],
        "headroom_up": [1.5, 1.5, 1.5, 1.5, 1.0, 1.0],
        "reserve_up": [1.5, 1.5, 1.5, 1.5, 1.0, 1.0],
    })


def test_headroom_freed_positive_when_targeted_leaves_more():
    freed = headroom_freed(_uniform_schedule(), _targeted_schedule())
    # targeted total headroom = 4*1.5 + 2*1.0 = 8.0; baseline = 4*1 + 2*2 = 8.0 -> 0
    assert freed == pytest.approx(0.0)
    # IBR-only headroom did rise (6.0 vs 4.0), which the reserve_shift captures.


def test_reserve_by_class_splits_correctly():
    r = reserve_by_class(_uniform_schedule())
    assert r["IBR"] == pytest.approx(2.0)   # 4 * 0.5
    assert r["SG"] == pytest.approx(6.0)    # 2 * 3.0


def test_reserve_shift_moves_off_sg():
    shift = reserve_shift(_uniform_schedule(), _targeted_schedule())
    # SG reserve: 6.0 -> 2.0 (down 4); IBR: 2.0 -> 6.0 (up 4)
    assert shift["delta_reserve_sg"] == pytest.approx(-4.0)
    assert shift["delta_reserve_ibr"] == pytest.approx(4.0)
    assert shift["sg_to_ibr_shift"] == pytest.approx(4.0)  # 4 units freed off SG


def test_nonuniformity_zero_for_uniform_positive_for_targeted():
    uni = allocation_nonuniformity([4.0, 4.0, 4.0, 4.0])
    tgt = allocation_nonuniformity([7.0, 1.0, 5.0, 2.0])
    assert uni["spread"] == pytest.approx(0.0)
    assert uni["cv"] == pytest.approx(0.0)
    assert tgt["spread"] > 0 and tgt["cv"] > 0


def test_compare_schedules_report():
    rep = compare_schedules(_uniform_schedule(), _targeted_schedule())
    d = rep.as_dict()
    assert d["reserve_shift"]["sg_to_ibr_shift"] == pytest.approx(4.0)
    # M allocation is uniform in baseline (cv 0) but targeted is not.
    assert d["m_nonuniformity"]["cv"] > 0
    assert d["d_nonuniformity"]["spread"] > 0


def test_missing_column_raises():
    with pytest.raises(KeyError):
        reserve_by_class(pd.DataFrame({"unit_type": ["IBR"]}))
