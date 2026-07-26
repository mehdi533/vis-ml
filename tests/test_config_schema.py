"""Tests for the optimization config validator + that committed configs pass."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scheduling.config_schema import validate_optimization_config

ROOT = Path(__file__).resolve().parents[1]
OPT_CONFIGS = [
    "configs/scheduling/base_optimization.yaml",
    "configs/scheduling/smoke/optimization_smoke.yaml",
    "configs/scheduling/ieee118/optimization_ieee118.yaml",
    "configs/scheduling/conformal/solvetime_base.yaml",
]


@pytest.mark.parametrize("rel", OPT_CONFIGS)
def test_committed_optimization_configs_validate(rel):
    cfg = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
    validate_optimization_config(cfg)  # must not raise


def test_validator_flags_mismatched_bounds():
    cfg = {
        "system": {"case": "x.xlsx"},
        "outputs": {"y_names": ["a", "b", "c"]},
        "bounds": {"y_min": [-1, -1], "y_max": [1, 1], "M_bounds": [0, 8], "D_bounds": [0, 6]},
        "constraints": {"use_nn": True},
        "solver": {"name": "SCIP"},
        "model": {"type": "MTLSH"},
    }
    with pytest.raises(ValueError) as e:
        validate_optimization_config(cfg)
    assert "length" in str(e.value)


def test_validator_flags_missing_sections_and_bad_solver():
    with pytest.raises(ValueError) as e:
        validate_optimization_config({"solver": {"name": "NOPE"}})
    msg = str(e.value)
    assert "system" in msg and "known solvers" in msg
