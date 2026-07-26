"""Tests for the multi-system registry (pure parts; no ANDES needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.systems.registry import (
    REGCV1_TEMPLATE,
    SYSTEM_REGISTRY,
    SystemSpec,
    augment_with_grid_forming_ibrs,
    resolve_case_path,
)


def test_registry_has_baseline_and_scaleup():
    assert "ieee39" in SYSTEM_REGISTRY
    assert "npcc140" in SYSTEM_REGISTRY
    ieee39 = SYSTEM_REGISTRY["ieee39"]
    assert ieee39.n_ibrs == 4
    assert ieee39.ibr_gen_idxs == (1, 6, 8, 9)
    npcc = SYSTEM_REGISTRY["npcc140"]
    assert npcc.n_buses == 140
    assert npcc.n_ibrs == 0  # needs augmentation


def test_regcv1_template_is_complete():
    for key in ("fn", "Tc", "gammap", "gammaq", "ra"):
        assert key in REGCV1_TEMPLATE


def test_resolve_repo_relative_path():
    spec = SystemSpec(name="x", case_path="data_generation/andes_cases/ieee39_full_ibrs.xlsx", n_buses=39, description="")
    p = resolve_case_path(spec)
    assert p.is_absolute()
    assert p.name == "ieee39_full_ibrs.xlsx"


def test_resolve_andes_scheme_uses_bundled_root():
    # Only checks the resolved path shape (requires andes installed for the root).
    p = resolve_case_path("andes:npcc/npcc.xlsx")
    assert p.name == "npcc.xlsx"
    assert "cases" in str(Path(p)).replace("\\", "/")


def test_augment_validates_lengths_before_touching_system():
    # Mismatched lengths must raise before any ss access, so ss=None is safe.
    with pytest.raises(ValueError):
        augment_with_grid_forming_ibrs(None, gen_idxs=[1, 2], m_values=[4.0], d_values=[2.0, 2.0])


def test_dynamify_makes_ieee118_tds_ready():
    """IEEE 118 ships power-flow-only; dynamify_case must make TDS run cleanly."""
    import os
    import warnings

    import andes

    from research.systems.dynamify import dynamify_case

    warnings.filterwarnings("ignore")
    andes.config_logger(stream_level=40)
    case = os.path.join(os.path.dirname(andes.__file__), "cases", "matpower", "case118.m")
    ss = andes.load(case, setup=False, no_output=True)
    created = dynamify_case(ss, target_H=4.0)
    ss.setup()
    assert len(created["GENROU"]) == ss.GENROU.n > 0
    assert len(created["TGOV1N"]) == ss.TGOV1N.n > 0
    assert bool(ss.PFlow.run())
    ss.TDS.config.tf = 1.0
    ss.TDS.config.no_tqdm = 1
    ss.TDS.run()
    assert ss.exit_code == 0  # TDS initialised and integrated
