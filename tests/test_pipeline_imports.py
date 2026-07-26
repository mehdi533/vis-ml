"""Fast safety-net tests: packages import, configs parse, models instantiate.

These run in seconds (no ANDES/solve) and catch the most common breakages
introduced by refactoring. The full end-to-end check is `scripts/run_smoke.sh`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_core_packages_import():
    import data_generation  # noqa: F401
    import models  # noqa: F401
    import scheduling  # noqa: F401
    from models import models as model_registry

    assert "MTLSH" in model_registry.MODEL_FACTORY


def test_model_factory_builds_mtlsh():
    import torch
    from models.models import create_model

    model, device = create_model("MTLSH", in_dim=178, out_dim=6, shared_sizes=[32], head_sizes=[16])
    out = model(torch.zeros(4, 178))
    assert out.shape == (4, 6)


@pytest.mark.parametrize(
    "rel",
    [
        "configs/data_generation/smoke.yaml",
        "configs/model/smoke_train.yaml",
        "configs/model/smoke_optimization_ready.yaml",
        "configs/scheduling/smoke/optimization_smoke.yaml",
        "configs/scheduling/base_optimization.yaml",
    ],
)
def test_configs_parse(rel):
    path = ROOT / rel
    assert path.exists(), f"missing config: {rel}"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict) and cfg, f"empty/invalid config: {rel}"


def test_smoke_optimization_uses_free_solver():
    """Guard: the committed smoke optimization config must stay solver-free (SCIP)."""
    path = ROOT / "configs/scheduling/smoke/optimization_smoke.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert cfg["solver"]["name"].upper() == "SCIP"
