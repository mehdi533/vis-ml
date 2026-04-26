# __init__.py
# Public entrypoints for the data-generation workflow with lazy imports.

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def load_generation_config(path: str) -> dict:
    """Load a generation config without eager submodule imports."""
    from data_generation.run_sims import load_config

    return load_config(path)


def run_generation(config_path: str):
    """Run full generation pipeline."""
    from data_generation.run_sims import run_generation as _run_generation

    return _run_generation(config_path)


def run_one_sim(config: dict, sim_id: int, rng: Optional[np.random.Generator] = None):
    """Run one simulation."""
    from data_generation.run_sims import run_one_sim as _run_one_sim

    return _run_one_sim(config, sim_id, rng)


def configure_tds(ss, tds_cfg: dict) -> None:
    """Configure ANDES TDS settings."""
    from data_generation.run_sims import configure_tds as _configure_tds

    _configure_tds(ss, tds_cfg)


def define_operating_point(
    ss,
    *,
    base_scale: float,
    M_vec,
    D_vec,
    ed_cfg: dict,
    scale_pv: bool,
):
    """Define the pre-disturbance operating point."""
    from data_generation.run_sims import define_operating_point as _define_operating_point

    return _define_operating_point(
        ss,
        base_scale=base_scale,
        M_vec=M_vec,
        D_vec=D_vec,
        ed_cfg=ed_cfg,
        scale_pv=scale_pv,
    )


def run_ed_dispatch(ss, ed_cfg: dict, *, ibr_idx: Optional[Sequence[int]] = None):
    """Solve ED dispatch."""
    from data_generation.run_sims import run_ed_dispatch as _run_ed_dispatch

    return _run_ed_dispatch(ss, ed_cfg, ibr_idx=ibr_idx)


def pick_line_contingencies(ss, cont_cfg: dict, rng: np.random.Generator):
    """Resolve selectable line contingencies."""
    from data_generation.disturbance_dispatch import pick_line_contingencies as _pick_line_contingencies

    return _pick_line_contingencies(ss, cont_cfg, rng)


def select_step_targets(ss, load_cfg: dict, rng: Optional[np.random.Generator] = None):
    """Resolve PQ targets for load-step disturbance."""
    from data_generation.disturbance_dispatch import select_step_targets as _select_step_targets

    return _select_step_targets(ss, load_cfg, rng=rng)

__all__ = [
    "configure_tds",
    "define_operating_point",
    "load_generation_config",
    "pick_line_contingencies",
    "run_ed_dispatch",
    "run_generation",
    "run_one_sim",
    "select_step_targets",
]
