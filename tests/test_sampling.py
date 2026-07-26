"""Tests for directed-walk boundary sampling."""

from __future__ import annotations

import numpy as np
import torch

from research.sampling.directed_walks import boundary_samples, directed_walk


def _toy_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))


def test_directed_walk_moves_toward_target():
    model = _toy_model()
    x0 = np.full(5, 0.5)
    sched = [0, 1, 2]
    with torch.no_grad():
        start = float(model(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0, 0])
    target = start + 0.5  # a reachable shift
    xb, achieved = directed_walk(model, x0, sched, target_index=0, target_value=target, steps=120)
    assert abs(achieved - target) < abs(start - target)          # got closer
    assert np.allclose(xb[3:], x0[3:])                           # non-sched frozen
    assert xb.min() >= -1e-6 and xb.max() <= 1 + 1e-6            # stayed in box


def test_boundary_samples_reduces_gap_on_average():
    model = _toy_model()
    x_seed = np.full(5, 0.5)
    with torch.no_grad():
        base = float(model(torch.tensor(x_seed, dtype=torch.float32).unsqueeze(0))[0, 0])
    res = boundary_samples(model, x_seed, sched_idx=[0, 1, 2], target_index=0,
                           target_value=base + 0.3, n=8, steps=120)
    assert res["mean_gap_after"] < res["mean_gap_before"]
    assert res["points"].shape == (8, 5)
