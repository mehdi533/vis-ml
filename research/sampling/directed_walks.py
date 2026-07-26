"""Gradient-driven boundary sampling over the schedulable (M/D) inputs."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


def directed_walk(
    model: torch.nn.Module,
    x_seed: np.ndarray,
    sched_idx: Sequence[int],
    target_index: int,
    target_value: float,
    steps: int = 60,
    lr: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """Move the schedulable inputs so output ``target_index`` approaches ``target_value``.

    Non-schedulable inputs are held at their seed; schedulable inputs are updated
    by gradient descent on ``(output - target_value)**2`` and clamped to the
    scaled box [0, 1]. Returns the resulting input vector and its achieved output.
    """
    model.eval()
    x0 = torch.tensor(np.asarray(x_seed, float), dtype=torch.float32)
    x = x0.clone().requires_grad_(True)
    sched = torch.zeros_like(x0, dtype=torch.bool)
    sched[list(sched_idx)] = True
    opt = torch.optim.Adam([x], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        out = model(x.unsqueeze(0))[0, target_index]
        loss = (out - float(target_value)) ** 2
        loss.backward()
        with torch.no_grad():
            x.grad[~sched] = 0.0          # freeze non-schedulable inputs
        opt.step()
        with torch.no_grad():
            x[~sched] = x0[~sched]        # re-pin them exactly
            x.clamp_(0.0, 1.0)

    with torch.no_grad():
        achieved = float(model(x.unsqueeze(0))[0, target_index])
    return x.detach().numpy(), achieved


def boundary_samples(
    model: torch.nn.Module,
    x_seed: np.ndarray,
    sched_idx: Sequence[int],
    target_index: int,
    target_value: float,
    n: int = 16,
    steps: int = 60,
    seed: int = 0,
) -> dict:
    """Run ``n`` directed walks from random schedulable starts; collect boundary points.

    Returns the sampled inputs, their achieved outputs, and how much closer to the
    target the walk got on average (a measure of boundary focusing).
    """
    rng = np.random.default_rng(seed)
    x_seed = np.asarray(x_seed, float)
    pts, achieved, start_gap, end_gap = [], [], [], []
    for _ in range(n):
        x0 = x_seed.copy()
        x0[list(sched_idx)] = rng.uniform(0.0, 1.0, size=len(sched_idx))
        with torch.no_grad():
            s = float(model(torch.tensor(x0, dtype=torch.float32).unsqueeze(0))[0, target_index])
        xb, a = directed_walk(model, x0, sched_idx, target_index, target_value, steps=steps)
        pts.append(xb)
        achieved.append(a)
        start_gap.append(abs(s - target_value))
        end_gap.append(abs(a - target_value))
    return {
        "points": np.asarray(pts),
        "achieved": np.asarray(achieved),
        "mean_gap_before": float(np.mean(start_gap)),
        "mean_gap_after": float(np.mean(end_gap)),
    }
