"""Tests for interval bound propagation / ReLU stability (embeddability tooling)."""

from __future__ import annotations

import numpy as np
import pytest

from research.embeddability.bounds import (
    LinearLayer,
    propagate_interval_bounds,
    relu_stability,
)


def test_bounds_contain_sampled_forward_pass():
    """IBP intervals must contain the true pre-activations for all inputs in the box."""
    rng = np.random.default_rng(0)
    layers = [
        LinearLayer(rng.normal(size=(8, 4)), rng.normal(size=8)),
        LinearLayer(rng.normal(size=(6, 8)), rng.normal(size=6)),
        LinearLayer(rng.normal(size=(2, 6)), rng.normal(size=2)),
    ]
    x_lo = np.full(4, -1.0)
    x_hi = np.full(4, 1.0)
    preacts = propagate_interval_bounds(layers, x_lo, x_hi)

    # Monte-Carlo: every sampled input's pre-activations lie within the intervals.
    for _ in range(2000):
        x = rng.uniform(-1.0, 1.0, size=4)
        h = x
        for k, layer in enumerate(layers):
            z = layer.W @ h + layer.b
            z_lo, z_hi = preacts[k]
            assert np.all(z >= z_lo - 1e-9)
            assert np.all(z <= z_hi + 1e-9)
            if k < len(layers) - 1:
                h = np.maximum(z, 0.0)


def test_stable_neurons_need_no_binary():
    # Layer with strictly positive weights and positive input box -> all active.
    W = np.ones((3, 2))
    b = np.array([0.5, 0.5, 0.5])
    layers = [LinearLayer(W, b), LinearLayer(np.ones((1, 3)), np.zeros(1))]
    preacts = propagate_interval_bounds(layers, x_lo=[0.1, 0.1], x_hi=[1.0, 1.0])
    stab = relu_stability(preacts)  # excludes output layer
    assert len(stab) == 1
    assert stab[0].n_unstable == 0
    assert stab[0].n_active == 3
    assert stab[0].binary_fraction == 0.0


def test_tighter_input_box_reduces_binaries():
    """Shrinking the input box should never increase the unstable-neuron count."""
    rng = np.random.default_rng(3)
    layers = [
        LinearLayer(rng.normal(size=(20, 5)), rng.normal(size=20)),
        LinearLayer(rng.normal(size=(1, 20)), rng.normal(size=1)),
    ]
    wide = relu_stability(propagate_interval_bounds(layers, [-2] * 5, [2] * 5))[0]
    tight = relu_stability(propagate_interval_bounds(layers, [-0.2] * 5, [0.2] * 5))[0]
    assert tight.n_unstable <= wide.n_unstable
    assert tight.max_abs_bigM <= wide.max_abs_bigM + 1e-9


def test_input_validation():
    layer = [LinearLayer(np.ones((2, 2)), np.zeros(2))]
    with pytest.raises(ValueError):
        propagate_interval_bounds(layer, x_lo=[1.0, 1.0], x_hi=[0.0, 0.0])  # lo > hi


def test_obbt_never_wider_than_ibp():
    """OBBT must tighten (or match) IBP bounds, and never increase binaries/big-M."""
    from research.embeddability.bounds import propagate_interval_bounds
    from research.embeddability.obbt import compare_ibp_obbt, obbt_bounds

    rng = np.random.default_rng(1)
    layers = [
        LinearLayer(rng.normal(size=(6, 4)), rng.normal(size=6)),
        LinearLayer(rng.normal(size=(4, 6)), rng.normal(size=4)),
        LinearLayer(rng.normal(size=(1, 4)), rng.normal(size=1)),
    ]
    lo, hi = np.full(4, -0.5), np.full(4, 0.5)
    ibp = propagate_interval_bounds(layers, lo, hi)
    obbt = obbt_bounds(layers, lo, hi)
    for (il, ih), (ol, oh) in zip(ibp, obbt):
        assert np.all(ol >= il - 1e-6)   # OBBT lower bound no lower than IBP
        assert np.all(oh <= ih + 1e-6)   # OBBT upper bound no higher than IBP
    cmp = compare_ibp_obbt(layers, lo, hi)
    assert cmp["obbt_binaries"] <= cmp["ibp_binaries"]
    assert cmp["obbt_max_bigM"] <= cmp["ibp_max_bigM"] + 1e-6
    with pytest.raises(ValueError):
        LinearLayer(np.ones((2, 2)), np.zeros(3))  # b shape mismatch
