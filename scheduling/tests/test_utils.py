import numpy as np
import pytest

from scheduling.utils import (
    activation_masks_mtlshared,
    compute_x_bounds,
    scale_values_with_scaler,
    unscale_values_with_scaler,
)


torch = pytest.importorskip("torch")
from models.models import MTLSharedHeads  # noqa: E402


class CenterScaler:
    def __init__(self, center, scale):
        self.center_ = np.asarray(center, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.center_) / self.scale_


class MinMaxScaler:
    def __init__(self, min_, scale):
        self.min_ = np.asarray(min_, dtype=float)
        self.scale_ = np.asarray(scale, dtype=float)

    def transform(self, x):
        return np.asarray(x, dtype=float) * self.scale_ + self.min_


def test_scale_unscale_roundtrip_center():
    scaler = CenterScaler(center=[1.0, 2.0, 3.0], scale=[2.0, 4.0, 5.0])
    values = np.array([10.0, 20.0], dtype=float)
    idx = [0, 2]
    scaled = scale_values_with_scaler(scaler, values, idx)
    unscaled = unscale_values_with_scaler(scaler, scaled, idx)
    assert np.allclose(unscaled, values)


def test_scale_unscale_roundtrip_minmax():
    scaler = MinMaxScaler(min_=[-1.0, -2.0, -3.0], scale=[0.5, 1.5, 2.0])
    values = np.array([4.0, 5.0], dtype=float)
    idx = [0, 2]
    scaled = scale_values_with_scaler(scaler, values, idx)
    unscaled = unscale_values_with_scaler(scaler, scaled, idx)
    assert np.allclose(unscaled, values)


def test_compute_x_bounds_with_scaler():
    cfg = {
        "bounds": {
            "x_min": [0.0, 2.0],
            "x_max": [2.0, 6.0],
            "use_scaler_for_bounds": True,
        }
    }
    scaler = CenterScaler(center=[1.0, 1.0], scale=[1.0, 2.0])
    x_min, x_max = compute_x_bounds(cfg, x_scaler=scaler)
    expected_min = scaler.transform(np.array([0.0, 2.0]))
    expected_max = scaler.transform(np.array([2.0, 6.0]))
    assert np.allclose(x_min, expected_min)
    assert np.allclose(x_max, expected_max)


def test_activation_masks_mtlshared_shapes():
    model = MTLSharedHeads(in_dim=4, n_tasks=2, shared_sizes=[3], head_sizes=[2])
    x_np = np.zeros((1, 4), dtype=np.float32)
    shared_masks, head_masks = activation_masks_mtlshared(model, x_np)

    shared_relu_count = sum(1 for layer in model.shared if isinstance(layer, torch.nn.ReLU))
    head_relu_per_head = sum(1 for layer in model.heads[0] if isinstance(layer, torch.nn.ReLU))
    assert len(shared_masks) == shared_relu_count
    assert len(head_masks) == head_relu_per_head * model.n_tasks
