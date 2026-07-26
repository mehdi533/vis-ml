"""Tests for the loss registry, focusing on the pinball (quantile) loss."""

from __future__ import annotations

import pytest
import torch

from models.losses import LOSS_FACTORY, PinballLoss, build_loss, list_losses


def test_pinball_registered():
    assert "pinball" in LOSS_FACTORY
    assert "pinball" in list_losses()


def test_pinball_penalises_underprediction_more_when_tau_high():
    loss = PinballLoss(tau=0.9)
    target = torch.zeros(4, 1)
    under = torch.full((4, 1), -1.0)  # pred below target by 1 (under-prediction)
    over = torch.full((4, 1), 1.0)    # pred above target by 1 (over-prediction)
    # tau=0.9: under-prediction weight 0.9, over-prediction weight 0.1.
    assert float(loss(under, target)) == pytest.approx(0.9)
    assert float(loss(over, target)) == pytest.approx(0.1)
    assert float(loss(under, target)) > float(loss(over, target))


def test_pinball_symmetric_at_half():
    loss = PinballLoss(tau=0.5)
    target = torch.zeros(3, 1)
    a = float(loss(torch.full((3, 1), -2.0), target))
    b = float(loss(torch.full((3, 1), 2.0), target))
    assert a == pytest.approx(b)  # symmetric == scaled L1


def test_pinball_invalid_tau():
    with pytest.raises(ValueError):
        PinballLoss(tau=0.0)
    with pytest.raises(ValueError):
        PinballLoss(tau=1.0)


def test_build_loss_pinball():
    fn, params = build_loss("pinball", ce_weights=[], device=torch.device("cpu"), out_dim=6)
    preds = torch.zeros(5, 6)
    targets = torch.ones(5, 6)
    val = float(fn(preds, targets))
    assert val > 0 and params == []
