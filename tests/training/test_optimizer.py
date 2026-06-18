"""Tests for OptimizerConfig field constraints."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training.optimizer import OptimizerConfig


def test_defaults():
    """Default optimizer is adamw with a sane learning rate."""
    opt = OptimizerConfig()
    assert opt.name == "adamw"
    assert opt.lr == pytest.approx(1e-3)
    assert opt.betas is None


def test_valid_betas_tuple():
    """A well-formed 2-tuple of betas is accepted."""
    opt = OptimizerConfig(betas=(0.9, 0.999))
    assert opt.betas == (0.9, 0.999)


@pytest.mark.parametrize("lr", [0.0, -0.1, 1.5])
def test_lr_out_of_range_rejected(lr):
    """lr must be in (0, 1]; zero, negative, and >1 are rejected."""
    with pytest.raises(ValidationError):
        OptimizerConfig(lr=lr)


@pytest.mark.parametrize("weight_decay", [-0.1, 1.1])
def test_weight_decay_out_of_range_rejected(weight_decay):
    """weight_decay must be in [0, 1]."""
    with pytest.raises(ValidationError):
        OptimizerConfig(weight_decay=weight_decay)


@pytest.mark.parametrize("momentum", [-0.1, 1.1])
def test_momentum_out_of_range_rejected(momentum):
    """momentum must be in [0, 1] when provided."""
    with pytest.raises(ValidationError):
        OptimizerConfig(momentum=momentum)


@pytest.mark.parametrize("betas", [(0.9,), (0.9, 0.999, 0.5)])
def test_betas_wrong_arity_rejected(betas):
    """betas must be a 2-tuple; the type enforces arity."""
    with pytest.raises(ValidationError):
        OptimizerConfig(betas=betas)


def test_extra_field_rejected():
    """Unknown fields are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        OptimizerConfig(nesterov=True)
