"""Tests for TrainerConfig field constraints and union fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training.callbacks import CallbacksConfig, EarlyStoppingConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig


def test_defaults():
    """Trainer defaults match the documented values."""
    trainer = TrainerConfig()
    assert trainer.max_epochs == 100
    assert trainer.accelerator == "auto"
    assert trainer.precision == "32-true"
    assert trainer.devices is None
    assert trainer.callbacks is None


@pytest.mark.parametrize("precision", ["16-mixed", "bf16", 16, 32])
def test_precision_accepts_str_or_int(precision):
    """precision is a str | int union."""
    assert TrainerConfig(precision=precision).precision == precision


@pytest.mark.parametrize("devices", [1, "auto", None])
def test_devices_accepts_int_str_or_none(devices):
    """devices is an int | str | None union."""
    assert TrainerConfig(devices=devices).devices == devices


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_epochs": 0},
        {"accumulate_grad_batches": 0},
        {"log_every_n_steps": 0},
        {"check_val_every_n_epoch": 0},
        {"gradient_clip_val": -1.0},
        {"val_check_interval": -0.5},
    ],
)
def test_out_of_range_values_rejected(kwargs):
    """Field range constraints reject out-of-range values."""
    with pytest.raises(ValidationError):
        TrainerConfig(**kwargs)


def test_nested_callbacks_and_roundtrip():
    """A nested CallbacksConfig survives a dict round-trip."""
    trainer = TrainerConfig(
        callbacks=CallbacksConfig(early_stopping=[EarlyStoppingConfig(monitor="val_loss")])
    )
    assert TrainerConfig.from_dict(trainer.to_dict()) == trainer


def test_extra_field_rejected():
    """Unknown fields are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        TrainerConfig(unknown="x")
