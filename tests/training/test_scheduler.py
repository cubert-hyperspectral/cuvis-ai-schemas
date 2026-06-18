"""Tests for SchedulerConfig field constraints."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training.scheduler import SchedulerConfig


def test_defaults():
    """A default scheduler config is constructible with no name."""
    sched = SchedulerConfig()
    assert sched.name is None
    assert sched.warmup_epochs == 0
    assert sched.mode == "min"


def test_valid_construction_and_roundtrip():
    """A populated scheduler round-trips through dict serialization."""
    sched = SchedulerConfig(name="cosine", warmup_epochs=5, t_max=100, gamma=0.5)
    assert SchedulerConfig.from_dict(sched.to_dict()) == sched


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_epochs": -1},
        {"min_lr": -1e-6},
        {"t_max": 0},
        {"step_size": 0},
        {"gamma": 1.5},
        {"factor": -0.1},
        {"patience": -1},
        {"cooldown": -1},
        {"eps": -1e-9},
    ],
)
def test_out_of_range_values_rejected(kwargs):
    """Field range constraints (ge/le) reject out-of-range values."""
    with pytest.raises(ValidationError):
        SchedulerConfig(**kwargs)


def test_extra_field_rejected():
    """Unknown fields are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        SchedulerConfig(unknown="x")
