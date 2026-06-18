"""Tests for the NodeProfilingStats frozen dataclass."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from cuvis_ai_schemas.pipeline.profiling import NodeProfilingStats


def _stats(**overrides: Any) -> NodeProfilingStats:
    defaults: dict[str, Any] = {
        "node_name": "DoubleNode",
        "stage": "inference",
        "count": 10,
        "mean_ms": 1.5,
        "median_ms": 1.4,
        "std_ms": 0.2,
        "min_ms": 1.0,
        "max_ms": 2.0,
        "total_ms": 15.0,
        "last_ms": 1.6,
    }
    defaults.update(overrides)
    return NodeProfilingStats(**defaults)


def test_fields_round_trip_via_asdict():
    """All declared fields are stored and recoverable via dataclasses.asdict."""
    stats = _stats()
    as_dict = dataclasses.asdict(stats)
    assert as_dict["node_name"] == "DoubleNode"
    assert as_dict["stage"] == "inference"
    assert as_dict["count"] == 10
    assert as_dict["total_ms"] == 15.0


def test_is_frozen():
    """NodeProfilingStats is immutable."""
    stats = _stats()
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.count = 11  # type: ignore[misc]


def test_equality_by_value():
    """Two stats with identical fields compare equal."""
    assert _stats() == _stats()
    assert _stats(count=11) != _stats(count=10)
