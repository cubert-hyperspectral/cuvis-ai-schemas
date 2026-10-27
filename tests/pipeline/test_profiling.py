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


def test_flops_per_call_defaults_to_none():
    """A row built without a FLOP count carries None."""
    assert _stats().flops_per_call is None


def test_positional_construction_of_the_timing_fields_still_works():
    """The ten timing fields keep their positions; the FLOP count comes last."""
    stats = NodeProfilingStats("N", "inference", 3, 1.0, 1.0, 0.0, 1.0, 1.0, 3.0, 1.0)
    assert stats.last_ms == 1.0
    assert stats.flops_per_call is None
    assert [f.name for f in dataclasses.fields(NodeProfilingStats)][-1] == "flops_per_call"


def test_gflop_per_s_golden_pair():
    """2e9 FLOPs per call at a mean of 4 ms is 500 GFLOP/s."""
    assert _stats(flops_per_call=2_000_000_000, mean_ms=4.0).gflop_per_s == pytest.approx(500.0)


def test_gflop_per_s_is_zero_for_a_present_zero_count():
    """A counted node without matmul, conv or attention reports a rate of 0."""
    assert _stats(flops_per_call=0, mean_ms=4.0).gflop_per_s == 0.0


@pytest.mark.parametrize(
    ("flops", "mean_ms"),
    [(None, 4.0), (2_000_000_000, 0.0), (2_000_000_000, -1.0)],
    ids=["no-count", "zero-mean", "negative-mean"],
)
def test_gflop_per_s_is_none_without_count_or_time(flops, mean_ms):
    """No rate without a count or without a positive mean time."""
    assert _stats(flops_per_call=flops, mean_ms=mean_ms).gflop_per_s is None
