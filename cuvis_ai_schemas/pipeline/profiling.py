"""Profiling statistics dataclass for pipeline node runtime profiling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeProfilingStats:
    """Immutable snapshot of accumulated runtime statistics for a single node.

    All timing values are in milliseconds.

    Attributes
    ----------
    node_name : str
        Unique node identifier within the pipeline (e.g. ``"DoubleNode"``
        or ``"DoubleNode-2"`` for counter > 0).
    stage : str
        Canonical lowercase execution stage value from
        ``ExecutionStage.value`` (e.g. ``"inference"``, ``"train"``).
    count : int
        Number of accumulated samples (after warm-up skip).
    mean_ms : float
        Online mean of recorded durations.
    median_ms : float
        Approximate median (P² estimator after warm-up buffer).
    std_ms : float
        Population standard deviation of recorded durations.
    min_ms : float
        Minimum recorded duration.
    max_ms : float
        Maximum recorded duration.
    total_ms : float
        Sum of all recorded durations.
    last_ms : float
        Most recently recorded duration.
    """

    node_name: str
    stage: str
    count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    total_ms: float
    last_ms: float
