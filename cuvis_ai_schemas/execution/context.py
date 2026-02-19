"""Execution context for pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cuvis_ai_schemas.enums.types import ExecutionStage


@dataclass
class Context:
    """Execution context passed to executor and nodes.

    Contains runtime information that doesn't flow through data edges.
    This replaces mutable global state with explicit context parameters.

    Attributes
    ----------
    stage : ExecutionStage
        Execution stage: "train", "val", "test", "inference"
    epoch : int
        Current training epoch
    batch_idx : int
        Current batch index within epoch
    global_step : int
        Global training step across all epochs

    Examples
    --------
    >>> context = Context(stage=ExecutionStage.TRAIN, epoch=5, batch_idx=42, global_step=1337)
    >>> executor.forward(context=context, batch=batch)

    Notes
    -----
    Future extensions for distributed training:
    - rank: int (process rank in distributed training)
    - world_size: int (total number of processes)
    """

    stage: ExecutionStage = ExecutionStage.INFERENCE
    epoch: int = 0
    batch_idx: int = 0
    global_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "epoch": self.epoch,
            "batch_idx": self.batch_idx,
            "global_step": self.global_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Context:
        """Create from dictionary."""
        return cls(
            stage=ExecutionStage(data["stage"]),
            epoch=data["epoch"],
            batch_idx=data["batch_idx"],
            global_step=data["global_step"],
        )
