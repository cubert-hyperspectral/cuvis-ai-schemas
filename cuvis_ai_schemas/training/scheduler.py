"""Scheduler configuration schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError:
        cuvis_ai_pb2 = None  # type: ignore[assignment]


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    name: str | None = Field(
        default=None, description="Scheduler type: cosine, step, exponential, plateau"
    )
    warmup_epochs: int = Field(default=0, ge=0, description="Number of warmup epochs")
    min_lr: float = Field(default=1e-6, ge=0.0, description="Minimum learning rate")
    t_max: int | None = Field(
        default=None, ge=1, description="Maximum iterations (for cosine annealing)"
    )
    step_size: int | None = Field(
        default=None, ge=1, description="Period of LR decay (for step scheduler)"
    )
    gamma: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Multiplicative factor of LR decay"
    )
    monitor: str | None = Field(
        default=None, description="Metric to monitor (for plateau/reduce_on_plateau)"
    )
    mode: str = Field(default="min", description="min or max for monitored metrics")
    factor: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LR reduction factor for ReduceLROnPlateau",
    )
    patience: int = Field(default=10, ge=0, description="Patience for plateau scheduler")
    threshold: float = Field(default=1e-4, ge=0.0, description="Plateau threshold")
    threshold_mode: str = Field(default="rel", description="Plateau threshold mode")
    cooldown: int = Field(default=0, ge=0, description="Cooldown epochs for plateau")
    eps: float = Field(default=1e-8, ge=0.0, description="Minimum change in LR for plateau")
    verbose: bool = Field(default=False, description="Verbose scheduler logging")

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.SchedulerConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))


__all__ = ["SchedulerConfig"]
