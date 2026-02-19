"""Scheduler configuration schema."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from cuvis_ai_schemas.base import BaseSchemaModel


class SchedulerConfig(BaseSchemaModel):
    """Learning rate scheduler configuration."""

    __proto_message__: ClassVar[str] = "SchedulerConfig"

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


__all__ = ["SchedulerConfig"]
