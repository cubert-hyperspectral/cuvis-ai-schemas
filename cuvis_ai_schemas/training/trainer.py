"""Trainer configuration schema."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cuvis_ai_schemas.training.callbacks import CallbacksConfig


class TrainerConfig(BaseModel):
    """Lightning Trainer configuration."""

    max_epochs: int = Field(default=100, ge=1, description="Maximum number of epochs")
    accelerator: str = Field(default="auto", description="Accelerator type")
    devices: int | str | None = Field(default=None, description="Number of devices or IDs")
    default_root_dir: str | None = Field(default=None, description="Root directory for outputs")
    precision: str | int = Field(default="32-true", description="Precision mode")
    accumulate_grad_batches: int = Field(default=1, ge=1, description="Accumulate gradients")
    enable_progress_bar: bool = Field(default=True, description="Show progress bar")
    enable_checkpointing: bool = Field(default=False, description="Enable checkpointing")
    log_every_n_steps: int = Field(default=50, ge=1, description="Log frequency in steps")
    val_check_interval: float | int | None = Field(
        default=1.0, ge=0.0, description="Validation interval"
    )
    check_val_every_n_epoch: int | None = Field(
        default=1, ge=1, description="Validate every n epochs"
    )
    gradient_clip_val: float | None = Field(
        default=None, ge=0.0, description="Gradient clipping value"
    )
    deterministic: bool = Field(default=False, description="Deterministic training")
    benchmark: bool = Field(default=False, description="Enable cudnn benchmark")
    callbacks: CallbacksConfig | None = Field(
        default=None, description="Callback configurations for trainer"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


__all__ = ["TrainerConfig"]
