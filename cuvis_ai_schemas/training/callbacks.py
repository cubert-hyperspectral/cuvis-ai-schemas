"""Callback configuration schemas."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar, Literal

from pydantic import Field

from cuvis_ai_schemas.base import BaseSchemaModel


class EarlyStoppingConfig(BaseSchemaModel):
    """Early stopping callback configuration."""

    monitor: str = Field(description="Metric to monitor")
    patience: int = Field(default=10, ge=1, description="Number of epochs to wait")
    mode: str = Field(default="min", description="min or max")
    min_delta: float = Field(default=0.0, ge=0.0, description="Minimum change to qualify")
    stopping_threshold: float | None = Field(
        default=None, description="Stop once monitored metric reaches this threshold"
    )
    verbose: bool = Field(default=True, description="Whether to log state changes")
    strict: bool = Field(default=True, description="Whether to crash if monitor is not found")
    check_finite: bool = Field(
        default=True, description="Stop when monitor becomes NaN or infinite"
    )
    divergence_threshold: float | None = Field(
        default=None,
        description="Stop training when monitor becomes worse than this threshold",
    )
    check_on_train_epoch_end: bool | None = Field(
        default=None,
        description="Whether to run early stopping at end of training epoch",
    )
    log_rank_zero_only: bool = Field(
        default=False, description="Log status only for rank 0 process"
    )


class ModelCheckpointConfig(BaseSchemaModel):
    """Model checkpoint callback configuration."""

    dirpath: str = Field(default="checkpoints", description="Directory to save checkpoints")
    filename: str | None = Field(default=None, description="Checkpoint filename pattern")
    monitor: str = Field(default="val_loss", description="Metric to monitor")
    mode: str = Field(default="min", description="min or max")
    save_top_k: int = Field(default=3, ge=-1, description="Save top k checkpoints (-1 for all)")
    every_n_epochs: int = Field(default=1, ge=1, description="Save every n epochs")
    save_last: bool | Literal["link"] | None = Field(
        default=False, description="Also save last checkpoint (or 'link' for symlink)"
    )
    auto_insert_metric_name: bool = Field(
        default=True, description="Automatically insert metric name into filename"
    )
    verbose: bool = Field(default=False, description="Verbosity mode")
    save_on_exception: bool = Field(
        default=False, description="Whether to save checkpoint when exception is raised"
    )
    save_weights_only: bool = Field(
        default=False,
        description="If True, only save model weights, not optimizer states",
    )
    every_n_train_steps: int | None = Field(
        default=None,
        description="How many training steps to wait before saving checkpoint",
    )
    train_time_interval: timedelta | None = Field(
        default=None, description="Checkpoints monitored at specified time interval"
    )
    save_on_train_epoch_end: bool | None = Field(
        default=None,
        description="Whether to run checkpointing at end of training epoch",
    )
    enable_version_counter: bool = Field(
        default=True, description="Whether to append version to existing file name"
    )


class LearningRateMonitorConfig(BaseSchemaModel):
    """Learning rate monitor callback configuration."""

    logging_interval: Literal["step", "epoch"] | None = Field(
        default="epoch", description="Log lr at 'epoch' or 'step'"
    )
    log_momentum: bool = Field(default=False, description="Log momentum values as well")
    log_weight_decay: bool = Field(default=False, description="Log weight decay values as well")


class CallbacksConfig(BaseSchemaModel):
    """Callbacks configuration."""

    __proto_message__: ClassVar[str] = "CallbacksConfig"

    checkpoint: ModelCheckpointConfig | None = Field(
        default=None,
        description="Model checkpoint configuration",
    )
    early_stopping: list[EarlyStoppingConfig] = Field(
        default_factory=list, description="Early stopping configuration(s)"
    )
    learning_rate_monitor: LearningRateMonitorConfig | None = Field(
        default=None, description="Learning rate monitor configuration"
    )


def create_callbacks_from_config(config: CallbacksConfig | None) -> list[Any]:
    """Create PyTorch Lightning callback instances from configuration.

    Requires pytorch_lightning to be installed (optional dependency).
    """
    if config is None:
        return []

    try:
        from pytorch_lightning.callbacks import (
            EarlyStopping,
            LearningRateMonitor,
            ModelCheckpoint,
        )
    except ImportError as exc:
        raise ImportError(
            "pytorch_lightning is required for create_callbacks_from_config(). "
            "Install with: pip install pytorch-lightning"
        ) from exc

    # Each config's fields mirror its Lightning callback's keyword arguments 1:1.
    callbacks: list[Any] = [
        EarlyStopping(**es_config.model_dump()) for es_config in config.early_stopping
    ]
    if config.checkpoint is not None:
        callbacks.append(ModelCheckpoint(**config.checkpoint.model_dump()))
    if config.learning_rate_monitor is not None:
        callbacks.append(LearningRateMonitor(**config.learning_rate_monitor.model_dump()))

    return callbacks


__all__ = [
    "EarlyStoppingConfig",
    "ModelCheckpointConfig",
    "LearningRateMonitorConfig",
    "CallbacksConfig",
    "create_callbacks_from_config",
]
