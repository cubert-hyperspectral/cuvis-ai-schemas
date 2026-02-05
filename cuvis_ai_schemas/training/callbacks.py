"""Callback configuration schemas."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError:
        cuvis_ai_pb2 = None  # type: ignore[assignment]


class _BaseConfig(BaseModel):
    """Base model with strict validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)


class EarlyStoppingConfig(_BaseConfig):
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


class ModelCheckpointConfig(_BaseConfig):
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


class LearningRateMonitorConfig(_BaseConfig):
    """Learning rate monitor callback configuration."""

    logging_interval: Literal["step", "epoch"] | None = Field(
        default="epoch", description="Log lr at 'epoch' or 'step'"
    )
    log_momentum: bool = Field(default=False, description="Log momentum values as well")
    log_weight_decay: bool = Field(default=False, description="Log weight decay values as well")


class CallbacksConfig(_BaseConfig):
    """Callbacks configuration."""

    checkpoint: ModelCheckpointConfig | None = Field(
        default=None,
        description="Model checkpoint configuration",
        alias="model_checkpoint",
    )
    early_stopping: list[EarlyStoppingConfig] = Field(
        default_factory=list, description="Early stopping configuration(s)"
    )
    learning_rate_monitor: LearningRateMonitorConfig | None = Field(
        default=None, description="Learning rate monitor configuration"
    )

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.CallbacksConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))


__all__ = [
    "EarlyStoppingConfig",
    "ModelCheckpointConfig",
    "LearningRateMonitorConfig",
    "CallbacksConfig",
]
