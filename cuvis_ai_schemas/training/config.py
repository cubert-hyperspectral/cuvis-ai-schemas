"""Main training configuration schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cuvis_ai_schemas.training.callbacks import CallbacksConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.scheduler import SchedulerConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig

if TYPE_CHECKING:
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError:
        cuvis_ai_pb2 = None  # type: ignore[assignment]


class TrainingConfig(BaseModel):
    """Complete training configuration."""

    seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    scheduler: SchedulerConfig | None = Field(
        default=None, description="Learning rate scheduler (optional)"
    )
    callbacks: CallbacksConfig | None = Field(
        default=None, description="Training callbacks (optional)"
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig, description="Lightning Trainer configuration"
    )
    max_epochs: int = Field(default=100, ge=1, le=10000, description="Maximum training epochs")
    batch_size: int = Field(default=32, ge=1, description="Batch size")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    gradient_clip_val: float | None = Field(
        default=None, ge=0.0, description="Gradient clipping value (optional)"
    )
    accumulate_grad_batches: int = Field(
        default=1, ge=1, description="Accumulate gradients over n batches"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="after")
    def _sync_trainer_fields(self) -> TrainingConfig:
        """Keep top-level hyperparameters in sync with trainer config."""
        fields_set: set[str] = getattr(self, "model_fields_set", set())

        # max_epochs: prefer explicit trainer value when top-level not provided
        if "max_epochs" not in fields_set and self.trainer.max_epochs is not None:
            self.max_epochs = self.trainer.max_epochs
        else:
            self.trainer.max_epochs = self.max_epochs

        # gradient_clip_val
        if "gradient_clip_val" not in fields_set and self.trainer.gradient_clip_val is not None:
            self.gradient_clip_val = self.trainer.gradient_clip_val
        elif self.gradient_clip_val is not None:
            self.trainer.gradient_clip_val = self.gradient_clip_val

        # accumulate_grad_batches
        if (
            "accumulate_grad_batches" not in fields_set
            and self.trainer.accumulate_grad_batches is not None
        ):
            self.accumulate_grad_batches = self.trainer.accumulate_grad_batches
        else:
            self.trainer.accumulate_grad_batches = self.accumulate_grad_batches

        # callbacks
        if self.callbacks is not None:
            self.trainer.callbacks = self.callbacks
        return self

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.TrainingConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        """JSON serialization helper for legacy callers."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> TrainingConfig:
        """Create from JSON string."""
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    def to_dict_config(self) -> dict[str, Any]:
        """Compatibility shim for legacy OmegaConf usage."""
        try:
            from omegaconf import OmegaConf
        except Exception:
            return self.model_dump(mode="json")

        return OmegaConf.create(self.model_dump())  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_dict_config(cls, config: dict[str, Any]) -> TrainingConfig:
        """Create from DictConfig (OmegaConf) or dictionary."""
        if config.__class__.__name__ == "DictConfig":  # Avoid hard dependency in type hints
            from omegaconf import OmegaConf

            config = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
        elif not isinstance(config, dict):
            config = dict(config)
        return cls.model_validate(config)


__all__ = ["TrainingConfig"]
