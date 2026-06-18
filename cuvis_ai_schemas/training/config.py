"""Main training configuration schema."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.training.callbacks import CallbacksConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.scheduler import SchedulerConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig


class TrainingConfig(BaseSchemaModel):
    """Complete training configuration."""

    __proto_message__: ClassVar[str] = "TrainingConfig"

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

    @model_validator(mode="after")
    def _sync_trainer_fields(self) -> TrainingConfig:
        """Keep overlapping top-level hyperparameters in sync with the trainer.

        One uniform rule per overlapping field: an explicitly-set top-level
        value (including ``None``) is authoritative and pushed down to the
        trainer; otherwise a non-``None`` trainer value is pulled up. This
        treats all three fields identically (the previous code special-cased
        ``gradient_clip_val`` so an explicit top-level ``None`` did not clear a
        stale trainer value).
        """
        fields_set: set[str] = getattr(self, "model_fields_set", set())

        def _sync(top_name: str, trainer_name: str) -> None:
            """Sync one overlapping field; explicit top-level value wins."""
            if top_name not in fields_set and getattr(self.trainer, trainer_name) is not None:
                setattr(self, top_name, getattr(self.trainer, trainer_name))
            else:
                setattr(self.trainer, trainer_name, getattr(self, top_name))

        _sync("max_epochs", "max_epochs")
        _sync("gradient_clip_val", "gradient_clip_val")
        _sync("accumulate_grad_batches", "accumulate_grad_batches")

        if self.callbacks is not None:
            self.trainer.callbacks = self.callbacks
        return self

    def to_dict_config(self) -> dict[str, Any]:
        """Compatibility shim for legacy OmegaConf usage."""
        try:
            from omegaconf import OmegaConf
        except Exception:
            return self.model_dump(mode="json")

        return OmegaConf.create(self.model_dump())  # type: ignore[return-value]

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
