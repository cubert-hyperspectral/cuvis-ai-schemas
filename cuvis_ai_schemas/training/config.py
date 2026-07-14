"""Main training configuration schema."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.training.callbacks import CallbacksConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.scheduler import SchedulerConfig


class TrainingConfig(BaseSchemaModel):
    """Complete gradient-training configuration.

    Flat set of hyperparameters for a gradient-training run: the run-level
    knobs (``seed``, ``optimizer``, ``scheduler``, ``callbacks``) plus the
    ``pytorch_lightning.Trainer`` keyword arguments. ``callbacks`` is schema
    input used to build real Lightning callbacks (see
    ``to_lightning_kwargs`` / ``create_callbacks_from_config``), not a raw
    ``pl.Trainer`` keyword.
    """

    __proto_message__: ClassVar[str] = "TrainingConfig"

    # Orchestration (not pl.Trainer kwargs)
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

    # pl.Trainer keyword arguments
    max_epochs: int = Field(default=100, ge=1, le=10000, description="Maximum training epochs")
    accelerator: str = Field(default="auto", description="Accelerator type")
    devices: int | str | None = Field(default=None, description="Number of devices or IDs")
    default_root_dir: str | None = Field(default=None, description="Root directory for outputs")
    precision: str | int = Field(default="32-true", description="Precision mode")
    accumulate_grad_batches: int = Field(
        default=1, ge=1, description="Accumulate gradients over n batches"
    )
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
        default=None, ge=0.0, description="Gradient clipping value (optional)"
    )
    deterministic: bool = Field(default=False, description="Deterministic training")
    benchmark: bool = Field(default=False, description="Enable cudnn benchmark")

    _LIGHTNING_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "max_epochs",
            "accelerator",
            "devices",
            "default_root_dir",
            "precision",
            "accumulate_grad_batches",
            "enable_progress_bar",
            "enable_checkpointing",
            "log_every_n_steps",
            "val_check_interval",
            "check_val_every_n_epoch",
            "gradient_clip_val",
            "deterministic",
            "benchmark",
        }
    )

    def to_lightning_kwargs(self) -> dict[str, Any]:
        """Return the subset of fields passed directly to ``pl.Trainer(**kwargs)``.

        An explicit allowlist: ``pl.Trainer`` raises on an unknown keyword, so
        a field that is not a raw trainer argument (``seed``, ``optimizer``,
        ``scheduler``, ``callbacks``) is never forwarded. ``callbacks`` is built
        into real Lightning callbacks separately by the trainer.
        """
        return self.model_dump(include=set(self._LIGHTNING_FIELDS), exclude_none=True)

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
