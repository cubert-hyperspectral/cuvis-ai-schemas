"""Training run configuration schema."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.pipeline.config import PipelineConfig
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.data import DataConfig


class TrainRunConfig(BaseSchemaModel):
    """Complete reproducible training configuration."""

    __proto_message__: str = "TrainRunConfig"

    name: str = Field(description="Train run identifier")
    pipeline: PipelineConfig | None = Field(
        default=None, description="Pipeline configuration (optional if already built)"
    )
    data: DataConfig = Field(description="Data configuration")

    training: TrainingConfig | None = Field(
        default=None,
        description="Training configuration (required if gradient training)",
    )

    loss_nodes: list[str] = Field(
        default_factory=list, description="Loss node names for gradient training"
    )
    metric_nodes: list[str] = Field(
        default_factory=list, description="Metric node names for monitoring"
    )
    freeze_nodes: list[str] = Field(
        default_factory=list,
        description="Nodes to freeze for this training run (runtime action)",
    )
    unfreeze_nodes: list[str] = Field(
        default_factory=list,
        description="Nodes to unfreeze for this training run (runtime action)",
    )
    output_dir: str = Field(default="./outputs", description="Output directory for artifacts")
    tags: dict[str, str] = Field(default_factory=dict, description="Metadata tags for tracking")

    def save_to_file(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str | Path) -> TrainRunConfig:
        """Load configuration from YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @model_validator(mode="after")
    def _validate_training_config(self) -> TrainRunConfig:
        """Ensure training config has optimizer if provided."""
        if self.training is not None and self.training.optimizer is None:
            raise ValueError("Training configuration must include optimizer when provided")
        return self


__all__ = ["TrainRunConfig"]
