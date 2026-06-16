"""Training run configuration schema."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import yaml
from pydantic import Field, field_validator, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.data import DataConfig


class TrainRunConfig(BaseSchemaModel):
    """Complete reproducible training configuration."""

    __proto_message__: ClassVar[str] = "TrainRunConfig"

    name: str = Field(description="Train run identifier")
    pipeline: str | None = Field(
        default=None,
        description=(
            "Reference to the pipeline YAML this run trains. A path resolved "
            "relative to the trainrun file's directory (a bare name or short "
            "path also resolves against the pipeline search path). The pipeline "
            "is authored once and referenced here, never inlined."
        ),
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
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str | Path) -> TrainRunConfig:
        """Load configuration from YAML file."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @field_validator("pipeline", mode="before")
    @classmethod
    def _reject_inline_pipeline(cls, value: object) -> object:
        """Reject an inline pipeline mapping; ``pipeline`` is a reference, not a config.

        A trainrun references its pipeline by path; the pipeline's ``nodes`` /
        ``connections`` are authored in a separate pipeline YAML. An inline
        mapping (the legacy embedded ``PipelineConfig`` shape, or a Hydra
        ``@pipeline`` group composition) is rejected with a fix-it hint.
        """
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            msg = (
                "Inline pipelines are no longer supported in a trainrun. Save the "
                "pipeline to its own YAML and set 'pipeline:' to its path "
                "(resolved relative to the trainrun file). Got an inline mapping "
                f"with keys {sorted(value)!r}."
            )
            raise ValueError(msg)
        raise ValueError(
            "'pipeline' must be a path string referencing a pipeline YAML, "
            f"got {type(value).__name__}."
        )

    @model_validator(mode="after")
    def _validate_training_config(self) -> TrainRunConfig:
        """Ensure training config has optimizer if provided."""
        if self.training is not None and self.training.optimizer is None:
            raise ValueError("Training configuration must include optimizer when provided")
        return self


__all__ = ["TrainRunConfig"]
