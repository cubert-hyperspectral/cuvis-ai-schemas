"""Training run configuration schema."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.data import DataConfig

if TYPE_CHECKING:
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError:
        cuvis_ai_pb2 = None  # type: ignore[assignment]


class PipelineMetadata(BaseModel):
    """Pipeline metadata for documentation and discovery."""

    name: str
    description: str = ""
    created: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    cuvis_ai_version: str = Field(default="0.1.0")

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineMetadata:
        """Create from dictionary."""
        return cls.model_validate(data)

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.PipelineMetadata(
            name=self.name,
            description=self.description,
            created=self.created,
            tags=list(self.tags),
            author=self.author,
            cuvis_ai_version=self.cuvis_ai_version,
        )


class PipelineConfig(BaseModel):
    """Pipeline structure configuration."""

    name: str = Field(default="", description="Pipeline name")
    nodes: list[dict[str, Any]] = Field(description="Node definitions")
    connections: list[dict[str, Any]] = Field(description="Node connections")
    frozen_nodes: list[str] = Field(
        default_factory=list, description="Node names to keep frozen during training"
    )
    metadata: PipelineMetadata | None = Field(
        default=None, description="Optional pipeline metadata"
    )

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.PipelineConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        """JSON serialization helper for legacy callers."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> PipelineConfig:
        """Create from JSON string."""
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create from dictionary."""
        return cls.model_validate(data)


class TrainRunConfig(BaseModel):
    """Complete reproducible training configuration."""

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
        default_factory=list, description="Node names to keep frozen during training"
    )
    unfreeze_nodes: list[str] = Field(
        default_factory=list, description="Node names to unfreeze during training"
    )
    output_dir: str = Field(default="./outputs", description="Output directory for artifacts")
    tags: dict[str, str] = Field(default_factory=dict, description="Metadata tags for tracking")

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.TrainRunConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        """JSON serialization helper for legacy callers."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> TrainRunConfig:
        """Create from JSON string."""
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainRunConfig:
        """Create from dictionary."""
        return cls.model_validate(data)

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


__all__ = ["PipelineMetadata", "PipelineConfig", "TrainRunConfig"]
