"""Data configuration schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError:
        cuvis_ai_pb2 = None  # type: ignore[assignment]


class DataConfig(BaseModel):
    """Data loading configuration."""

    cu3s_file_path: str = Field(description="Path to .cu3s file")
    annotation_json_path: str | None = Field(
        default=None, description="Path to annotation JSON (optional)"
    )
    train_ids: list[int] = Field(default_factory=list, description="Training sample IDs")
    val_ids: list[int] = Field(default_factory=list, description="Validation sample IDs")
    test_ids: list[int] = Field(default_factory=list, description="Test sample IDs")
    train_split: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Training split ratio"
    )
    val_split: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Validation split ratio"
    )
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    batch_size: int = Field(default=1, ge=1, description="Batch size")
    processing_mode: str = Field(default="Reflectance", description="Raw or Reflectance mode")

    model_config = ConfigDict(extra="forbid", validate_assignment=True, populate_by_name=True)

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.DataConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        """JSON serialization helper for legacy callers."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> DataConfig:
        """Create from JSON string."""
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Create from dictionary."""
        return cls.model_validate(data)


__all__ = ["DataConfig"]
