"""Data configuration schema."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from cuvis_ai_schemas.base import BaseSchemaModel


class DataConfig(BaseSchemaModel):
    """Data loading configuration."""

    __proto_message__: ClassVar[str] = "DataConfig"

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


__all__ = ["DataConfig"]
