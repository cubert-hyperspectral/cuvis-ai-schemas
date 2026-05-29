"""Static node catalog — schema for per-plugin ``metadata.json``.

Plugins emit ``metadata.json`` at release time to describe their node
classes (port specs, category, tags, icon, doc summary). The cuvis-ai
server reads that JSON to answer the node-palette RPC without ever
importing the plugin's Python modules.

This module defines the wire shape of that JSON. Plugin repos import
these models in their CI emit scripts; the server-side loader in
cuvis-ai-core's ``orchestrator.catalog`` validates JSON against the
same models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, field_validator, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)


class CatalogPortSpec(BaseSchemaModel):
    """One port spec as recorded in metadata.json."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    dtype: str = Field(
        default="",
        description=(
            "NumPy-style dtype string (e.g. 'float32', 'uint8', 'int64'). "
            "Empty string is allowed for generic-tensor markers — proto "
            "conversion maps it to D_TYPE_UNSPECIFIED."
        ),
    )
    shape: list[int] = Field(
        default_factory=list,
        description="Dimensions; use -1 for dynamic dims (batch, height, etc.).",
    )
    optional: bool = False
    description: str = ""


class CatalogNodeEntry(BaseSchemaModel):
    """One node class as recorded in metadata.json."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    class_name: str = Field(min_length=1)
    full_path: str = Field(
        default="",
        description="Fully-qualified class path; falls back to class_name if empty.",
    )
    category: str = Field(
        default="unspecified",
        description="NodeCategory enum value, e.g. 'transform', 'source', 'sink'.",
    )
    tags: list[str] = Field(default_factory=list)
    icon_svg: str = Field(
        default="",
        description="Raw SVG XML for the node's palette icon; empty for default.",
    )
    input_specs: dict[str, list[CatalogPortSpec]] = Field(default_factory=dict)
    output_specs: dict[str, list[CatalogPortSpec]] = Field(default_factory=dict)
    doc_summary: str = ""

    @field_validator("input_specs", "output_specs", mode="before")
    @classmethod
    def _coerce_single_spec_to_list(cls, value: Any) -> Any:
        """Allow `{port: {...spec}}` as shorthand for `{port: [{...spec}]}`.

        Authors frequently write single-spec ports as a bare dict; the
        wire form is always a list so variadic ports compose naturally.
        """
        if not isinstance(value, dict):
            return value
        out: dict[str, Any] = {}
        for port_name, specs in value.items():
            if isinstance(specs, dict):
                out[port_name] = [specs]
            else:
                out[port_name] = specs
        return out

    @model_validator(mode="after")
    def _default_full_path(self) -> CatalogNodeEntry:
        """When full_path is empty, populate it from class_name."""
        if not self.full_path:
            # frozen=True blocks direct assignment; rebuild via object.__setattr__
            object.__setattr__(self, "full_path", self.class_name)
        return self


class CatalogPluginEntry(BaseSchemaModel):
    """A plugin's complete static node catalog (the metadata.json root)."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    SUPPORTED_VERSIONS: ClassVar[tuple[int, ...]] = SUPPORTED_SCHEMA_VERSIONS

    schema_version: int = Field(
        description="Catalog schema version; current writer must use the latest.",
    )
    plugin_name: str = Field(min_length=1)
    plugin_version: str = ""
    nodes: list[CatalogNodeEntry] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: int) -> int:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported schema_version={value!r}; supported: {SUPPORTED_SCHEMA_VERSIONS}"
            )
        return value

    @classmethod
    def from_metadata_file(cls, path: Path | str) -> CatalogPluginEntry:
        """Load and validate a metadata.json from disk."""
        json_path = Path(path)
        if not json_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {json_path}")
        return cls.model_validate_json(json_path.read_text(encoding="utf-8"))


__all__ = [
    "CatalogNodeEntry",
    "CatalogPluginEntry",
    "CatalogPortSpec",
    "SUPPORTED_SCHEMA_VERSIONS",
]
