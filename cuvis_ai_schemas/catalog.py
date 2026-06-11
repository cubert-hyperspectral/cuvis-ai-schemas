"""Static node catalog — schema for the inline plugin manifest catalog.

Each plugin manifest entry's ``provides`` list *is* its node catalog:
every item is a :class:`CatalogNodeEntry` (an FQCN ``class_name`` plus
optional palette metadata — port specs, category, tags, icon, doc
summary). The cuvis-ai server reads that inline catalog to answer the
node-palette RPC without ever importing the plugin's Python modules.

This module defines the wire shape of those entries. Release tooling
emits them into the manifest; the server-side loader in cuvis-ai-core's
``orchestrator.catalog`` validates them against the same models.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel

SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (1,)


class CatalogPortSpec(BaseSchemaModel):
    """One port spec in a node's inline catalog entry."""

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
    variadic: bool = Field(
        default=False,
        description=(
            "Input ports only: the port accepts fan-in from multiple upstream "
            "connections. Always false for outputs."
        ),
    )


class CatalogNodeEntry(BaseSchemaModel):
    """One node class as recorded in a plugin's inline manifest catalog."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    class_name: str = Field(
        min_length=1,
        description="Fully-qualified class path, e.g. 'pkg.module.MyNode'.",
    )
    kind: Literal["node", "data_module"] = Field(
        default="node",
        description="Which registry bucket this entry registers into.",
    )
    data_module_name: str = Field(
        default="",
        description=(
            "The DataModule's unique runtime name (its DATA_MODULE_NAME); "
            "empty for nodes. Globally unique across loaded plugins."
        ),
    )
    extras: list[str] = Field(
        default_factory=list,
        description=(
            "Pip extras gating this module's heavy deps (e.g. ['cu3s']); "
            "consumed by the orchestrator child-env composer. Empty for nodes."
        ),
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
    input_specs: dict[str, CatalogPortSpec] = Field(default_factory=dict)
    output_specs: dict[str, CatalogPortSpec] = Field(default_factory=dict)
    doc_summary: str = ""

    @model_validator(mode="after")
    def _check_kind_invariant(self) -> CatalogNodeEntry:
        """Reconcile ``kind`` with ``data_module_name`` / ``extras``.

        A ``node`` entry must carry neither a ``data_module_name`` nor ``extras``
        (so a misclassified module cannot silently leak into the node palette);
        a ``data_module`` entry must declare a non-empty ``data_module_name``.
        """
        if self.kind == "node":
            if self.data_module_name or self.extras:
                raise ValueError("kind='node' entries must not set 'data_module_name' or 'extras'.")
        elif not self.data_module_name:
            raise ValueError(f"kind={self.kind!r} requires a non-empty 'data_module_name'.")
        return self

    @field_validator("class_name")
    @classmethod
    def _validate_class_name(cls, value: str) -> str:
        """Require a fully-qualified dotted path of Python identifiers.

        ``class_name`` is the import target the server uses, so it must split
        into at least two dot-separated segments, each a valid Python
        identifier. Malformed forms such as ``pkg.``, ``.Node``, ``pkg..Node``,
        or ``pkg.1Node`` are rejected (not just the no-dot case).
        """
        parts = value.split(".")
        if len(parts) < 2 or not all(part.isidentifier() for part in parts):
            msg = (
                f"Invalid class path '{value}'. "
                "Must be a fully-qualified dotted path of Python identifiers "
                "(e.g., 'package.module.ClassName')."
            )
            raise ValueError(msg)
        return value


class CatalogPluginEntry(BaseSchemaModel):
    """A plugin's complete static node catalog (the manifest ``provides`` block)."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    SUPPORTED_VERSIONS: ClassVar[tuple[int, ...]] = SUPPORTED_SCHEMA_VERSIONS

    schema_version: int = Field(
        default=SUPPORTED_SCHEMA_VERSIONS[-1],
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
    def from_manifest_entry(
        cls, plugin_name: str, config_dict: dict[str, Any]
    ) -> CatalogPluginEntry | None:
        """Build a plugin's node catalog from its inline manifest entry.

        The manifest entry's ``provides`` list *is* the node catalog — each
        item is a :class:`CatalogNodeEntry` (FQCN ``class_name`` plus optional
        palette metadata). Returns ``None`` when the entry provides no nodes,
        so the caller surfaces nothing in the palette for it.
        """
        provides = config_dict.get("provides") or []
        if not provides:
            return None
        return cls(
            schema_version=config_dict.get("schema_version") or SUPPORTED_SCHEMA_VERSIONS[-1],
            plugin_name=plugin_name,
            plugin_version=config_dict.get("plugin_version", ""),
            nodes=provides,
        )


__all__ = [
    "CatalogNodeEntry",
    "CatalogPluginEntry",
    "CatalogPortSpec",
    "SUPPORTED_SCHEMA_VERSIONS",
]
