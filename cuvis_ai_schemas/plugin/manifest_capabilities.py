"""Plugin manifest and capability schemas.

One yaml file is one plugin manifest. A manifest declares where the plugin
comes from (a git ``repo`` + ``tag`` or a local ``path``), its logical
``name``, and the list of ``capabilities`` it provides. Each capability is a
:class:`PluginCapabilityEntry`: an FQCN ``class_name`` plus the bucket it
registers into (``node`` or ``data_module``) and, for nodes, optional palette
metadata (port specs, category, tags, icon, doc summary).

The cuvis-ai server reads a plugin's declared capabilities to answer the
node-palette RPC without ever importing the plugin's Python modules.
:class:`PluginCapabilities` is the install-stripped view used for exactly that:
it drops the source (repo/path) and keeps only what the palette needs.

The types are declared in dependency order so the whole module is a clean DAG
(``NodePortSpec`` -> ``PluginCapabilityEntry`` -> manifests -> capabilities).
That is why :meth:`PluginCapabilities.from_manifest` can be fully typed against
:data:`PluginManifest` with no forward reference and no ``TYPE_CHECKING`` trick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel


def _require_non_empty(value: str, label: str) -> str:
    """Return ``value`` stripped, raising when it is empty after stripping."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} cannot be empty")
    return stripped


# ---------------------------------------------------------------------------
# 1. NodePortSpec — serialized spec of a single node port (leaf, no deps)
# ---------------------------------------------------------------------------
class NodePortSpec(BaseSchemaModel):
    """The JSON-serializable spec of one of a node's input/output ports.

    This is the wire sibling of the runtime ``PortSpec`` dataclass (which
    imports torch and carries symbolic shape dims, so it cannot be serialized
    directly). Only nodes have ports; a data module never carries one.
    """

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


# ---------------------------------------------------------------------------
# 2. PluginCapabilityEntry — one provided item (a node or a data module)
# ---------------------------------------------------------------------------
class PluginCapabilityEntry(BaseSchemaModel):
    """One capability a plugin provides: a node class or a data module."""

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
    input_specs: dict[str, NodePortSpec] = Field(default_factory=dict)
    output_specs: dict[str, NodePortSpec] = Field(default_factory=dict)
    doc_summary: str = ""

    @model_validator(mode="after")
    def _check_kind_invariant(self) -> PluginCapabilityEntry:
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


# ---------------------------------------------------------------------------
# 3. _BasePluginManifest — shared base: name + capabilities + package_name
# ---------------------------------------------------------------------------
class _BasePluginManifest(BaseSchemaModel):
    """Shared base for a single plugin's manifest.

    Carries the logical ``name`` (how pipelines and the catalog refer to the
    plugin), the declared ``capabilities``, and an optional ``package_name``.
    Concrete subclasses add the source: a git ``repo`` + ``tag`` or a local
    ``path``.
    """

    name: str = Field(
        min_length=1,
        description=(
            "Logical plugin name (e.g. 'sam3'). Required and explicit; it is "
            "NEVER derived from the manifest filename. Pipelines reference this "
            "name in their bare 'plugins:' list, and the directory loader keys "
            "by it. Distinct from 'package_name' (the installable, e.g. "
            "'cuvis-ai-sam3')."
        ),
    )

    capabilities: list[PluginCapabilityEntry] = Field(
        min_length=1,
        description=(
            "The capabilities this plugin provides. Each entry is one node or "
            "data module: an FQCN 'class_name' (the install + import target) "
            "plus, for nodes, optional palette metadata (category, tags, "
            "icon_svg, input/output specs, doc_summary)."
        ),
    )

    package_name: str | None = Field(
        default=None,
        description=(
            "Optional PyPI-style package name (the value of [project].name in "
            "the plugin's pyproject.toml). When it differs from the logical "
            "'name' (e.g. 'cuvis-ai-sam3' vs 'sam3') the composer needs the real "
            "name so uv's metadata check passes. Local plugins may omit it (the "
            "composer reads [project].name from pyproject.toml); git plugins "
            "default it to the logical name."
        ),
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Require ``name`` to be a valid Python identifier."""
        if not value.isidentifier():
            msg = f"Invalid plugin name '{value}'. Must be a valid Python identifier."
            raise ValueError(msg)
        return value


# ---------------------------------------------------------------------------
# 4. Concrete manifests + the PluginManifest union
# ---------------------------------------------------------------------------
class GitPluginSource(_BasePluginManifest):
    """A plugin sourced from a git repository at a fixed tag.

    Supports:
    - SSH URLs: git@gitlab.com:user/repo.git
    - HTTPS URLs: https://github.com/user/repo.git
    - Git tags only: v1.2.3, v0.1.0-alpha, etc.

    Note: branches and commit hashes are NOT supported, for reproducibility.
    """

    repo: str = Field(
        description="Git repository URL (SSH or HTTPS)",
        min_length=1,
    )

    tag: str = Field(
        description="Git tag (e.g., v1.2.3, v0.1.0-alpha). "
        "Branches and commit hashes are not supported.",
        min_length=1,
    )

    @field_validator("repo")
    @classmethod
    def _validate_repo_url(cls, value: str) -> str:
        """Validate Git repository URL format."""
        if not (
            value.startswith("git@") or value.startswith("https://") or value.startswith("http://")
        ):
            msg = f"Invalid repo URL '{value}'. Must start with 'git@', 'https://', or 'http://'"
            raise ValueError(msg)
        return value

    @field_validator("tag")
    @classmethod
    def _validate_tag(cls, value: str) -> str:
        """Validate Git tag is not empty."""
        return _require_non_empty(value, "Git tag")


class LocalPluginSource(_BasePluginManifest):
    """A plugin sourced from a local filesystem path.

    Supports:
    - Absolute paths: /home/user/my-plugin
    - Relative paths: ../my-plugin (resolved relative to the manifest file)
    - Windows paths: C:\\Users\\user\\my-plugin
    """

    path: str = Field(
        description="Absolute or relative path to plugin directory",
        min_length=1,
    )

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """Validate path is not empty."""
        return _require_non_empty(value, "Path")

    def resolve_path(self, manifest_dir: Path) -> Path:
        """Resolve a relative ``path`` to an absolute path.

        Args:
            manifest_dir: Directory containing the manifest file.

        Returns:
            Absolute path to the plugin directory.
        """
        plugin_path = Path(self.path)
        if not plugin_path.is_absolute():
            plugin_path = (manifest_dir / plugin_path).resolve()
        return plugin_path


PluginManifest = GitPluginSource | LocalPluginSource
"""A single plugin manifest: either a git (repo + tag) or local (path) source."""

_MANIFEST_ADAPTER: TypeAdapter[PluginManifest] = TypeAdapter(PluginManifest)


# ---------------------------------------------------------------------------
# 5. PluginCapabilities — install-stripped capability set for the palette
# ---------------------------------------------------------------------------
class PluginCapabilities(BaseSchemaModel):
    """A plugin's capabilities, stripped of its install source.

    Built from a :class:`PluginManifest` via :meth:`from_manifest`. The
    cuvis-ai server uses it to enumerate a plugin's node/data-module exports for
    the palette RPC without importing the plugin package.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    plugin_name: str = Field(min_length=1)
    plugin_version: str = ""
    capabilities: list[PluginCapabilityEntry] = Field(default_factory=list)

    @classmethod
    def from_manifest(cls, manifest: PluginManifest) -> PluginCapabilities | None:
        """Build a plugin's capability set from its manifest.

        Returns ``None`` when the manifest declares no capabilities, so the
        caller surfaces nothing in the palette for it. A validated manifest
        always declares at least one capability, so in practice this returns a
        populated instance; the ``None`` branch stays for caller symmetry.
        """
        if not manifest.capabilities:
            return None
        return cls(
            plugin_name=manifest.name,
            capabilities=list(manifest.capabilities),
        )


# ---------------------------------------------------------------------------
# 6. YAML + directory loaders
# ---------------------------------------------------------------------------
def parse_plugin_manifest(data: dict[str, object]) -> PluginManifest:
    """Validate an in-memory dict into the :data:`PluginManifest` union.

    Unlike :func:`load_plugin_manifest`, this does no filesystem access and no
    relative-path resolution: it is for data that is already in memory and whose
    ``path`` (if local) is already absolute, such as a stored manifest dump or a
    single element of a ``resolved_plugins_json`` list.

    Raises:
        ValueError: the data fails schema validation
            (``pydantic.ValidationError`` is a ``ValueError`` subclass).
    """
    return _MANIFEST_ADAPTER.validate_python(data)


def load_plugin_manifest(yaml_path: Path) -> PluginManifest:
    """Load and validate a single bare plugin manifest from a YAML file.

    One file is one plugin. A :class:`LocalPluginSource`'s relative ``path``
    is resolved to an absolute path against the manifest file's parent
    directory, so downstream consumers never need manifest-dir context.

    Args:
        yaml_path: Path to a single-plugin manifest YAML file.

    Returns:
        The validated :data:`PluginManifest` (git or local).

    Raises:
        FileNotFoundError: ``yaml_path`` does not exist.
        ValueError: the file is empty or fails schema validation
            (``pydantic.ValidationError`` is a ``ValueError`` subclass).
    """
    if not yaml_path.exists():
        msg = f"Plugin manifest not found: {yaml_path}"
        raise FileNotFoundError(msg)

    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        msg = (
            f"Plugin manifest is empty: {yaml_path}. A manifest must declare "
            "'name', a source ('path' or 'repo'+'tag'), and 'capabilities'."
        )
        raise ValueError(msg)

    manifest = _MANIFEST_ADAPTER.validate_python(data)
    if isinstance(manifest, LocalPluginSource):
        manifest = manifest.model_copy(
            update={"path": str(manifest.resolve_path(yaml_path.parent))}
        )
    return manifest


def write_plugin_manifest(manifest: PluginManifest, yaml_path: Path) -> None:
    """Write a single bare plugin manifest to a YAML file (no ``plugins:`` wrapper).

    Args:
        manifest: The manifest to serialize.
        yaml_path: Destination path; parent directories are created.
    """
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            manifest.model_dump(exclude_none=True, mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )


__all__ = [
    "NodePortSpec",
    "PluginCapabilityEntry",
    "GitPluginSource",
    "LocalPluginSource",
    "PluginManifest",
    "PluginCapabilities",
    "parse_plugin_manifest",
    "load_plugin_manifest",
    "write_plugin_manifest",
]
