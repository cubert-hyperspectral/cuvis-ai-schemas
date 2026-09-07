"""Plugin manifest and capability schemas.

One yaml file is one plugin manifest. A manifest declares where the plugin
comes from (a git ``repo`` + ``tag`` or a local ``path``), its logical
``name``, the list of ``capabilities`` it provides and, optionally, the model
``weights`` those capabilities need. Each capability is a
:class:`PluginCapabilityEntry`: an FQCN ``class_name`` plus the bucket it
registers into (``node`` or ``data_module``) and, for nodes, optional palette
metadata (port specs, category, tags, icon, doc summary). Each weight is a
:class:`PluginWeightEntry`: one pinned file in a public Hugging Face mirror
(repo id, revision, sha256, size) plus what it is used for and which node
hyper-parameter selects it, so a consumer can provision and check the weights
without importing the plugin.

The cuvis-ai server reads a plugin's declared capabilities to answer the
node-palette RPC without ever importing the plugin's Python modules.
:class:`PluginCapabilities` is the install-stripped view used for exactly that:
it drops the source (repo/path) and keeps only what the palette needs.

The types are declared in dependency order so the whole module is a clean DAG
(``NodePortSpec`` -> ``PluginCapabilityEntry``, ``AuxFile`` -> ``PluginWeightEntry``
-> manifests -> capabilities).
That is why :meth:`PluginCapabilities.from_manifest` can be fully typed against
:data:`PluginManifest` with no forward reference and no ``TYPE_CHECKING`` trick.
"""

from __future__ import annotations

import re
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
# 2b. AuxFile + PluginWeightEntry: one pinned model file a plugin needs
# ---------------------------------------------------------------------------
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_WEIGHT_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+")


def _require_hex(value: str, label: str, pattern: re.Pattern[str], digits: int) -> str:
    """Require exactly ``digits`` lowercase hex digits (a git sha or a sha256)."""
    if not pattern.fullmatch(value):
        raise ValueError(f"{label} must be {digits} lowercase hex digits, got {value!r}")
    return value


def _require_repo_relative_path(value: str, label: str) -> str:
    """Require a relative, ``/``-separated path inside a mirror repo with no odd segments."""
    stripped = _require_non_empty(value, label)
    if stripped.startswith("/") or "\\" in stripped:
        raise ValueError(f"{label} must be a relative path with '/' separators, got {value!r}")
    if any(part in ("", ".", "..") for part in stripped.split("/")):
        raise ValueError(f"{label} must not contain empty, '.' or '..' segments, got {value!r}")
    return stripped


def _require_weight_key(value: str, label: str) -> str:
    """Require a registry key: letters, digits, ``_``, ``.`` and ``-``, no whitespace."""
    if not _WEIGHT_KEY.fullmatch(value):
        raise ValueError(
            f"{label} {value!r} must start with a letter or digit and contain only "
            "letters, digits, '_', '.' and '-'."
        )
    return value


def _require_identifiers(values: list[str], label: str) -> list[str]:
    """Require unique Python identifiers (node hyper-parameter names)."""
    for value in values:
        if not value.isidentifier():
            raise ValueError(f"{label} entries must be Python identifiers, got {value!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} entries must be unique, got {values!r}")
    return values


class AuxFile(BaseSchemaModel):
    """A file fetched beside a weight's primary file, at the same pinned revision.

    The pipeline yaml of a trained pipeline, or a config the loader reads next
    to the checkpoint. Pinned exactly like the primary file so a consumer can
    verify its presence and integrity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    path: str = Field(
        min_length=1,
        description="Path inside the mirror repo, '/'-separated, relative to the repo root.",
    )
    size_bytes: int = Field(gt=0, description="Size of the file in bytes.")
    sha256: str = Field(description="sha256 of the file, 64 lowercase hex digits.")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """Require a relative in-repo path."""
        return _require_repo_relative_path(value, "aux_files[].path")

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        """Require 64 lowercase hex digits."""
        return _require_hex(value, "aux_files[].sha256", _HEX64, 64)


class PluginWeightEntry(BaseSchemaModel):
    """One model weight a plugin needs: a pinned file in a public Hugging Face mirror.

    ``name`` is the registry key (``download-model download <name>``). Where a
    node hyper-parameter chooses between variants, ``selected_by`` names that
    hyper-parameter, ``default`` marks the row a pipeline gets without setting
    it, and ``aliases`` are the other values that pick this row; a plugin whose
    nodes always need the weight leaves ``selected_by`` unset. Names and aliases
    share one namespace across every plugin a consumer loads.

    ``kind='trained_pipeline'`` marks a Cubert-trained pipeline (a ``.pt`` plus
    its pipeline yaml as an aux file): offered for download, never required by
    a shipped preset, and therefore never selected by a hyper-parameter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)

    name: str = Field(
        min_length=1,
        description=(
            "Registry key, e.g. 'efficienttam_s'. Unique across all plugins' names and aliases."
        ),
    )
    display_name: str = Field(
        min_length=1,
        description="User-facing name, e.g. 'RTSAM (EfficientTAM small)'.",
    )
    summary: str = Field(
        default="",
        max_length=60,
        description="Plain-language one-liner (at most 60 characters) shown under the name.",
    )
    used_for: list[str] = Field(
        min_length=1,
        description="Short feature labels the weight enables, e.g. 'Point expansion'.",
    )
    kind: Literal["weights", "trained_pipeline"] = Field(
        default="weights",
        description=(
            "'weights': a checkpoint a plugin's node loads. 'trained_pipeline': a trained "
            "pipeline (.pt + its yaml) that a user points the pipeline picker at."
        ),
    )
    repo_id: str = Field(description="Hugging Face repo id, e.g. 'cubert-gmbh/sam3'.")
    filename: str = Field(
        min_length=1,
        description="Primary file inside the repo, '/'-separated, relative to the repo root.",
    )
    revision: str = Field(description="Pinned repo commit, 40 lowercase hex digits.")
    sha256: str = Field(description="sha256 of the primary file, 64 lowercase hex digits.")
    size_bytes: int = Field(gt=0, description="Size of the primary file in bytes.")
    aux_files: list[AuxFile] = Field(
        default_factory=list,
        description="Files fetched beside 'filename' at the same revision.",
    )
    license: str = Field(
        min_length=1,
        description="Licence label shown to users, e.g. 'Apache-2.0' or 'SAM License'.",
    )
    license_file: str | None = Field(
        default=None,
        description=(
            "Licence text file in the mirror repo (a bare filename such as 'LICENSE'); "
            "None when upstream states no licence for the weights."
        ),
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Other hyper-parameter values that select this row, e.g. 'efficienttam'.",
    )
    selected_by: str | None = Field(
        default=None,
        description=(
            "Node hyper-parameter whose value picks this row, e.g. 'model_type'; None when "
            "the plugin always needs this weight."
        ),
    )
    default: bool = Field(
        default=False,
        description="The row a pipeline gets when it does not set 'selected_by'.",
    )
    explicit_path_hparams: list[str] = Field(
        default_factory=list,
        description=(
            "Node hyper-parameters that bypass the model cache for this plugin, e.g. "
            "'checkpoint_path'; a pipeline setting one of them needs no download."
        ),
    )
    description: str = Field(
        default="",
        description="One sentence on what the weight is needed for, plus its provenance.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Require a registry-key shaped name."""
        return _require_weight_key(value, "Weight name")

    @field_validator("aliases")
    @classmethod
    def _validate_aliases(cls, values: list[str]) -> list[str]:
        """Require registry-key shaped, unique aliases."""
        for value in values:
            _require_weight_key(value, "Weight alias")
        if len(set(values)) != len(values):
            raise ValueError(f"aliases must be unique, got {values!r}")
        return values

    @field_validator("used_for")
    @classmethod
    def _validate_used_for(cls, values: list[str]) -> list[str]:
        """Require non-empty, unique labels."""
        stripped = [_require_non_empty(value, "used_for label") for value in values]
        if len(set(stripped)) != len(stripped):
            raise ValueError(f"used_for labels must be unique, got {values!r}")
        return stripped

    @field_validator("repo_id")
    @classmethod
    def _validate_repo_id(cls, value: str) -> str:
        """Require the 'owner/name' shape of a Hugging Face repo id."""
        if not _REPO_ID.fullmatch(value):
            raise ValueError(f"repo_id must look like 'owner/name', got {value!r}")
        return value

    @field_validator("filename")
    @classmethod
    def _validate_filename(cls, value: str) -> str:
        """Require a relative in-repo path."""
        return _require_repo_relative_path(value, "filename")

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: str) -> str:
        """Require a full 40-digit commit sha."""
        return _require_hex(value, "revision", _HEX40, 40)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        """Require 64 lowercase hex digits."""
        return _require_hex(value, "sha256", _HEX64, 64)

    @field_validator("license_file")
    @classmethod
    def _validate_license_file(cls, value: str | None) -> str | None:
        """Require a bare filename when set."""
        if value is None:
            return None
        stripped = _require_non_empty(value, "license_file")
        if "/" in stripped or "\\" in stripped:
            raise ValueError(
                f"license_file must be a bare filename in the mirror repo, got {value!r}"
            )
        return stripped

    @field_validator("selected_by")
    @classmethod
    def _validate_selected_by(cls, value: str | None) -> str | None:
        """Require a Python identifier when set (it names a node hyper-parameter)."""
        if value is not None and not value.isidentifier():
            raise ValueError(f"selected_by must be a hyper-parameter name, got {value!r}")
        return value

    @field_validator("explicit_path_hparams")
    @classmethod
    def _validate_explicit_path_hparams(cls, values: list[str]) -> list[str]:
        """Require unique Python identifiers (node hyper-parameter names)."""
        return _require_identifiers(values, "explicit_path_hparams")

    @model_validator(mode="after")
    def _check_selection_invariants(self) -> PluginWeightEntry:
        """Reconcile the selection fields with each other and with ``kind``.

        An alias must not repeat the name; aux paths must be unique and differ
        from the primary file; ``default`` only means something together with
        ``selected_by``; a trained pipeline is never picked by a hyper-parameter.
        """
        if self.name in self.aliases:
            raise ValueError(f"alias {self.name!r} repeats the weight's own name")
        aux_paths = [aux.path for aux in self.aux_files]
        if len(set(aux_paths)) != len(aux_paths):
            raise ValueError(f"aux_files paths must be unique, got {aux_paths!r}")
        if self.filename in aux_paths:
            raise ValueError(f"aux_files must not repeat the primary file {self.filename!r}")
        if self.kind == "trained_pipeline" and (self.selected_by is not None or self.default):
            raise ValueError(
                "kind='trained_pipeline' rows are never selected by a hyper-parameter: "
                "leave 'selected_by' unset and 'default' false."
            )
        if self.default and self.selected_by is None:
            raise ValueError(
                "default=True requires 'selected_by' (the hyper-parameter whose absence "
                "picks this row)"
            )
        return self


# ---------------------------------------------------------------------------
# 3. _BasePluginManifest — shared base: name + capabilities + weights + package_name
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

    weights: list[PluginWeightEntry] = Field(
        default_factory=list,
        description=(
            "The model weights this plugin's capabilities need, one pinned mirror file "
            "per entry; empty for plugins without downloadable weights. Projected from the "
            "plugin's own declaration by cuvis-ai-core's emit_metadata, and read by the "
            "weight registry and the installer without importing the plugin."
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

    @model_validator(mode="after")
    def _check_weight_namespace(self) -> _BasePluginManifest:
        """Require one namespace over weight names and aliases within the manifest.

        A consumer resolves a hyper-parameter value to a row by name or alias,
        so a key that two rows claim would make the choice order-dependent.
        """
        owner_by_key: dict[str, str] = {}
        for entry in self.weights:
            for key in (entry.name, *entry.aliases):
                if key in owner_by_key:
                    raise ValueError(
                        f"weights: {key!r} is declared by both {owner_by_key[key]!r} and "
                        f"{entry.name!r}; names and aliases share one namespace."
                    )
                owner_by_key[key] = entry.name
        return self


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
    data = manifest.model_dump(exclude_none=True, mode="json")
    if not data.get("weights"):
        # A manifest without weights is written exactly as earlier releases wrote it.
        data.pop("weights", None)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


__all__ = [
    "NodePortSpec",
    "PluginCapabilityEntry",
    "AuxFile",
    "PluginWeightEntry",
    "GitPluginSource",
    "LocalPluginSource",
    "PluginManifest",
    "PluginCapabilities",
    "parse_plugin_manifest",
    "load_plugin_manifest",
    "write_plugin_manifest",
]
