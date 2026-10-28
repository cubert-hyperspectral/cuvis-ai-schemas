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
without importing the plugin. A manifest may also request pip ``extras`` of its
own package, installed whenever the manifest is part of a pipeline's plugin set.

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
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    field_validator,
    model_validator,
)

from cuvis_ai_schemas.base import BaseSchemaModel

# A PEP 508 extra name: letters and digits, with runs of '-', '_' or '.' between them.
_EXTRA_NAME_RE = re.compile(r"[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?")
# PEP 685: extras compare equal after lower-casing and collapsing separator runs to '-'.
_EXTRA_SEPARATORS_RE = re.compile(r"[-_.]+")


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
# 1b. HparamSpec: one declared constructor parameter of a node (leaf, no deps)
# ---------------------------------------------------------------------------
_NUMERIC_TYPES = ("int", "float")


def _matches_type(kind: str, value: object) -> bool:
    """Whether a JSON value fits an hparam ``type`` (bool is never a number here)."""
    if kind == "bool":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind == "int":
        return isinstance(value, int)
    if kind == "float":
        return isinstance(value, int | float)
    if kind == "str":
        return isinstance(value, str)
    return isinstance(value, int | float | str)


class HparamSpec(BaseSchemaModel):
    """One constructor parameter of a node, as its manifest publishes it.

    The palette and the docs read these to show a node's parameters without
    importing the node, which matters for plugins whose code the reader may not
    have. ``list`` marks a parameter whose value is a list of ``type``; ``min``,
    ``max`` and ``choices`` then apply to each item.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        populate_by_name=True,
        serialize_by_alias=True,
    )

    type: Literal["int", "float", "bool", "str", "enum"] = Field(
        description="Value type; 'enum' takes one of 'choices'."
    )
    is_list: bool = Field(default=False, alias="list", description="The value is a list of 'type'.")
    default: Any = Field(
        default=None,
        description="Default value (JSON); absent for a required parameter.",
    )
    required: bool = False
    nullable: bool = Field(default=False, description="None is an allowed value.")
    min: float | None = Field(default=None, description="Inclusive lower bound, numbers only.")
    max: float | None = Field(default=None, description="Inclusive upper bound, numbers only.")
    choices: list[Any] = Field(default_factory=list, description="Allowed values, 'enum' only.")
    description: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> HparamSpec:
        """Check bounds, choices and the default against the declared type."""
        if self.choices and self.type != "enum":
            raise ValueError("choices are only allowed for type 'enum'")
        if self.type == "enum":
            if not self.choices:
                raise ValueError("type 'enum' needs a non-empty 'choices' list")
            if len(set(map(repr, self.choices))) != len(self.choices):
                raise ValueError(f"choices must be unique, got {self.choices!r}")
        if (self.min is not None or self.max is not None) and self.type not in _NUMERIC_TYPES:
            raise ValueError("min and max only apply to types 'int' and 'float'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min} must not exceed max {self.max}")
        if self.required:
            if self.default is not None:
                raise ValueError("a required hparam has no default")
            return self
        if self.default is None:
            if not self.nullable:
                raise ValueError("an optional hparam without a default must be nullable")
            return self
        if self.is_list and not isinstance(self.default, list):
            raise ValueError(f"default of a list hparam must be a list, got {self.default!r}")
        for value in self.default if self.is_list else [self.default]:
            self._check_value(value)
        return self

    def _check_value(self, value: Any) -> None:
        """Check one default value (or list item) against type, range and choices."""
        if not _matches_type(self.type, value):
            raise ValueError(f"default {value!r} does not match type {self.type!r}")
        if self.type == "enum" and value not in self.choices:
            raise ValueError(f"default {value!r} is not one of the choices {self.choices!r}")
        if self.min is not None and value < self.min or self.max is not None and value > self.max:
            raise ValueError(f"default {value!r} is outside the range [{self.min}, {self.max}]")


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
    hparams: dict[str, HparamSpec] | None = Field(
        default=None,
        description=(
            "The node's constructor parameters by name; None when the manifest does not "
            "declare them. Required for nodes of a package source."
        ),
    )

    @field_validator("hparams")
    @classmethod
    def _validate_hparam_names(
        cls, value: dict[str, HparamSpec] | None
    ) -> dict[str, HparamSpec] | None:
        """Require Python identifiers as parameter names."""
        if value is not None:
            _require_identifiers(list(value), "hparams")
        return value

    @model_validator(mode="after")
    def _check_kind_invariant(self) -> PluginCapabilityEntry:
        """Reconcile ``kind`` with ``data_module_name`` / ``extras`` / ``hparams``.

        A ``node`` entry must carry neither a ``data_module_name`` nor ``extras``
        (so a misclassified module cannot silently leak into the node palette);
        a ``data_module`` entry must declare a non-empty ``data_module_name`` and
        no ``hparams``.
        """
        if self.kind == "node":
            if self.data_module_name or self.extras:
                raise ValueError(
                    "kind='node' entries must not set 'data_module_name' or 'extras'; pip extras "
                    "for a node plugin are declared at the manifest level ('extras:' next to 'name')."
                )
        elif not self.data_module_name:
            raise ValueError(f"kind={self.kind!r} requires a non-empty 'data_module_name'.")
        elif self.hparams is not None:
            raise ValueError(f"kind={self.kind!r} entries must not set 'hparams'.")
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

    extras: list[str] = Field(
        default_factory=list,
        description=(
            "pip extras of this plugin's package that the env composer installs whenever "
            "this manifest is in a pipeline's plugin set, united with the selected data "
            "module's extras; manifests that share one package merge into one requirement. "
            "PEP 508 extra names, normalised per PEP 685 (lower-case, runs of '-', '_' and "
            "'.' become one '-'), no duplicates. For extras that only a selected data module "
            "needs, use that capability entry's 'extras' instead."
        ),
    )

    @field_validator("extras")
    @classmethod
    def _require_extra_names(cls, value: list[str]) -> list[str]:
        """Normalise the extras per PEP 685 and reject malformed or duplicate names.

        The composer hashes ``pkg[a,b]`` into the environment cache key, so two
        spellings of one extra must not produce two environments.
        """
        normalised: list[str] = []
        for raw in value:
            if not _EXTRA_NAME_RE.fullmatch(raw):
                raise ValueError(
                    f"Invalid extra {raw!r}: an extra name is letters, digits, '-', '_' and "
                    "'.', starting and ending with a letter or digit (PEP 508)."
                )
            name = _EXTRA_SEPARATORS_RE.sub("-", raw).lower()
            if name in normalised:
                raise ValueError(f"Duplicate extra {name!r} (after normalisation).")
            normalised.append(name)
        return normalised

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


# An index name: lower-case letters and digits, with single '-' runs between them.
_INDEX_NAME_RE = re.compile(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?")
# '==' and a PEP 440 public version without wildcards, local labels or extra clauses.
_EXACT_PIN_RE = re.compile(
    r"==([0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?)"
)


class PackagePluginSource(_BasePluginManifest):
    """A plugin installed as a prebuilt package from a named package index.

    The manifest names the index; it never holds its location or credentials.
    Each host maps the index name to a registry URL or a wheel folder in its own
    configuration, so a manifest stays the same wherever the packages are served
    from. Since the package's code may not be readable, every node entry declares
    its ``hparams`` for the palette and the docs.
    """

    package_name: str = Field(
        min_length=1,
        description="The package to install, e.g. 'cuvis-ai-closed-samples'.",
    )
    version: str = Field(
        description="Exact version pin, e.g. '==0.1.0'. Ranges and wildcards are rejected.",
    )
    index: str = Field(
        description=(
            "Name of the package index the host maps to a registry URL or a wheel folder, "
            "e.g. 'cubert-closed'. A name only: never a URL, a path or a credential."
        ),
    )

    @field_validator("package_name")
    @classmethod
    def _validate_package_name(cls, value: str) -> str:
        """Require a PEP 508 project name."""
        if not _EXTRA_NAME_RE.fullmatch(value):
            raise ValueError(f"Invalid package_name {value!r} (PEP 508 project name).")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        """Require an exact '==' pin of one release."""
        if not _EXACT_PIN_RE.fullmatch(value):
            raise ValueError(
                f"Invalid version {value!r}: a package source pins one release exactly, "
                "e.g. '==0.1.0'."
            )
        return value

    @field_validator("index")
    @classmethod
    def _validate_index(cls, value: str) -> str:
        """Require an index name, which rules out URLs, paths and credentials."""
        if not _INDEX_NAME_RE.fullmatch(value):
            raise ValueError(
                f"Invalid index {value!r}: an index name is lower-case letters, digits and "
                "'-' (e.g. 'cubert-closed'), never a URL, path or credential."
            )
        return value

    @model_validator(mode="after")
    def _require_node_hparams(self) -> PackagePluginSource:
        """Require declared ``hparams`` on every node entry."""
        missing = [
            entry.class_name
            for entry in self.capabilities
            if entry.kind == "node" and entry.hparams is None
        ]
        if missing:
            raise ValueError(
                f"Every node of a package source must declare 'hparams'; missing on {missing}."
            )
        return self

    @property
    def pinned_version(self) -> str:
        """The pinned release without the '==' prefix, e.g. '0.1.0'."""
        return self.version.removeprefix("==")


PluginManifest = GitPluginSource | LocalPluginSource | PackagePluginSource
"""A single plugin manifest: a git (repo + tag), local (path) or package (index) source."""


def _manifest_source(data: object) -> str | None:
    """Pick the manifest model from the keys present, before validation.

    ``repo`` or ``tag`` selects the git model, ``index`` or ``version`` the
    package model, ``path`` the local model, so a validation error reports one
    shape instead of the errors of all three. Keys of another source then fail
    as unknown keys of the selected model.
    """
    if not isinstance(data, dict):
        return None
    if "repo" in data or "tag" in data:
        return "git"
    if "index" in data or "version" in data:
        return "package"
    return "local" if "path" in data else None


_MANIFEST_ADAPTER: TypeAdapter[PluginManifest] = TypeAdapter(
    Annotated[
        Annotated[GitPluginSource, Tag("git")]
        | Annotated[LocalPluginSource, Tag("local")]
        | Annotated[PackagePluginSource, Tag("package")],
        Discriminator(
            _manifest_source,
            custom_error_type="plugin_manifest_source",
            custom_error_message=(
                "A plugin manifest needs a source: 'path' for a local checkout, "
                "'repo' + 'tag' for a git release, or 'package_name' + 'version' + "
                "'index' for a prebuilt package."
            ),
        ),
    ]
)


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
        The validated :data:`PluginManifest` (git, local or package).

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
    for key in ("weights", "extras"):
        if not data.get(key):
            # A manifest without weights or extras is written as earlier releases wrote it.
            data.pop(key, None)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


__all__ = [
    "NodePortSpec",
    "HparamSpec",
    "PluginCapabilityEntry",
    "AuxFile",
    "PluginWeightEntry",
    "GitPluginSource",
    "LocalPluginSource",
    "PackagePluginSource",
    "PluginManifest",
    "PluginCapabilities",
    "parse_plugin_manifest",
    "load_plugin_manifest",
    "write_plugin_manifest",
]
