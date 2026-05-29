"""Pipeline configuration schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from pydantic import Field, field_validator

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.plugin.config import GitPluginConfig, LocalPluginConfig

if TYPE_CHECKING:
    from pathlib import Path

    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2


class PipelineMetadata(BaseSchemaModel):
    """Pipeline metadata for documentation and discovery.

    Attributes
    ----------
    name : str
        Pipeline name
    description : str
        Human-readable description
    created : str
        Creation timestamp (ISO format)
    tags : list[str]
        Tags for categorization and search
    author : str
        Author name or email
    cuvis_ai_version : str
        Version of cuvis-ai-schemas used
    """

    name: str
    description: str = ""
    created: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    cuvis_ai_version: str = "0.1.0"

    def to_proto(self) -> cuvis_ai_pb2.PipelineMetadata:
        """Convert to proto message.

        Uses field-by-field mapping (not config_bytes) because the proto
        message has typed fields that gRPC services access directly.
        """
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as exc:
            msg = "Proto support not installed. Install with: pip install cuvis-ai-schemas[proto]"
            raise ImportError(msg) from exc

        return cuvis_ai_pb2.PipelineMetadata(
            name=self.name,
            description=self.description,
            created=self.created,
            tags=list(self.tags),
            author=self.author,
            cuvis_ai_version=self.cuvis_ai_version,
        )


class NodeConfig(BaseSchemaModel):
    """Node configuration within a pipeline.

    Attributes
    ----------
    name : str
        Node identifier / base name
    class_name : str
        Fully-qualified class name (e.g., 'my_package.MyNode')
    hparams : dict[str, Any]
        Node hyperparameters
    """

    name: str = Field(description="Node identifier / base name")
    class_name: str = Field(description="Fully-qualified class name")
    hparams: dict[str, Any] = Field(default_factory=dict, description="Node hyperparameters")


class ConnectionConfig(BaseSchemaModel):
    """Connection between two nodes using compact string format.

    Attributes
    ----------
    source : str
        Source endpoint in format ``"node.outputs.port"``
    target : str
        Target endpoint in format ``"node.inputs.port"``
    """

    source: str = Field(description='Source: "node.outputs.port"')
    target: str = Field(description='Target: "node.inputs.port"')

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or parts[1] != "outputs":
            raise ValueError(f"Invalid source: '{v}'. Expected: 'node.outputs.port'")
        return v

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) != 3 or parts[1] != "inputs":
            raise ValueError(f"Invalid target: '{v}'. Expected: 'node.inputs.port'")
        return v

    @property
    def from_node(self) -> str:
        """Source node name."""
        return self.source.split(".")[0]

    @property
    def from_port(self) -> str:
        """Source port name."""
        return self.source.split(".")[2]

    @property
    def to_node(self) -> str:
        """Target node name."""
        return self.target.split(".")[0]

    @property
    def to_port(self) -> str:
        """Target port name."""
        return self.target.split(".")[2]


class CatalogPluginRef(BaseSchemaModel):
    """Reference to a plugin defined in the session's plugin catalog.

    Object-form alternative to a bare-string entry in ``PipelineConfig.plugins``.
    Use this form when you want to override the catalog entry's ``tag``;
    otherwise prefer the shorter bare-string form (``- my_plugin``).
    """

    name: str = Field(
        description="Plugin name (must be a valid Python identifier).",
        min_length=1,
    )
    tag: str | None = Field(
        default=None,
        description="Optional Git tag override for the catalog entry. "
        "Only meaningful when the catalog entry is a Git plugin.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Reject plugin names that are not valid Python identifiers."""
        if not value.isidentifier():
            msg = f"Invalid plugin name '{value}'. Must be a valid Python identifier"
            raise ValueError(msg)
        return value


class InlineGitPluginRef(GitPluginConfig):
    """Inline Git plugin entry embedded in a pipeline YAML.

    Same fields as :class:`cuvis_ai_schemas.plugin.config.GitPluginConfig`
    plus a ``name`` field (the manifest form uses dict keys for the name;
    the pipeline-yaml form embeds it inline).
    """

    name: str = Field(
        description="Plugin name (must be a valid Python identifier).",
        min_length=1,
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Reject plugin names that are not valid Python identifiers."""
        if not value.isidentifier():
            msg = f"Invalid plugin name '{value}'. Must be a valid Python identifier"
            raise ValueError(msg)
        return value


class InlineLocalPluginRef(LocalPluginConfig):
    """Inline local-path plugin entry embedded in a pipeline YAML.

    Same fields as :class:`cuvis_ai_schemas.plugin.config.LocalPluginConfig`
    plus a ``name`` field.
    """

    name: str = Field(
        description="Plugin name (must be a valid Python identifier).",
        min_length=1,
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        """Reject plugin names that are not valid Python identifiers."""
        if not value.isidentifier():
            msg = f"Invalid plugin name '{value}'. Must be a valid Python identifier"
            raise ValueError(msg)
        return value


# A plugin reference in a pipeline YAML's ``plugins:`` list.
#
# Discriminated by shape (BaseSchemaModel has ``extra="forbid"`` so each
# form only validates against the model whose fields match exactly):
#
# - ``str`` — bare-name reference to a catalog entry.
# - :class:`InlineGitPluginRef` — ``{name, repo, tag, provides}`` self-contained git plugin.
# - :class:`InlineLocalPluginRef` — ``{name, path, provides}`` self-contained local plugin.
# - :class:`CatalogPluginRef` — ``{name}`` or ``{name, tag}`` object-form catalog reference.
PluginRef = str | InlineGitPluginRef | InlineLocalPluginRef | CatalogPluginRef


class PipelineConfig(BaseSchemaModel):
    """Pipeline structure configuration.

    Attributes
    ----------
    plugins : list[PluginRef] | None
        Declaration of the plugin set this pipeline depends on. Mandatory
        in the loader: a pipeline that omits this field is rejected with a
        fix-it hint pointing at the ``suggest-plugins-fix`` CLI.
    nodes : list[NodeConfig]
        Node definitions
    connections : list[ConnectionConfig]
        Node connections
    metadata : PipelineMetadata | None
        Optional pipeline metadata
    """

    __proto_message__: ClassVar[str] = "PipelineConfig"

    plugins: list[PluginRef] | None = Field(
        default=None,
        description=(
            "Plugin set this pipeline depends on. Each entry is a bare catalog "
            "name (string), a catalog ref with optional tag override "
            "({name, tag}), or an inline self-contained entry "
            "({name, repo, tag, provides} for Git or {name, path, provides} "
            "for local). Auto-resolved from class_name if omitted."
        ),
    )
    nodes: list[NodeConfig] = Field(default_factory=list, description="Node definitions")
    connections: list[ConnectionConfig] = Field(default_factory=list, description="Connections")
    metadata: PipelineMetadata | None = Field(
        default=None, description="Optional pipeline metadata"
    )

    def save_to_file(self, path: str | Path) -> None:
        """Save pipeline configuration to YAML file.

        Parameters
        ----------
        path : str | Path
            Output file path
        """
        from pathlib import Path

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str | Path) -> PipelineConfig:
        """Load pipeline configuration from YAML file.

        Parameters
        ----------
        path : str | Path
            Input file path

        Returns
        -------
        PipelineConfig
            Loaded configuration
        """
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
