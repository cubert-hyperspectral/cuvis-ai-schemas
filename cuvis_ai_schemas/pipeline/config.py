"""Pipeline configuration schemas."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from pydantic import Field, field_validator

from cuvis_ai_schemas.base import BaseSchemaModel

if TYPE_CHECKING:
    from pathlib import Path

    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2


@lru_cache(maxsize=1)
def _schemas_version() -> str:
    """Installed cuvis-ai-schemas version, stamped into freshly created metadata."""
    from cuvis_ai_schemas import __version__

    return __version__


def _validate_endpoint(value: str, role: str, expected_middle: str) -> str:
    """Validate a ``node.{outputs,inputs}.port`` connection endpoint string.

    Parameters
    ----------
    value : str
        The endpoint string to validate.
    role : str
        ``"source"`` or ``"target"`` (used in the error message).
    expected_middle : str
        The required middle segment: ``"outputs"`` for a source, ``"inputs"``
        for a target.

    Returns
    -------
    str
        ``value`` unchanged when valid.
    """
    parts = value.split(".")
    if len(parts) != 3 or parts[1] != expected_middle or not parts[0] or not parts[2]:
        raise ValueError(f"Invalid {role}: '{value}'. Expected: 'node.{expected_middle}.port'")
    return value


def _endpoint_parts(value: str) -> tuple[str, str]:
    """Return ``(node, port)`` from a validated endpoint string."""
    parts = value.split(".")
    return parts[0], parts[2]


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
        cuvis-ai-schemas version that created the pipeline (auto-stamped from the
        installed package; an explicitly recorded value, e.g. from an older
        snapshot, is preserved on load)
    """

    name: str
    description: str = ""
    created: str = ""
    tags: list[str] = Field(default_factory=list)
    author: str = ""
    cuvis_ai_version: str = Field(default_factory=_schemas_version)

    def to_proto(self) -> cuvis_ai_pb2.PipelineMetadata:
        """Convert to proto message.

        Uses field-by-field mapping (not config_bytes) because the proto
        message has typed fields that gRPC services access directly.
        """
        from cuvis_ai_schemas.base import _get_pb2

        pb2 = _get_pb2()
        return pb2.PipelineMetadata(
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
        return _validate_endpoint(v, "source", "outputs")

    @field_validator("target")
    @classmethod
    def _validate_target(cls, v: str) -> str:
        return _validate_endpoint(v, "target", "inputs")

    @property
    def from_node(self) -> str:
        """Source node name."""
        return _endpoint_parts(self.source)[0]

    @property
    def from_port(self) -> str:
        """Source port name."""
        return _endpoint_parts(self.source)[1]

    @property
    def to_node(self) -> str:
        """Target node name."""
        return _endpoint_parts(self.target)[0]

    @property
    def to_port(self) -> str:
        """Target port name."""
        return _endpoint_parts(self.target)[1]


class PipelineConfig(BaseSchemaModel):
    """Pipeline structure configuration.

    Attributes
    ----------
    plugins : list[str] | None
        Declaration of the plugin set this pipeline depends on. Each entry is a
        bare plugin name that must resolve to a manifest yaml in the plugins
        directory. Mandatory in the loader: a pipeline that omits this field is
        rejected with a fix-it hint pointing at the ``suggest-plugins-fix`` CLI.
    nodes : list[NodeConfig]
        Node definitions
    connections : list[ConnectionConfig]
        Node connections
    metadata : PipelineMetadata | None
        Optional pipeline metadata
    """

    __proto_message__: ClassVar[str] = "PipelineConfig"

    plugins: list[str] | None = Field(
        default=None,
        description=(
            "Plugin set this pipeline depends on. Each entry is a bare plugin "
            "name (string) that must resolve to a manifest yaml in the plugins "
            "directory. Auto-resolved from class_name if omitted."
        ),
    )
    nodes: list[NodeConfig] = Field(default_factory=list, description="Node definitions")
    connections: list[ConnectionConfig] = Field(default_factory=list, description="Connections")
    metadata: PipelineMetadata | None = Field(
        default=None, description="Optional pipeline metadata"
    )

    @field_validator("plugins", mode="before")
    @classmethod
    def _validate_plugins(cls, value: object) -> object:
        """Accept only bare plugin-name strings.

        Inline (``{name, repo, tag, provides}`` / ``{name, path, provides}``) and
        tag-override (``{name, tag}``) entries are no longer supported: a plugin
        must be declared by name and resolve to a manifest yaml in the plugins
        directory. Each name must be a valid Python identifier.
        """
        if value is None:
            return value
        if not isinstance(value, list):
            msg = "'plugins' must be a list of bare plugin-name strings."
            raise ValueError(msg)
        for entry in value:
            if isinstance(entry, Mapping):
                msg = (
                    "Inline and tag-override plugin entries are no longer supported. "
                    "Each 'plugins' entry must be a bare plugin name (string) that "
                    "resolves to a manifest yaml in the plugins directory. "
                    f"Got object entry: {dict(entry)!r}."
                )
                raise ValueError(msg)
            if not isinstance(entry, str):
                msg = f"'plugins' entries must be strings, got {type(entry).__name__}."
                raise ValueError(msg)
            if not entry.isidentifier():
                msg = f"Invalid plugin name '{entry}'. Must be a valid Python identifier."
                raise ValueError(msg)
        return value

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
