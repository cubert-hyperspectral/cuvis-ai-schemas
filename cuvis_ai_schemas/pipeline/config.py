"""Pipeline configuration schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from pathlib import Path

    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2


class _BaseConfig(BaseModel):
    """Base model with strict validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PipelineMetadata(_BaseConfig):
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineMetadata:
        """Load from dictionary."""
        return cls.model_validate(data)

    def to_proto(self) -> cuvis_ai_pb2.PipelineMetadata:
        """Convert to proto message.

        Requires cuvis-ai-schemas[proto] to be installed.

        Returns
        -------
        cuvis_ai_pb2.PipelineMetadata
            Proto message representation
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


class NodeConfig(_BaseConfig):
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


class ConnectionConfig(_BaseConfig):
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


class PipelineConfig(_BaseConfig):
    """Pipeline structure configuration.

    Attributes
    ----------
    name : str
        Pipeline name
    nodes : list[NodeConfig]
        Node definitions
    connections : list[ConnectionConfig]
        Node connections
    frozen_nodes : list[str]
        Node names to keep frozen during training
    metadata : PipelineMetadata | None
        Optional pipeline metadata
    """

    name: str = Field(default="", description="Pipeline name")
    nodes: list[NodeConfig] = Field(default_factory=list, description="Node definitions")
    connections: list[ConnectionConfig] = Field(default_factory=list, description="Connections")
    frozen_nodes: list[str] = Field(
        default_factory=list,
        description="Nodes frozen in the pipeline definition (static state)",
    )
    metadata: PipelineMetadata | None = Field(
        default=None, description="Optional pipeline metadata"
    )

    def to_proto(self) -> cuvis_ai_pb2.PipelineConfig:
        """Convert to proto message.

        Requires cuvis-ai-schemas[proto] to be installed.

        Returns
        -------
        cuvis_ai_pb2.PipelineConfig
            Proto message representation
        """
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as exc:
            msg = "Proto support not installed. Install with: pip install cuvis-ai-schemas[proto]"
            raise ImportError(msg) from exc

        return cuvis_ai_pb2.PipelineConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config: cuvis_ai_pb2.PipelineConfig) -> PipelineConfig:
        """Load from proto message.

        Parameters
        ----------
        proto_config : cuvis_ai_pb2.PipelineConfig
            Proto message to deserialize

        Returns
        -------
        PipelineConfig
            Loaded configuration
        """
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> PipelineConfig:
        """Load from JSON string."""
        return cls.model_validate_json(payload)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Load from dictionary."""
        return cls.model_validate(data)

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
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)

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
