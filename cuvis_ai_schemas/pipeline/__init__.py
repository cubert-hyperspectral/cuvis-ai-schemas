"""Pipeline structure schemas."""

from cuvis_ai_schemas.pipeline.config import (
    CatalogPluginRef,
    ConnectionConfig,
    InlineGitPluginRef,
    InlineLocalPluginRef,
    NodeConfig,
    PipelineConfig,
    PipelineMetadata,
    PluginRef,
)
from cuvis_ai_schemas.pipeline.exceptions import PortCompatibilityError
from cuvis_ai_schemas.pipeline.ports import (
    DimensionResolver,
    InputPort,
    OutputPort,
    PortSpec,
)
from cuvis_ai_schemas.pipeline.profiling import NodeProfilingStats

__all__ = [
    "CatalogPluginRef",
    "ConnectionConfig",
    "DimensionResolver",
    "InlineGitPluginRef",
    "InlineLocalPluginRef",
    "InputPort",
    "NodeConfig",
    "NodeProfilingStats",
    "OutputPort",
    "PipelineConfig",
    "PipelineMetadata",
    "PluginRef",
    "PortCompatibilityError",
    "PortSpec",
]
