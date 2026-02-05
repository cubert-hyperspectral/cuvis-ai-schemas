"""Plugin system schemas."""

from cuvis_ai_schemas.plugin.config import GitPluginConfig, LocalPluginConfig
from cuvis_ai_schemas.plugin.manifest import PluginManifest

__all__ = ["PluginManifest", "GitPluginConfig", "LocalPluginConfig"]
