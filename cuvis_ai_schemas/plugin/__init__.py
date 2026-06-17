"""Plugin system schemas."""

from cuvis_ai_schemas.plugin.manifest_capabilities import (
    SUPPORTED_SCHEMA_VERSIONS,
    GitPluginManifest,
    LocalPluginManifest,
    NodePortSpec,
    PluginCapabilities,
    PluginCapabilityEntry,
    PluginManifest,
    load_plugin_manifest,
    load_plugin_manifests,
    parse_plugin_manifest,
    write_plugin_manifest,
)

__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "GitPluginManifest",
    "LocalPluginManifest",
    "NodePortSpec",
    "PluginCapabilities",
    "PluginCapabilityEntry",
    "PluginManifest",
    "parse_plugin_manifest",
    "load_plugin_manifest",
    "load_plugin_manifests",
    "write_plugin_manifest",
]
