"""Plugin system schemas."""

from cuvis_ai_schemas.plugin.manifest_capabilities import (
    AuxFile,
    GitPluginSource,
    HparamSpec,
    LocalPluginSource,
    NodePortSpec,
    PackagePluginSource,
    PluginCapabilities,
    PluginCapabilityEntry,
    PluginManifest,
    PluginWeightEntry,
    load_plugin_manifest,
    parse_plugin_manifest,
    write_plugin_manifest,
)

__all__ = [
    "AuxFile",
    "GitPluginSource",
    "HparamSpec",
    "LocalPluginSource",
    "NodePortSpec",
    "PackagePluginSource",
    "PluginCapabilities",
    "PluginCapabilityEntry",
    "PluginManifest",
    "PluginWeightEntry",
    "parse_plugin_manifest",
    "load_plugin_manifest",
    "write_plugin_manifest",
]
