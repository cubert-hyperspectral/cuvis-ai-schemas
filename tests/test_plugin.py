"""Tests for plugin manifest schemas and loaders."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from cuvis_ai_schemas.plugin import (
    GitPluginSource,
    LocalPluginSource,
    PluginCapabilities,
    PluginManifest,
    load_plugin_manifest,
    write_plugin_manifest,
)


def test_git_plugin_manifest():
    """A git manifest validates name + repo + tag + capabilities."""
    plugin = GitPluginSource(
        name="my_plugin",
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        capabilities=[{"class_name": "my.package.MyNode"}],
    )
    assert plugin.name == "my_plugin"
    assert plugin.repo == "https://github.com/user/repo.git"
    assert plugin.tag == "v1.0.0"
    assert plugin.capabilities[0].class_name == "my.package.MyNode"

    with pytest.raises(ValueError, match="Invalid repo URL"):
        GitPluginSource(
            name="my_plugin",
            repo="invalid-url",
            tag="v1.0.0",
            capabilities=[{"class_name": "my.package.MyNode"}],
        )

    with pytest.raises(ValidationError):
        GitPluginSource(
            name="my_plugin",
            repo="https://github.com/user/repo.git",
            tag="",
            capabilities=[{"class_name": "my.package.MyNode"}],
        )


def test_capabilities_is_node_list():
    """`capabilities` entries carry class_name (FQCN) plus optional palette metadata."""
    plugin = GitPluginSource(
        name="my_plugin",
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        capabilities=[
            {
                "class_name": "my.package.MyNode",
                "category": "model",
                "tags": ["rgb"],
                "input_specs": {
                    "x": {"dtype": "float32", "shape": [-1, -1, -1, 3], "variadic": True}
                },
            }
        ],
    )
    node = plugin.capabilities[0]
    assert node.category == "model"
    assert node.tags == ["rgb"]
    assert node.input_specs["x"].dtype == "float32"
    assert node.input_specs["x"].variadic is True

    # Cold break: bare-string capability entries are not accepted.
    with pytest.raises(ValidationError):
        GitPluginSource(
            name="my_plugin",
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            capabilities=["my.package.MyNode"],
        )

    # class_name must be a fully-qualified (dotted) path.
    with pytest.raises(ValueError, match="Invalid class path"):
        GitPluginSource(
            name="my_plugin",
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            capabilities=[{"class_name": "NotDotted"}],
        )


def test_capabilities_required_min_length():
    """A manifest must declare at least one capability."""
    with pytest.raises(ValidationError):
        LocalPluginSource(name="my_plugin", path="/p", capabilities=[])


def test_local_plugin_manifest():
    """A local manifest validates name + path + capabilities."""
    plugin = LocalPluginSource(
        name="my_plugin",
        path="/path/to/plugin",
        capabilities=[{"class_name": "my.package.MyNode"}],
    )
    assert plugin.path == "/path/to/plugin"

    with pytest.raises(ValidationError):
        LocalPluginSource(
            name="my_plugin", path="", capabilities=[{"class_name": "my.package.MyNode"}]
        )


def test_name_is_required_and_must_be_identifier():
    """`name` is required and must be a valid Python identifier (never the filename)."""
    with pytest.raises(ValidationError):
        LocalPluginSource(path="/p", capabilities=[{"class_name": "my.package.MyNode"}])

    with pytest.raises(ValueError, match="Invalid plugin name"):
        LocalPluginSource(
            name="invalid-name",
            path="/p",
            capabilities=[{"class_name": "my.package.MyNode"}],
        )


def test_plugin_manifest_to_dict_round_trip():
    """Manifests survive to_dict → from_dict round-trip."""
    git_plugin = GitPluginSource(
        name="git_plugin",
        repo="git@gitlab.com:user/repo.git",
        tag="v2.0.0",
        capabilities=[
            {"class_name": "my.package.MyNode"},
            {"class_name": "my.package.OtherNode"},
        ],
    )
    restored = GitPluginSource.from_dict(git_plugin.to_dict())
    assert restored.name == git_plugin.name
    assert restored.repo == git_plugin.repo
    assert restored.tag == git_plugin.tag
    assert restored.capabilities == git_plugin.capabilities

    local_plugin = LocalPluginSource(
        name="local_plugin",
        path="/opt/plugins/my-plugin",
        capabilities=[{"class_name": "local.package.LocalNode"}],
    )
    restored_local = LocalPluginSource.from_dict(local_plugin.to_dict())
    assert restored_local.name == local_plugin.name
    assert restored_local.path == local_plugin.path
    assert restored_local.capabilities == local_plugin.capabilities


def test_package_name_optional_override():
    """`package_name` is an optional author override: defaults to None and round-trips."""
    git_plugin = GitPluginSource(
        name="my_plugin",
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        capabilities=[{"class_name": "my.package.MyNode"}],
    )
    assert git_plugin.package_name is None

    named = LocalPluginSource(
        name="sam3",
        path="/opt/plugins/sam3",
        package_name="cuvis-ai-sam3",
        capabilities=[{"class_name": "cuvis_ai_sam3.node.Sam3"}],
    )
    assert named.package_name == "cuvis-ai-sam3"
    assert LocalPluginSource.from_dict(named.to_dict()).package_name == "cuvis-ai-sam3"


def test_manifest_rejects_unknown_key():
    """extra='forbid' carries over from BaseSchemaModel — stray keys are rejected."""
    with pytest.raises(ValidationError):
        GitPluginSource(
            name="my_plugin",
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            capabilities=[{"class_name": "my.package.MyNode"}],
            bogus_field="nope",
        )


def test_plugin_manifest_union_export():
    """The PluginManifest union alias is exported for consumers that need the type."""
    git = GitPluginSource(
        name="g",
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        capabilities=[{"class_name": "my.package.MyNode"}],
    )
    local = LocalPluginSource(
        name="loc", path="/p", capabilities=[{"class_name": "my.package.MyNode"}]
    )
    assert isinstance(git, PluginManifest)
    assert isinstance(local, PluginManifest)


def _write_yaml(tmp_path, filename: str, data: dict):
    p = tmp_path / filename
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return p


def test_load_plugin_manifest_bare_shape(tmp_path):
    """A bare manifest file (no `plugins:` wrapper) loads into the union type."""
    path = _write_yaml(
        tmp_path,
        "sam3.yaml",
        {
            "name": "sam3",
            "repo": "https://github.com/user/repo.git",
            "tag": "v1.0.0",
            "capabilities": [{"class_name": "cuvis_ai_sam3.node.Sam3"}],
        },
    )
    manifest = load_plugin_manifest(path)
    assert isinstance(manifest, GitPluginSource)
    assert manifest.name == "sam3"


def test_load_plugin_manifest_resolves_local_path(tmp_path):
    """A local manifest's relative path resolves to absolute against its parent dir."""
    path = _write_yaml(
        tmp_path,
        "local.yaml",
        {
            "name": "local",
            "path": "../sibling",
            "capabilities": [{"class_name": "pkg.mod.Node"}],
        },
    )
    manifest = load_plugin_manifest(path)
    assert isinstance(manifest, LocalPluginSource)
    expected = (tmp_path / ".." / "sibling").resolve()
    assert manifest.path == str(expected)


def test_load_plugin_manifest_empty_file_errors(tmp_path):
    """An empty manifest file raises a clear error, not a silent empty manifest."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_plugin_manifest(path)


def test_write_then_load_round_trip(tmp_path):
    """write_plugin_manifest emits a bare file that load_plugin_manifest reads back."""
    manifest = GitPluginSource(
        name="rt",
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        capabilities=[{"class_name": "pkg.mod.Node"}],
    )
    path = tmp_path / "rt.yaml"
    write_plugin_manifest(manifest, path)
    # No `plugins:` wrapper on disk.
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "plugins" not in raw
    assert raw["name"] == "rt"
    restored = load_plugin_manifest(path)
    assert restored == manifest


def test_from_manifest_builds_capabilities():
    """PluginCapabilities.from_manifest strips the source, keeps name + capabilities."""
    manifest = LocalPluginSource(
        name="my_plugin",
        path="/p",
        capabilities=[
            {"class_name": "my_plugin.node.MyNode", "category": "model"},
            {
                "class_name": "my_plugin.data.Cu3sDataModule",
                "kind": "data_module",
                "data_module_name": "cu3s",
                "extras": ["cu3s"],
            },
        ],
    )
    caps = PluginCapabilities.from_manifest(manifest)
    assert caps is not None
    assert caps.plugin_name == "my_plugin"
    assert len(caps.capabilities) == 2
    dm = [c for c in caps.capabilities if c.kind == "data_module"]
    assert dm[0].data_module_name == "cu3s"
