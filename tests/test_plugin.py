"""Tests for plugin schemas."""

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.plugin import (
    GitPluginConfig,
    LocalPluginConfig,
    PluginConfig,
    PluginManifest,
)


def test_git_plugin_config():
    """Test GitPluginConfig validation."""
    plugin = GitPluginConfig(
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        provides=[{"class_name": "my.package.MyNode"}],
    )
    assert plugin.repo == "https://github.com/user/repo.git"
    assert plugin.tag == "v1.0.0"
    assert plugin.provides[0].class_name == "my.package.MyNode"

    # Test repo validation
    with pytest.raises(ValueError, match="Invalid repo URL"):
        GitPluginConfig(
            repo="invalid-url",
            tag="v1.0.0",
            provides=[{"class_name": "my.package.MyNode"}],
        )

    # Test tag validation
    with pytest.raises(ValidationError):
        GitPluginConfig(
            repo="https://github.com/user/repo.git",
            tag="",
            provides=[{"class_name": "my.package.MyNode"}],
        )


def test_provides_is_node_list():
    """`provides` entries carry class_name (FQCN) plus optional palette metadata."""
    plugin = GitPluginConfig(
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        provides=[
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
    node = plugin.provides[0]
    assert node.category == "model"
    assert node.tags == ["rgb"]
    assert node.input_specs["x"].dtype == "float32"
    assert node.input_specs["x"].variadic is True

    # Cold break: bare-string provides entries are no longer accepted.
    with pytest.raises(ValidationError):
        GitPluginConfig(
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            provides=["my.package.MyNode"],
        )

    # class_name must be a fully-qualified (dotted) path.
    with pytest.raises(ValueError, match="Invalid class path"):
        GitPluginConfig(
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            provides=[{"class_name": "NotDotted"}],
        )


@pytest.mark.parametrize(
    "bad_class_name",
    ["NotDotted", "pkg.", ".Node", "pkg..Node", "pkg.1Node", "pkg. Node"],
)
def test_provides_rejects_malformed_class_path(bad_class_name):
    """A class_name with empty or non-identifier segments is rejected.

    Containing a '.' is not enough: every dot-separated segment must be a valid
    Python identifier, so 'pkg.', '.Node', and 'pkg..Node' are all invalid.
    """
    with pytest.raises(ValueError, match="Invalid class path"):
        GitPluginConfig(
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            provides=[{"class_name": bad_class_name}],
        )


def test_local_plugin_config():
    """Test LocalPluginConfig."""
    plugin = LocalPluginConfig(
        path="/path/to/plugin",
        provides=[{"class_name": "my.package.MyNode"}],
    )
    assert plugin.path == "/path/to/plugin"

    # Test path validation
    with pytest.raises(ValidationError):
        LocalPluginConfig(path="", provides=[{"class_name": "my.package.MyNode"}])


def test_plugin_manifest():
    """Test PluginManifest."""
    manifest = PluginManifest(
        plugins={
            "my_plugin": GitPluginConfig(
                repo="https://github.com/user/repo.git",
                tag="v1.0.0",
                provides=[{"class_name": "my.package.MyNode"}],
            ),
            "local_plugin": LocalPluginConfig(
                path="/path/to/plugin",
                provides=[{"class_name": "other.package.OtherNode"}],
            ),
        }
    )

    assert "my_plugin" in manifest.plugins
    assert "local_plugin" in manifest.plugins
    assert isinstance(manifest.plugins["my_plugin"], GitPluginConfig)
    assert isinstance(manifest.plugins["local_plugin"], LocalPluginConfig)

    # Test invalid plugin name
    with pytest.raises(ValueError, match="Invalid plugin name"):
        PluginManifest(
            plugins={
                "invalid-name": GitPluginConfig(
                    repo="https://github.com/user/repo.git",
                    tag="v1.0.0",
                    provides=[{"class_name": "my.package.MyNode"}],
                ),
            }
        )


def test_plugin_config_to_dict_round_trip():
    """Plugin configs survive to_dict → from_dict round-trip."""
    git_plugin = GitPluginConfig(
        repo="git@gitlab.com:user/repo.git",
        tag="v2.0.0",
        provides=[
            {"class_name": "my.package.MyNode"},
            {"class_name": "my.package.OtherNode"},
        ],
    )
    restored = GitPluginConfig.from_dict(git_plugin.to_dict())
    assert restored.repo == git_plugin.repo
    assert restored.tag == git_plugin.tag
    assert restored.provides == git_plugin.provides

    local_plugin = LocalPluginConfig(
        path="/opt/plugins/my-plugin",
        provides=[{"class_name": "local.package.LocalNode"}],
    )
    restored_local = LocalPluginConfig.from_dict(local_plugin.to_dict())
    assert restored_local.path == local_plugin.path
    assert restored_local.provides == local_plugin.provides


def test_manifest_to_dict_round_trip():
    """PluginManifest survives to_dict → from_dict round-trip."""
    manifest = PluginManifest(
        plugins={
            "git_plugin": GitPluginConfig(
                repo="https://github.com/user/repo.git",
                tag="v1.0.0",
                provides=[{"class_name": "my.package.MyNode"}],
            ),
            "local_plugin": LocalPluginConfig(
                path="/path/to/plugin",
                provides=[{"class_name": "other.package.OtherNode"}],
            ),
        }
    )
    d = manifest.to_dict()
    restored = PluginManifest.from_dict(d)
    assert set(restored.plugins.keys()) == set(manifest.plugins.keys())
    assert restored.plugins["git_plugin"].repo == "https://github.com/user/repo.git"
    assert restored.plugins["local_plugin"].path == "/path/to/plugin"


def test_package_name_optional_override():
    """`package_name` is an optional author override: defaults to None and round-trips."""
    git_plugin = GitPluginConfig(
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        provides=[{"class_name": "my.package.MyNode"}],
    )
    assert git_plugin.package_name is None

    named = LocalPluginConfig(
        path="/opt/plugins/sam3",
        package_name="cuvis-ai-sam3",
        provides=[{"class_name": "cuvis_ai_sam3.node.Sam3"}],
    )
    assert named.package_name == "cuvis-ai-sam3"
    assert LocalPluginConfig.from_dict(named.to_dict()).package_name == "cuvis-ai-sam3"


def test_plugin_config_rejects_unknown_key():
    """extra='forbid' carries over from BaseSchemaModel — stray keys are rejected."""
    with pytest.raises(ValidationError):
        GitPluginConfig(
            repo="https://github.com/user/repo.git",
            tag="v1.0.0",
            provides=[{"class_name": "my.package.MyNode"}],
            bogus_field="nope",
        )


def test_plugin_config_union_export():
    """The PluginConfig union alias is exported for consumers that need the type."""
    git = GitPluginConfig(
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        provides=[{"class_name": "my.package.MyNode"}],
    )
    local = LocalPluginConfig(path="/p", provides=[{"class_name": "my.package.MyNode"}])
    assert isinstance(git, PluginConfig)
    assert isinstance(local, PluginConfig)
