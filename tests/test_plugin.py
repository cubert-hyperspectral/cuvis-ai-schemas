"""Tests for plugin schemas."""

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.plugin import GitPluginConfig, LocalPluginConfig, PluginManifest


def test_git_plugin_config():
    """Test GitPluginConfig validation."""
    plugin = GitPluginConfig(
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        provides=["my.package.MyNode"],
    )
    assert plugin.repo == "https://github.com/user/repo.git"
    assert plugin.tag == "v1.0.0"

    # Test repo validation
    with pytest.raises(ValueError, match="Invalid repo URL"):
        GitPluginConfig(
            repo="invalid-url",
            tag="v1.0.0",
            provides=["my.package.MyNode"],
        )

    # Test tag validation
    with pytest.raises(ValidationError):
        GitPluginConfig(
            repo="https://github.com/user/repo.git",
            tag="",
            provides=["my.package.MyNode"],
        )


def test_local_plugin_config():
    """Test LocalPluginConfig."""
    plugin = LocalPluginConfig(
        path="/path/to/plugin",
        provides=["my.package.MyNode"],
    )
    assert plugin.path == "/path/to/plugin"

    # Test path validation
    with pytest.raises(ValidationError):
        LocalPluginConfig(path="", provides=["my.package.MyNode"])


def test_plugin_manifest():
    """Test PluginManifest."""
    manifest = PluginManifest(
        plugins={
            "my_plugin": GitPluginConfig(
                repo="https://github.com/user/repo.git",
                tag="v1.0.0",
                provides=["my.package.MyNode"],
            ),
            "local_plugin": LocalPluginConfig(
                path="/path/to/plugin",
                provides=["other.package.OtherNode"],
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
                    provides=["my.package.MyNode"],
                ),
            }
        )
