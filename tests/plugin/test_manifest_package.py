"""Tests for the package plugin source: a prebuilt package from a named index."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from cuvis_ai_schemas.plugin import (
    PackagePluginSource,
    load_plugin_manifest,
    parse_plugin_manifest,
    write_plugin_manifest,
)

_NODE = {
    "class_name": "closed_pkg._contrast.SpectralContrast",
    "doc_summary": "Score each pixel against the fitted mean spectrum.",
    "hparams": {
        "num_channels": {"type": "int", "required": True, "min": 1},
        "window": {"type": "int", "default": 5, "min": 1, "max": 31},
    },
}


def _package(**overrides) -> dict:
    data = {
        "name": "closed_samples",
        "package_name": "cuvis-ai-closed-samples",
        "version": "==0.1.0",
        "index": "cubert-closed",
        "capabilities": [_NODE],
    }
    data.update(overrides)
    return data


def test_parse_selects_package_source():
    manifest = parse_plugin_manifest(_package())
    assert isinstance(manifest, PackagePluginSource)
    assert manifest.package_name == "cuvis-ai-closed-samples"
    assert manifest.version == "==0.1.0"
    assert manifest.pinned_version == "0.1.0"
    assert manifest.index == "cubert-closed"
    assert manifest.capabilities[0].hparams["window"].max == 31


@pytest.mark.parametrize("version", ["0.1.0", ">=0.1.0", "==0.1.*", "==", "== 0.1.0", "==0.1.0,<1"])
def test_version_must_be_an_exact_pin(version):
    with pytest.raises(ValidationError, match="version"):
        parse_plugin_manifest(_package(version=version))


@pytest.mark.parametrize("version", ["==1.2.3", "==0.1.0rc1", "==2.0.0.post1", "==1.0"])
def test_exact_pins_accepted(version):
    assert parse_plugin_manifest(_package(version=version)).version == version


@pytest.mark.parametrize(
    "index",
    [
        "https://registry.example/simple",
        "user:token@host",
        "Cubert-Closed",
        "-closed",
        "closed index",
        "closed/index",
        "",
    ],
)
def test_index_is_a_name_never_a_url_or_secret(index):
    with pytest.raises(ValidationError, match="index"):
        parse_plugin_manifest(_package(index=index))


@pytest.mark.parametrize("missing", ["package_name", "version", "index"])
def test_package_fields_are_required(missing):
    data = _package()
    del data[missing]
    with pytest.raises(ValidationError):
        parse_plugin_manifest(data)


def test_package_source_rejects_repo_or_path():
    with pytest.raises(ValidationError):
        parse_plugin_manifest(_package(repo="https://github.com/x/y.git", tag="v1.0.0"))
    with pytest.raises(ValidationError):
        parse_plugin_manifest(_package(path="../closed"))


def test_git_and_local_sources_reject_index_and_version():
    git = {
        "name": "p",
        "repo": "https://github.com/x/y.git",
        "tag": "v1.0.0",
        "capabilities": [{"class_name": "a.B"}],
    }
    with pytest.raises(ValidationError):
        parse_plugin_manifest({**git, "index": "cubert-closed"})
    local = {"name": "p", "path": "/tmp/p", "capabilities": [{"class_name": "a.B"}]}
    with pytest.raises(ValidationError):
        parse_plugin_manifest({**local, "version": "==0.1.0"})


def test_package_nodes_must_declare_hparams():
    node = {k: v for k, v in _NODE.items() if k != "hparams"}
    with pytest.raises(ValidationError, match="hparams"):
        parse_plugin_manifest(_package(capabilities=[node]))
    assert parse_plugin_manifest(_package(capabilities=[{**node, "hparams": {}}]))


def test_write_then_load_round_trip(tmp_path: Path):
    manifest = parse_plugin_manifest(_package(extras=["tensorrt"]))
    path = tmp_path / "closed_samples.yaml"
    write_plugin_manifest(manifest, path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["index"] == "cubert-closed"
    assert "repo" not in data and "path" not in data
    assert load_plugin_manifest(path) == manifest


def test_written_open_manifest_has_no_hparams_key(tmp_path: Path):
    manifest = parse_plugin_manifest(
        {"name": "p", "path": str(tmp_path), "capabilities": [{"class_name": "a.B"}]}
    )
    path = tmp_path / "p.yaml"
    write_plugin_manifest(manifest, path)
    assert "hparams" not in path.read_text(encoding="utf-8")


def test_missing_source_error_names_the_package_shape():
    with pytest.raises(ValidationError, match="index"):
        parse_plugin_manifest({"name": "p", "capabilities": [{"class_name": "a.B"}]})
