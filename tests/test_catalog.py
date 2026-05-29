"""Tests for the static node catalog Pydantic models."""

from __future__ import annotations

import json

import pytest

from cuvis_ai_schemas.catalog import (
    SUPPORTED_SCHEMA_VERSIONS,
    CatalogPluginEntry,
    CatalogPortSpec,
)


def _minimal_payload(*, plugin_name: str = "my_plugin") -> dict:
    return {
        "schema_version": 1,
        "plugin_name": plugin_name,
        "plugin_version": "0.1.0",
        "nodes": [
            {
                "class_name": "MyNode",
                "full_path": "my_plugin.node.MyNode",
                "category": "transform",
                "tags": ["image"],
                "icon_svg": "<svg></svg>",
                "input_specs": {
                    "x": {
                        "dtype": "float32",
                        "shape": [-1, -1, -1, -1],
                    }
                },
                "output_specs": {
                    "y": [
                        {"dtype": "float32", "shape": [-1, -1, -1, -1]},
                    ]
                },
                "doc_summary": "Does a thing.",
            }
        ],
    }


def test_catalog_plugin_entry_validates_full_payload():
    entry = CatalogPluginEntry.model_validate(_minimal_payload())
    assert entry.plugin_name == "my_plugin"
    assert entry.plugin_version == "0.1.0"
    assert entry.schema_version == 1
    assert len(entry.nodes) == 1

    node = entry.nodes[0]
    assert node.class_name == "MyNode"
    assert node.full_path == "my_plugin.node.MyNode"
    assert node.category == "transform"
    assert node.tags == ["image"]
    assert node.icon_svg == "<svg></svg>"
    assert node.doc_summary == "Does a thing."

    assert list(node.input_specs.keys()) == ["x"]
    spec = node.input_specs["x"][0]
    assert isinstance(spec, CatalogPortSpec)
    assert spec.dtype == "float32"
    assert spec.shape == [-1, -1, -1, -1]


def test_single_spec_dict_is_coerced_to_list():
    payload = _minimal_payload()
    # Already exercised by `input_specs`; assert the list is built.
    entry = CatalogPluginEntry.model_validate(payload)
    assert isinstance(entry.nodes[0].input_specs["x"], list)
    assert len(entry.nodes[0].input_specs["x"]) == 1


def test_full_path_defaults_to_class_name():
    payload = _minimal_payload()
    payload["nodes"][0].pop("full_path")
    entry = CatalogPluginEntry.model_validate(payload)
    assert entry.nodes[0].full_path == "MyNode"


def test_schema_version_must_be_supported():
    payload = _minimal_payload()
    payload["schema_version"] = 999

    with pytest.raises(ValueError, match="unsupported schema_version"):
        CatalogPluginEntry.model_validate(payload)


def test_missing_plugin_name_fails_validation():
    payload = _minimal_payload()
    del payload["plugin_name"]

    with pytest.raises(ValueError, match="plugin_name"):
        CatalogPluginEntry.model_validate(payload)


def test_empty_dtype_is_allowed_for_generic_tensor_markers():
    """Generic-tensor markers (no concrete dtype) emit dtype=''."""
    payload = _minimal_payload()
    del payload["nodes"][0]["input_specs"]["x"]["dtype"]

    entry = CatalogPluginEntry.model_validate(payload)
    assert entry.nodes[0].input_specs["x"][0].dtype == ""


def test_non_int_shape_fails_validation():
    payload = _minimal_payload()
    payload["nodes"][0]["input_specs"]["x"]["shape"] = ["dynamic", -1]

    with pytest.raises(ValueError, match="shape"):
        CatalogPluginEntry.model_validate(payload)


def test_extra_field_is_rejected():
    payload = _minimal_payload()
    payload["surprise"] = "uh oh"

    with pytest.raises(ValueError, match="surprise"):
        CatalogPluginEntry.model_validate(payload)


def test_from_metadata_file_round_trip(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(_minimal_payload()), encoding="utf-8")

    entry = CatalogPluginEntry.from_metadata_file(metadata_path)
    assert entry.plugin_name == "my_plugin"
    assert entry.nodes[0].class_name == "MyNode"


def test_from_metadata_file_raises_when_missing(tmp_path):
    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        CatalogPluginEntry.from_metadata_file(missing)


def test_supported_versions_constant_is_tuple_of_int():
    assert isinstance(SUPPORTED_SCHEMA_VERSIONS, tuple)
    assert all(isinstance(v, int) for v in SUPPORTED_SCHEMA_VERSIONS)
    assert 1 in SUPPORTED_SCHEMA_VERSIONS


def test_models_are_frozen():
    entry = CatalogPluginEntry.model_validate(_minimal_payload())
    with pytest.raises((ValueError, TypeError)):
        entry.plugin_name = "tampered"

    node = entry.nodes[0]
    with pytest.raises((ValueError, TypeError)):
        node.class_name = "tampered"

    spec = node.input_specs["x"][0]
    with pytest.raises((ValueError, TypeError)):
        spec.dtype = "tampered"


def test_to_dict_and_from_dict_round_trip():
    payload = _minimal_payload()
    entry = CatalogPluginEntry.model_validate(payload)
    rebuilt = CatalogPluginEntry.from_dict(entry.to_dict())
    assert rebuilt == entry
