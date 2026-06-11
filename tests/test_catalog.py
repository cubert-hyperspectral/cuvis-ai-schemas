"""Tests for the static node catalog Pydantic models."""

from __future__ import annotations

import pytest

from cuvis_ai_schemas.catalog import (
    SUPPORTED_SCHEMA_VERSIONS,
    CatalogNodeEntry,
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
                "class_name": "my_plugin.node.MyNode",
                "category": "transform",
                "tags": ["image"],
                "icon_svg": "<svg></svg>",
                "input_specs": {
                    "x": {
                        "dtype": "float32",
                        "shape": [-1, -1, -1, -1],
                        "variadic": True,
                    }
                },
                "output_specs": {
                    "y": {"dtype": "float32", "shape": [-1, -1, -1, -1]},
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
    assert node.class_name == "my_plugin.node.MyNode"
    assert node.category == "transform"
    assert node.tags == ["image"]
    assert node.icon_svg == "<svg></svg>"
    assert node.doc_summary == "Does a thing."

    assert list(node.input_specs.keys()) == ["x"]
    spec = node.input_specs["x"]
    assert isinstance(spec, CatalogPortSpec)
    assert spec.dtype == "float32"
    assert spec.shape == [-1, -1, -1, -1]
    assert spec.variadic is True
    assert node.output_specs["y"].variadic is False


def test_port_spec_is_a_single_spec_not_a_list():
    """Each port maps to exactly one CatalogPortSpec; variadic is a flag."""
    entry = CatalogPluginEntry.model_validate(_minimal_payload())
    node = entry.nodes[0]
    assert isinstance(node.input_specs["x"], CatalogPortSpec)
    assert isinstance(node.output_specs["y"], CatalogPortSpec)


def test_class_name_is_the_fqcn():
    """`class_name` carries the fully-qualified path; there is no `full_path`."""
    node = CatalogPluginEntry.model_validate(_minimal_payload()).nodes[0]
    assert node.class_name == "my_plugin.node.MyNode"
    assert not hasattr(node, "full_path")


def test_full_path_is_rejected_as_extra_field():
    """The removed `full_path` field is now an unknown key (extra='forbid')."""
    payload = _minimal_payload()
    payload["nodes"][0]["full_path"] = "my_plugin.node.MyNode"
    with pytest.raises(ValueError, match="full_path"):
        CatalogPluginEntry.model_validate(payload)


def test_node_entry_defaults_to_node_kind():
    """An entry with no `kind` defaults to a node with empty data_module_name/extras."""
    node = CatalogNodeEntry(class_name="my_plugin.node.MyNode")
    assert node.kind == "node"
    assert node.data_module_name == ""
    assert node.extras == []


def test_data_module_entry_is_valid():
    """A data_module entry carries a unique name + extras and a real class FQCN."""
    node = CatalogNodeEntry(
        class_name="cuvis_ai_dataloader.data.datamodule_cu3s.Cu3sDataModule",
        kind="data_module",
        data_module_name="cu3s",
        extras=["cu3s", "coco"],
    )
    assert node.kind == "data_module"
    assert node.data_module_name == "cu3s"
    assert node.extras == ["cu3s", "coco"]


def test_data_module_entry_requires_a_name():
    """kind='data_module' without data_module_name fails the invariant."""
    with pytest.raises(ValueError, match="requires a non-empty 'data_module_name'"):
        CatalogNodeEntry(
            class_name="pkg.module.SomeModule",
            kind="data_module",
        )


def test_node_kind_rejects_data_module_fields():
    """A node entry must not carry data_module_name or extras."""
    with pytest.raises(ValueError, match="must not set 'data_module_name' or 'extras'"):
        CatalogNodeEntry(class_name="pkg.module.MyNode", data_module_name="cu3s")
    with pytest.raises(ValueError, match="must not set 'data_module_name' or 'extras'"):
        CatalogNodeEntry(class_name="pkg.module.MyNode", extras=["cu3s"])


def test_data_module_entry_round_trips_through_manifest():
    """data_module entries flow through CatalogPluginEntry.from_manifest_entry."""
    config_dict = {
        "provides": [
            {"class_name": "my_plugin.node.MyNode", "category": "transform"},
            {
                "class_name": "cuvis_ai_dataloader.data.datamodule_cu3s.Cu3sDataModule",
                "kind": "data_module",
                "data_module_name": "cu3s",
                "extras": ["cu3s", "coco"],
            },
        ],
    }
    entry = CatalogPluginEntry.from_manifest_entry("cuvis_ai_dataloader", config_dict)
    assert entry is not None
    dm = [n for n in entry.nodes if n.kind == "data_module"]
    assert len(dm) == 1
    assert dm[0].data_module_name == "cu3s"


@pytest.mark.parametrize(
    "bad_class_name",
    ["NotDotted", "pkg.", ".Node", "pkg..Node", "pkg.1Node", "pkg. Node"],
)
def test_class_name_rejects_malformed_path(bad_class_name):
    """class_name must be a dotted path of Python identifiers (no empty segments).

    The check lives on CatalogNodeEntry, so it also guards the server-side
    catalog-load path (CatalogPluginEntry / from_manifest_entry), not only the
    plugin-config `provides` list.
    """
    payload = _minimal_payload()
    payload["nodes"][0]["class_name"] = bad_class_name
    with pytest.raises(ValueError, match="Invalid class path"):
        CatalogPluginEntry.model_validate(payload)


def test_schema_version_defaults_when_absent():
    payload = _minimal_payload()
    del payload["schema_version"]
    entry = CatalogPluginEntry.model_validate(payload)
    assert entry.schema_version == SUPPORTED_SCHEMA_VERSIONS[-1]


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
    assert entry.nodes[0].input_specs["x"].dtype == ""


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


def test_from_manifest_entry_builds_catalog_from_provides():
    """The manifest entry's `provides` list IS the node catalog."""
    config_dict = {
        "repo": "https://github.com/user/repo.git",
        "tag": "v1.0.0",
        "provides": [
            {
                "class_name": "my_plugin.node.MyNode",
                "category": "model",
                "tags": ["rgb"],
            }
        ],
    }
    entry = CatalogPluginEntry.from_manifest_entry("my_plugin", config_dict)
    assert entry is not None
    assert entry.plugin_name == "my_plugin"
    assert entry.schema_version == SUPPORTED_SCHEMA_VERSIONS[-1]
    assert len(entry.nodes) == 1
    assert entry.nodes[0].class_name == "my_plugin.node.MyNode"
    assert entry.nodes[0].category == "model"


def test_from_manifest_entry_returns_none_without_provides():
    """No provided nodes → no palette entry."""
    assert CatalogPluginEntry.from_manifest_entry("p", {}) is None
    assert CatalogPluginEntry.from_manifest_entry("p", {"provides": []}) is None


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

    spec = node.input_specs["x"]
    with pytest.raises((ValueError, TypeError)):
        spec.dtype = "tampered"


def test_to_dict_and_from_dict_round_trip():
    payload = _minimal_payload()
    entry = CatalogPluginEntry.model_validate(payload)
    rebuilt = CatalogPluginEntry.from_dict(entry.to_dict())
    assert rebuilt == entry
