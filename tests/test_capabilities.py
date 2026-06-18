"""Tests for plugin capability schemas (entries, port specs, capability sets)."""

from __future__ import annotations

import pytest

from cuvis_ai_schemas.plugin import (
    NodePortSpec,
    PluginCapabilities,
    PluginCapabilityEntry,
)


def _minimal_payload(*, plugin_name: str = "my_plugin") -> dict:
    return {
        "plugin_name": plugin_name,
        "plugin_version": "0.1.0",
        "capabilities": [
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


def test_plugin_capabilities_validates_full_payload():
    entry = PluginCapabilities.model_validate(_minimal_payload())
    assert entry.plugin_name == "my_plugin"
    assert entry.plugin_version == "0.1.0"
    assert len(entry.capabilities) == 1

    node = entry.capabilities[0]
    assert node.class_name == "my_plugin.node.MyNode"
    assert node.category == "transform"
    assert node.tags == ["image"]
    assert node.icon_svg == "<svg></svg>"
    assert node.doc_summary == "Does a thing."

    assert list(node.input_specs.keys()) == ["x"]
    spec = node.input_specs["x"]
    assert isinstance(spec, NodePortSpec)
    assert spec.dtype == "float32"
    assert spec.shape == [-1, -1, -1, -1]
    assert spec.variadic is True
    assert node.output_specs["y"].variadic is False


def test_port_spec_is_a_single_spec_not_a_list():
    """Each port maps to exactly one NodePortSpec; variadic is a flag."""
    entry = PluginCapabilities.model_validate(_minimal_payload())
    node = entry.capabilities[0]
    assert isinstance(node.input_specs["x"], NodePortSpec)
    assert isinstance(node.output_specs["y"], NodePortSpec)


def test_class_name_is_the_fqcn():
    """`class_name` carries the fully-qualified path; there is no `full_path`."""
    node = PluginCapabilities.model_validate(_minimal_payload()).capabilities[0]
    assert node.class_name == "my_plugin.node.MyNode"
    assert not hasattr(node, "full_path")


def test_full_path_is_rejected_as_extra_field():
    """The removed `full_path` field is now an unknown key (extra='forbid')."""
    payload = _minimal_payload()
    payload["capabilities"][0]["full_path"] = "my_plugin.node.MyNode"
    with pytest.raises(ValueError, match="full_path"):
        PluginCapabilities.model_validate(payload)


def test_capability_entry_defaults_to_node_kind():
    """An entry with no `kind` defaults to a node with empty data_module_name/extras."""
    node = PluginCapabilityEntry(class_name="my_plugin.node.MyNode")
    assert node.kind == "node"
    assert node.data_module_name == ""
    assert node.extras == []


def test_data_module_entry_is_valid():
    """A data_module entry carries a unique name + extras and a real class FQCN."""
    node = PluginCapabilityEntry(
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
        PluginCapabilityEntry(
            class_name="pkg.module.SomeModule",
            kind="data_module",
        )


def test_node_kind_rejects_data_module_fields():
    """A node entry must not carry data_module_name or extras."""
    with pytest.raises(ValueError, match="must not set 'data_module_name' or 'extras'"):
        PluginCapabilityEntry(class_name="pkg.module.MyNode", data_module_name="cu3s")
    with pytest.raises(ValueError, match="must not set 'data_module_name' or 'extras'"):
        PluginCapabilityEntry(class_name="pkg.module.MyNode", extras=["cu3s"])


@pytest.mark.parametrize(
    "bad_class_name",
    ["NotDotted", "pkg.", ".Node", "pkg..Node", "pkg.1Node", "pkg. Node"],
)
def test_class_name_rejects_malformed_path(bad_class_name):
    """class_name must be a dotted path of Python identifiers (no empty segments).

    The check lives on PluginCapabilityEntry, so it guards the server-side
    capability-load path (PluginCapabilities) too, not only direct construction.
    """
    payload = _minimal_payload()
    payload["capabilities"][0]["class_name"] = bad_class_name
    with pytest.raises(ValueError, match="Invalid class path"):
        PluginCapabilities.model_validate(payload)


def test_missing_plugin_name_fails_validation():
    payload = _minimal_payload()
    del payload["plugin_name"]

    with pytest.raises(ValueError, match="plugin_name"):
        PluginCapabilities.model_validate(payload)


def test_empty_dtype_is_allowed_for_generic_tensor_markers():
    """Generic-tensor markers (no concrete dtype) emit dtype=''."""
    payload = _minimal_payload()
    del payload["capabilities"][0]["input_specs"]["x"]["dtype"]

    entry = PluginCapabilities.model_validate(payload)
    assert entry.capabilities[0].input_specs["x"].dtype == ""


def test_non_int_shape_fails_validation():
    payload = _minimal_payload()
    payload["capabilities"][0]["input_specs"]["x"]["shape"] = ["dynamic", -1]

    with pytest.raises(ValueError, match="shape"):
        PluginCapabilities.model_validate(payload)


def test_extra_field_is_rejected():
    payload = _minimal_payload()
    payload["surprise"] = "uh oh"

    with pytest.raises(ValueError, match="surprise"):
        PluginCapabilities.model_validate(payload)


def test_models_are_frozen():
    entry = PluginCapabilities.model_validate(_minimal_payload())
    with pytest.raises((ValueError, TypeError)):
        entry.plugin_name = "tampered"

    node = entry.capabilities[0]
    with pytest.raises((ValueError, TypeError)):
        node.class_name = "tampered"

    spec = node.input_specs["x"]
    with pytest.raises((ValueError, TypeError)):
        spec.dtype = "tampered"


def test_to_dict_and_from_dict_round_trip():
    payload = _minimal_payload()
    entry = PluginCapabilities.model_validate(payload)
    rebuilt = PluginCapabilities.from_dict(entry.to_dict())
    assert rebuilt == entry
