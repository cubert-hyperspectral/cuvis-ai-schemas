"""Tests for the hyper-parameter specs a capability entry declares."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.plugin import HparamSpec, PluginCapabilityEntry


def test_int_spec_with_range_and_default():
    spec = HparamSpec(type="int", default=5, min=1, max=31, description="Pooling window")
    assert spec.default == 5
    assert spec.required is False
    assert spec.nullable is False


def test_required_spec_has_no_default():
    assert HparamSpec(type="int", required=True).default is None
    with pytest.raises(ValidationError, match="required"):
        HparamSpec(type="int", required=True, default=3)


def test_optional_spec_without_default_must_be_nullable():
    assert HparamSpec(type="str", nullable=True).default is None
    with pytest.raises(ValidationError, match="nullable"):
        HparamSpec(type="str")


@pytest.mark.parametrize(
    ("kind", "value"),
    [("int", 1.5), ("int", True), ("float", "x"), ("bool", 1), ("str", 3)],
)
def test_default_must_match_type(kind, value):
    with pytest.raises(ValidationError, match="default"):
        HparamSpec(type=kind, default=value)


def test_float_accepts_int_default():
    assert HparamSpec(type="float", default=1).default == 1


def test_default_outside_range_rejected():
    with pytest.raises(ValidationError, match="range"):
        HparamSpec(type="int", default=40, min=1, max=31)


def test_min_above_max_rejected():
    with pytest.raises(ValidationError, match="min"):
        HparamSpec(type="float", default=0.5, min=1.0, max=0.0)


def test_range_only_for_numbers():
    with pytest.raises(ValidationError, match="min"):
        HparamSpec(type="str", default="a", min=1)


def test_enum_needs_choices_and_default_among_them():
    spec = HparamSpec(type="enum", choices=["abs", "sq"], default="abs")
    assert spec.choices == ["abs", "sq"]
    with pytest.raises(ValidationError, match="choices"):
        HparamSpec(type="enum", default="abs")
    with pytest.raises(ValidationError, match="choices"):
        HparamSpec(type="enum", choices=["abs", "abs"], default="abs")
    with pytest.raises(ValidationError, match="choices"):
        HparamSpec(type="enum", choices=["abs", "sq"], default="cube")


def test_choices_only_for_enum():
    with pytest.raises(ValidationError, match="choices"):
        HparamSpec(type="int", default=1, choices=[1, 2])


def test_list_spec_checks_each_item():
    spec = HparamSpec(type="int", list=True, default=[1, 2], min=0)
    assert spec.default == [1, 2]
    with pytest.raises(ValidationError, match="list"):
        HparamSpec(type="int", list=True, default=3)
    with pytest.raises(ValidationError, match="range"):
        HparamSpec(type="int", list=True, default=[1, -1], min=0)


def test_capability_hparams_keys_are_identifiers():
    entry = PluginCapabilityEntry(
        class_name="pkg.mod.Node",
        hparams={"window": {"type": "int", "default": 5}},
    )
    assert entry.hparams["window"].default == 5
    with pytest.raises(ValidationError, match="identifier"):
        PluginCapabilityEntry(
            class_name="pkg.mod.Node", hparams={"not-valid": {"type": "int", "default": 5}}
        )


def test_capability_hparams_absent_by_default():
    entry = PluginCapabilityEntry(class_name="pkg.mod.Node")
    assert entry.hparams is None
    assert "hparams" not in entry.model_dump(exclude_none=True)


def test_data_module_entry_has_no_hparams():
    with pytest.raises(ValidationError, match="hparams"):
        PluginCapabilityEntry(
            class_name="pkg.mod.Data",
            kind="data_module",
            data_module_name="data",
            hparams={},
        )
