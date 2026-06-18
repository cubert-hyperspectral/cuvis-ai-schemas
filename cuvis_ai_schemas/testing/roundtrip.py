"""Reusable round-trip assertions for cuvis-ai-schemas models.

These helpers carry no Hypothesis dependency themselves; they are the assertion
half of the property-based tests and are equally usable from plain example
tests. Downstream repos (cuvis-ai, cuvis-ai-core) can import them to assert the
same serialization invariants on their own configs.
"""

from __future__ import annotations

from cuvis_ai_schemas.base import BaseSchemaModel


def assert_dict_roundtrip(instance: BaseSchemaModel) -> None:
    """Assert ``from_dict(to_dict(x))`` reproduces an equal model."""
    cls = type(instance)
    assert cls.from_dict(instance.to_dict()) == instance


def assert_json_roundtrip(instance: BaseSchemaModel) -> None:
    """Assert ``from_json(to_json(x))`` reproduces an equal model."""
    cls = type(instance)
    assert cls.from_json(instance.to_json()) == instance


def assert_dict_json_roundtrip(instance: BaseSchemaModel) -> None:
    """Assert both the dict and JSON round-trips reproduce an equal model.

    The single invariant most worth holding for every ``BaseSchemaModel``:
    serializing and re-parsing must yield an equal instance, for both the dict
    and JSON paths.
    """
    assert_dict_roundtrip(instance)
    assert_json_roundtrip(instance)
