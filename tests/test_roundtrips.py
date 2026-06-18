"""Property-based round-trip tests over the whole BaseSchemaModel family.

The single broad invariant: for every registered model, serializing and
re-parsing (dict and JSON) yields an equal instance. Strategies and the
assertion live in the shipped ``cuvis_ai_schemas.testing`` module so downstream
repos reuse them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402

from cuvis_ai_schemas.base import BaseSchemaModel  # noqa: E402
from cuvis_ai_schemas.testing import (  # noqa: E402
    MODEL_STRATEGIES,
    assert_dict_json_roundtrip,
)


def _by_name(model_cls: type[BaseSchemaModel]) -> str:
    """Sort/id key: the model class name."""
    return model_cls.__name__


# A typed list + in-place sort, rather than sorted(), so mypy keeps the element
# type (type[BaseSchemaModel]) instead of widening the sort key's argument to
# object during sorted()'s joint type-variable inference.
_MODELS: list[type[BaseSchemaModel]] = list(MODEL_STRATEGIES)
_MODELS.sort(key=_by_name)


@pytest.mark.parametrize(
    "model_cls",
    _MODELS,
    ids=_by_name,
)
def test_model_dict_json_roundtrip(model_cls):
    """model_validate(model_dump(x)) == x for both dict and JSON paths."""
    strategy = MODEL_STRATEGIES[model_cls]

    @given(instance=strategy)
    @settings(max_examples=50)
    def _check(instance) -> None:
        assert_dict_json_roundtrip(instance)

    _check()
