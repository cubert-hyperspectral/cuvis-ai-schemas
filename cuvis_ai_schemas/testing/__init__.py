"""Property-based testing helpers for cuvis-ai-schemas models.

A shipped, importable home for Hypothesis strategies and round-trip assertions,
so the schemas suite and downstream repos (cuvis-ai, cuvis-ai-core) reuse one
source of truth instead of each reinventing generators. Importing this package
requires the optional ``testing`` extra::

    pip install cuvis-ai-schemas[testing]

The round-trip assertions (:mod:`cuvis_ai_schemas.testing.roundtrip`) carry no
Hypothesis dependency and are usable from plain example tests too.
"""

from __future__ import annotations

from cuvis_ai_schemas.testing.roundtrip import (
    assert_dict_json_roundtrip,
    assert_dict_roundtrip,
    assert_json_roundtrip,
)
from cuvis_ai_schemas.testing.strategies import MODEL_STRATEGIES, model_strategy

__all__ = [
    "MODEL_STRATEGIES",
    "assert_dict_json_roundtrip",
    "assert_dict_roundtrip",
    "assert_json_roundtrip",
    "model_strategy",
]
