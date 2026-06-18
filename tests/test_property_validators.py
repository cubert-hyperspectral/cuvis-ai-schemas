"""Property-based tests for validator invariants + the testing-extra guard."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from cuvis_ai_schemas.plugin import GitPluginSource, PluginCapabilityEntry  # noqa: E402
from cuvis_ai_schemas.testing import strategies as S  # noqa: E402
from cuvis_ai_schemas.training import SampleRef  # noqa: E402

_non_identifiers = st.text(min_size=1, max_size=12).filter(lambda s: not s.isidentifier())


def _git(name: str) -> GitPluginSource:
    return GitPluginSource(
        name=name,
        repo="https://github.com/user/repo.git",
        tag="v1.0.0",
        capabilities=[PluginCapabilityEntry(class_name="pkg.mod.Node")],
    )


@given(name=S.identifiers)
@settings(max_examples=50)
def test_identifier_names_accepted(name):
    """Any valid Python identifier is an acceptable plugin name."""
    assert _git(name).name == name


@given(name=_non_identifiers)
@settings(max_examples=50)
def test_non_identifier_names_rejected(name):
    """Any non-identifier is rejected as a plugin name."""
    with pytest.raises(ValueError):
        _git(name)


@given(class_name=S.fqcns)
@settings(max_examples=50)
def test_fqcn_class_names_accepted(class_name):
    """Any dotted path of >= 2 identifiers is a valid capability class_name."""
    assert PluginCapabilityEntry(class_name=class_name).class_name == class_name


@given(name=S.identifiers)
@settings(max_examples=50)
def test_single_segment_class_name_rejected(name):
    """A single (non-dotted) segment is rejected as a class_name."""
    with pytest.raises(ValueError, match="Invalid class path"):
        PluginCapabilityEntry(class_name=name)


@given(source=S.identifiers, index=st.integers(min_value=0, max_value=100))
@settings(max_examples=50)
def test_sample_ref_uid_is_deterministic_from_content(source, index):
    """SampleRef.uid is a deterministic function of (source, index)."""
    a = SampleRef(source=source, index=index)
    b = SampleRef(source=source, index=index)
    assert a.uid == b.uid == f"{source}#{index}"


def test_model_strategy_unregistered_raises():
    """model_strategy raises a helpful KeyError for an unregistered model."""
    from cuvis_ai_schemas.training.trainer import TrainerConfig

    with pytest.raises(KeyError, match="No Hypothesis strategy registered"):
        S.model_strategy(TrainerConfig)


def test_testing_module_requires_hypothesis_extra():
    """Importing cuvis_ai_schemas.testing without hypothesis gives a clear hint."""
    saved = {
        name: sys.modules.pop(name, None)
        for name in (
            "cuvis_ai_schemas.testing",
            "cuvis_ai_schemas.testing.strategies",
        )
    }
    try:
        with patch.dict(sys.modules, {"hypothesis": None}):
            with pytest.raises(ImportError, match=r"cuvis-ai-schemas\[testing\]"):
                importlib.import_module("cuvis_ai_schemas.testing.strategies")
    finally:
        for name in ("cuvis_ai_schemas.testing", "cuvis_ai_schemas.testing.strategies"):
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            if mod is not None:
                sys.modules[name] = mod
        importlib.import_module("cuvis_ai_schemas.testing")
