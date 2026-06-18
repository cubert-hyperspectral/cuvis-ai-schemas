"""Tests for the runtime port system (PortSpec, DimensionResolver, ports).

Two layers: explicit parametrized cases that pin the branch behavior and error
messages, plus a Hypothesis property layer over the compatibility invariants.
Torch is an optional extra, so the whole module is skipped without it.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from cuvis_ai_schemas.pipeline.ports import (  # noqa: E402
    DimensionResolver,
    InputPort,
    OutputPort,
    PortSpec,
)


class _Node:
    """Minimal stand-in node carrying arbitrary attributes for resolution."""

    def __init__(self, **attrs: object) -> None:
        """Store the given attributes on the instance."""
        self.__dict__.update(attrs)


# ---------------------------------------------------------------------------
# DimensionResolver
# ---------------------------------------------------------------------------
def test_resolve_int_and_flexible_dims_pass_through():
    """Concrete ints and the -1 flexible marker resolve to themselves."""
    assert DimensionResolver.resolve((3, -1, 8), None) == (3, -1, 8)


def test_resolve_symbolic_dim_from_node_attr():
    """A string dim resolves from the matching int node attribute."""
    node = _Node(batch_size=4, channels=3)
    assert DimensionResolver.resolve(("batch_size", "channels", 5), node) == (4, 3, 5)


def test_resolve_symbolic_without_node_raises_value_error():
    """A symbolic dim without a node instance is a ValueError."""
    with pytest.raises(ValueError, match="without node instance"):
        DimensionResolver.resolve(("batch_size",), None)


def test_resolve_missing_attr_raises_attribute_error():
    """A symbolic dim with no matching node attribute is an AttributeError."""
    with pytest.raises(AttributeError, match="no attribute 'batch_size'"):
        DimensionResolver.resolve(("batch_size",), _Node(id="n1"))


def test_resolve_non_int_attr_raises_type_error():
    """A symbolic dim resolving to a non-int value is a TypeError."""
    with pytest.raises(TypeError, match="expected int"):
        DimensionResolver.resolve(("batch_size",), _Node(batch_size="big"))


def test_resolve_invalid_dim_type_raises_type_error():
    """A dim that is neither int nor str is a TypeError."""
    with pytest.raises(TypeError, match="Invalid dimension type"):
        DimensionResolver.resolve((1.5,), None)


def test_port_spec_resolve_shape_delegates():
    """PortSpec.resolve_shape resolves symbolic dims via the node."""
    spec = PortSpec(dtype=torch.float32, shape=("batch_size", 3))
    assert spec.resolve_shape(_Node(batch_size=2)) == (2, 3)


# ---------------------------------------------------------------------------
# PortSpec.is_compatible_with — explicit matrix
# ---------------------------------------------------------------------------
def test_generic_tensor_compatible_with_specific_dtype():
    """A generic torch.Tensor source connects to a specific-dtype target."""
    src = PortSpec(dtype=torch.Tensor, shape=(-1, 3))
    tgt = PortSpec(dtype=torch.float32, shape=(-1, 3))
    ok, msg = src.is_compatible_with(tgt, None, None)
    assert ok and msg == ""


def test_same_specific_dtype_compatible():
    """Identical specific dtypes are compatible."""
    src = PortSpec(dtype=torch.float32, shape=(2, 2))
    assert src.is_compatible_with(PortSpec(dtype=torch.float32, shape=(2, 2)), None, None)[0]


def test_tensor_dtype_mismatch_fails():
    """Two different specific torch dtypes are incompatible."""
    src = PortSpec(dtype=torch.float32, shape=(2,))
    ok, msg = src.is_compatible_with(PortSpec(dtype=torch.int64, shape=(2,)), None, None)
    assert not ok
    assert "Dtype mismatch" in msg


def test_non_tensor_exact_match_and_mismatch():
    """Non-tensor dtypes must match exactly."""
    assert PortSpec(dtype=int, shape=(1,)).is_compatible_with(
        PortSpec(dtype=int, shape=(1,)), None, None
    )[0]
    ok, msg = PortSpec(dtype=int, shape=(1,)).is_compatible_with(
        PortSpec(dtype=str, shape=(1,)), None, None
    )
    assert not ok
    assert "Dtype mismatch" in msg


def test_rank_mismatch_fails():
    """Differing shape ranks are incompatible."""
    ok, msg = PortSpec(dtype=torch.Tensor, shape=(3,)).is_compatible_with(
        PortSpec(dtype=torch.Tensor, shape=(3, 4)), None, None
    )
    assert not ok
    assert "rank mismatch" in msg


def test_dimension_mismatch_fails():
    """A differing fixed dimension is incompatible and names the index."""
    ok, msg = PortSpec(dtype=torch.Tensor, shape=(3, 4)).is_compatible_with(
        PortSpec(dtype=torch.Tensor, shape=(3, 5)), None, None
    )
    assert not ok
    assert "Dimension 1 mismatch" in msg


def test_flexible_dim_is_wildcard():
    """A -1 dim on either side is compatible with any size."""
    assert PortSpec(dtype=torch.Tensor, shape=(-1, 4)).is_compatible_with(
        PortSpec(dtype=torch.Tensor, shape=(3, 4)), None, None
    )[0]


def test_shape_resolution_failure_reported():
    """An unresolvable symbolic dim surfaces as a shape-resolution failure."""
    src = PortSpec(dtype=torch.Tensor, shape=("missing",))
    ok, msg = src.is_compatible_with(
        PortSpec(dtype=torch.Tensor, shape=("missing",)), _Node(id="n"), _Node(id="n")
    )
    assert not ok
    assert "Shape resolution failed" in msg


# ---------------------------------------------------------------------------
# B1: torch decoupling contract
# ---------------------------------------------------------------------------
def test_torch_free_paths_work_without_torch():
    """Dimension resolution works even with torch absent (it needs no torch)."""
    with patch.dict(sys.modules, {"torch": None}):
        assert DimensionResolver.resolve((1, 2, 3), None) == (1, 2, 3)


def test_is_compatible_with_guards_missing_torch():
    """With torch absent, is_compatible_with raises a clear install hint."""
    src = PortSpec(dtype="cube", shape=(1,))
    tgt = PortSpec(dtype="cube", shape=(1,))
    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match=r"cuvis-ai-schemas\[torch\]"):
            src.is_compatible_with(tgt, None, None)


# ---------------------------------------------------------------------------
# Port proxies
# ---------------------------------------------------------------------------
def test_port_repr_with_and_without_node_id():
    """The port repr uses node.id when present, else the node itself."""
    spec = PortSpec(dtype=torch.Tensor, shape=(1,))
    assert repr(OutputPort(_Node(id="N1"), "out", spec)) == "OutputPort(N1.out)"
    rep = repr(InputPort(_Node(), "in", spec))
    assert rep.startswith("InputPort(") and rep.endswith(".in)")


# ---------------------------------------------------------------------------
# Hypothesis property layer
# ---------------------------------------------------------------------------
_dtypes = st.sampled_from(
    [torch.Tensor, torch.float32, torch.float64, torch.int64, torch.bool, torch.uint8]
)
_static_shapes = st.lists(st.integers(min_value=-1, max_value=6), max_size=4).map(tuple)


@given(dtype=_dtypes, shape=_static_shapes)
@settings(max_examples=50)
def test_property_reflexive_compatibility(dtype, shape):
    """A spec with a static shape is always compatible with itself."""
    spec = PortSpec(dtype=dtype, shape=shape)
    assert spec.is_compatible_with(spec, None, None)[0]


@given(shape=st.lists(st.integers(min_value=0, max_value=6), max_size=4).map(tuple))
@settings(max_examples=50)
def test_property_generic_tensor_matches_specific(shape):
    """Generic torch.Tensor is compatible with any same-rank specific dtype."""
    src = PortSpec(dtype=torch.Tensor, shape=shape)
    tgt = PortSpec(dtype=torch.float32, shape=shape)
    assert src.is_compatible_with(tgt, None, None)[0]


@given(
    a=st.lists(st.integers(min_value=0, max_value=6), max_size=3).map(tuple),
    b=st.lists(st.integers(min_value=0, max_value=6), max_size=3).map(tuple),
)
@settings(max_examples=50)
def test_property_rank_mismatch_always_incompatible(a, b):
    """Same dtype but different rank is always incompatible."""
    if len(a) == len(b):
        return
    ok, msg = PortSpec(dtype=torch.Tensor, shape=a).is_compatible_with(
        PortSpec(dtype=torch.Tensor, shape=b), None, None
    )
    assert not ok
    assert "rank mismatch" in msg


@given(
    shape=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=4).map(tuple),
    idx=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=50)
def test_property_wildcard_never_dim_fails(shape, idx):
    """A -1 at any position keeps a same-rank, same-dtype pair compatible."""
    idx %= len(shape)
    src_shape = shape[:idx] + (-1,) + shape[idx + 1 :]
    tgt_shape = shape[:idx] + (shape[idx] + 1,) + shape[idx + 1 :]
    assert PortSpec(dtype=torch.Tensor, shape=src_shape).is_compatible_with(
        PortSpec(dtype=torch.Tensor, shape=tgt_shape), None, None
    )[0]
