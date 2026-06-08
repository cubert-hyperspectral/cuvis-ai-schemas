"""Tests for the UI port-display wrapper."""

from cuvis_ai_schemas.extensions.ui.port_display import (
    DEFAULT_COLOR,
    DTYPE_COLORS,
    PortDisplaySpec,
)
from cuvis_ai_schemas.pipeline import PortSpec


def test_color_known_dtype():
    """A known dtype string maps to its configured color."""
    display = PortDisplaySpec(PortSpec(dtype="torch.Tensor", shape=(1, 3, 224, 224)))
    assert display.color == DTYPE_COLORS["torch.Tensor"]


def test_color_unknown_dtype_falls_back():
    """An unrecognized dtype falls back to the default color."""
    display = PortDisplaySpec(PortSpec(dtype="something_unmapped", shape=(-1,)))
    assert display.color == DEFAULT_COLOR


def test_format_tooltip_includes_variadic():
    """A variadic port spec surfaces the [Variadic] tag in the tooltip."""
    display = PortDisplaySpec(PortSpec(dtype="torch.Tensor", shape=(-1,), variadic=True))
    tooltip = display.format_tooltip()
    assert "[Variadic]" in tooltip


def test_format_tooltip_omits_variadic_by_default():
    """A non-variadic port spec does not emit the [Variadic] tag."""
    display = PortDisplaySpec(PortSpec(dtype="torch.Tensor", shape=(-1,)))
    tooltip = display.format_tooltip()
    assert "[Variadic]" not in tooltip


def test_format_tooltip_includes_optional():
    """An optional port spec surfaces the [Optional] tag in the tooltip."""
    display = PortDisplaySpec(PortSpec(dtype="torch.Tensor", shape=(-1,), optional=True))
    assert "[Optional]" in display.format_tooltip()
