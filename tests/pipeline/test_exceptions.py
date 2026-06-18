"""Tests for pipeline exceptions."""

from __future__ import annotations

import pytest

from cuvis_ai_schemas.pipeline import PortCompatibilityError


def test_is_an_exception_subclass():
    """PortCompatibilityError is a plain Exception subclass."""
    assert issubclass(PortCompatibilityError, Exception)


def test_raise_and_catch_preserves_message():
    """The error round-trips its message when raised and caught."""
    with pytest.raises(PortCompatibilityError, match="ports x and y"):
        raise PortCompatibilityError("ports x and y are incompatible")
