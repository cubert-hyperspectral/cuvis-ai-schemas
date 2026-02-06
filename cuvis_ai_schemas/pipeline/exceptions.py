"""Pipeline exceptions."""

from __future__ import annotations


class PortCompatibilityError(Exception):
    """Raised when attempting to connect incompatible ports."""


__all__ = ["PortCompatibilityError"]
