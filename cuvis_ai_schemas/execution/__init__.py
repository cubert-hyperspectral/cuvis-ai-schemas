"""Execution context schemas."""

from collections.abc import Iterator
from typing import Any

from cuvis_ai_schemas.execution.context import Context
from cuvis_ai_schemas.execution.monitoring import Artifact, Metric

# Type alias for data streaming
InputStream = Iterator[dict[str, Any]]

__all__ = ["Context", "Artifact", "Metric", "InputStream"]
