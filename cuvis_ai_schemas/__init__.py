"""cuvis-ai-schemas: Lightweight schema definitions for the cuvis-ai ecosystem."""

from importlib.metadata import PackageNotFoundError, version

from cuvis_ai_schemas.base import BaseSchemaModel

try:
    __version__ = version("cuvis-ai-schemas")
except PackageNotFoundError:  # running from a source tree with no installed metadata
    __version__ = "0.0.0+unknown"

__all__ = ["BaseSchemaModel", "__version__"]
