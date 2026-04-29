"""UI-specific extensions for port and node display."""

from cuvis_ai_schemas.extensions.ui.node_display import (
    CATEGORY_STYLES,
    TAG_STYLES,
    is_plugin,
    resolve_display,
)
from cuvis_ai_schemas.extensions.ui.port_display import (
    DTYPE_COLORS,
    PortDisplaySpec,
)

__all__ = [
    "PortDisplaySpec",
    "DTYPE_COLORS",
    "CATEGORY_STYLES",
    "TAG_STYLES",
    "resolve_display",
    "is_plugin",
]
