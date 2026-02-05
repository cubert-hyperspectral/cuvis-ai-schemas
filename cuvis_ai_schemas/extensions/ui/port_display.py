"""UI display extensions for PortSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuvis_ai_schemas.pipeline.ports import PortSpec

# Default RGB colors for different data types
# Extended with hyperspectral/cuvis-ai specific types for comprehensive UI support
DTYPE_COLORS: dict[str, tuple[int, int, int]] = {
    # === PyTorch Tensor Types ===
    "torch.Tensor": (100, 150, 255),  # Blue for generic tensors
    "tensor": (255, 200, 150),  # Peach for generic tensor type
    # Floating point types - warm orange/yellow tones
    "torch.float16": (255, 180, 80),  # Light orange
    "torch.float32": (255, 150, 50),  # Orange
    "torch.float64": (255, 120, 30),  # Dark orange
    "torch.bfloat16": (255, 200, 100),  # Pale orange
    "float16": (255, 180, 80),
    "float32": (255, 150, 50),
    "float64": (255, 120, 30),
    "float": (150, 255, 150),  # Green for Python floats
    # Integer types - cool blue tones
    "torch.int8": (150, 200, 255),  # Light blue
    "torch.int16": (120, 180, 255),  # Medium blue
    "torch.int32": (100, 160, 255),  # Blue
    "torch.int64": (80, 140, 255),  # Dark blue
    "torch.uint8": (180, 220, 255),  # Pale blue
    "torch.uint16": (160, 210, 255),  # Light pale blue
    "int8": (150, 200, 255),
    "int16": (120, 180, 255),
    "int32": (100, 160, 255),
    "int64": (80, 140, 255),
    "uint8": (180, 220, 255),
    "uint16": (160, 210, 255),
    "int": (255, 200, 100),  # Orange for Python ints
    # Boolean types - green
    "torch.bool": (100, 220, 100),  # Green
    "bool": (255, 150, 150),  # Pink for Python bools (keep original for backward compat)
    # Complex types - purple tones
    "torch.complex64": (200, 150, 255),  # Light purple
    "torch.complex128": (180, 120, 255),  # Purple
    # === Python Built-in Types ===
    "str": (200, 200, 200),  # Gray for strings
    "list": (180, 180, 255),  # Light blue for lists
    "dict": (255, 180, 255),  # Light purple for dicts
    "object": (180, 180, 180),  # Gray for generic objects
    # === Hyperspectral/Cuvis-AI Specific Types ===
    "cube": (255, 180, 100),  # Orange-ish (hyperspectral data cube)
    "mask": (150, 255, 150),  # Light green (binary/segmentation mask)
    "wavelengths": (100, 200, 255),  # Light blue (spectral wavelength array)
    "bbox": (255, 150, 150),  # Light red (bounding box coordinates)
    "points": (200, 255, 200),  # Pale green (point cloud/coordinates)
    "labels": (255, 200, 200),  # Light pink (classification labels)
    "indices": (200, 180, 255),  # Light purple (index array)
    "scalar": (255, 255, 180),  # Light yellow (single value)
    "image": (200, 255, 255),  # Light cyan (2D image output)
    # === Special/Generic Types ===
    "any": (255, 255, 255),  # White (accepts anything)
}

DEFAULT_COLOR = (200, 200, 200)  # Light gray for unknown types


class PortDisplaySpec:
    """UI display wrapper for PortSpec providing color and formatting.

    This class wraps a PortSpec and adds UI-specific display properties
    like colors and formatted tooltips.

    Attributes
    ----------
    port_spec : PortSpec
        The underlying port specification

    Examples
    --------
    >>> from cuvis_ai_schemas.pipeline.ports import PortSpec
    >>> port = PortSpec(dtype="torch.Tensor", shape=(1, 3, 224, 224))
    >>> display = PortDisplaySpec(port)
    >>> display.color
    (100, 150, 255)
    >>> display.format_tooltip()
    'torch.Tensor\\nShape: (1, 3, 224, 224)'
    """

    def __init__(self, port_spec: PortSpec) -> None:
        """Initialize display spec.

        Parameters
        ----------
        port_spec : PortSpec
            Port specification to wrap
        """
        self.port_spec = port_spec

    @property
    def color(self) -> tuple[int, int, int]:
        """Get RGB color for this port's dtype.

        Returns
        -------
        tuple[int, int, int]
            RGB color tuple (0-255 range)
        """
        dtype_str = str(self.port_spec.dtype)

        # Try exact match first
        if dtype_str in DTYPE_COLORS:
            return DTYPE_COLORS[dtype_str]

        # Try to extract typename for partial matches
        if hasattr(self.port_spec.dtype, "__name__"):
            typename = self.port_spec.dtype.__name__
            if typename in DTYPE_COLORS:
                return DTYPE_COLORS[typename]

        # Check if it contains "Tensor"
        if "Tensor" in dtype_str or "tensor" in dtype_str:
            return DTYPE_COLORS["torch.Tensor"]

        return DEFAULT_COLOR

    @property
    def display_name(self) -> str:
        """Get formatted display name for the dtype.

        Returns
        -------
        str
            Human-readable dtype name
        """
        if hasattr(self.port_spec.dtype, "__name__"):
            return self.port_spec.dtype.__name__
        return str(self.port_spec.dtype)

    def format_tooltip(self) -> str:
        """Format a tooltip string for UI display.

        Returns
        -------
        str
            Formatted tooltip with dtype, shape, and description
        """
        lines = [self.display_name]

        if self.port_spec.shape:
            shape_str = ", ".join(str(d) for d in self.port_spec.shape)
            lines.append(f"Shape: ({shape_str})")

        if self.port_spec.description:
            lines.append(f"\\n{self.port_spec.description}")

        if self.port_spec.optional:
            lines.append("[Optional]")

        return "\\n".join(lines)
