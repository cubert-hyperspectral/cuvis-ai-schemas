"""Port specification for node inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PortSpec:
    """Specification for a node input or output port.

    This is a lightweight schema definition. Full compatibility checking
    logic is implemented in cuvis-ai-core.

    Attributes
    ----------
    dtype : Any
        Data type for the port (e.g., torch.Tensor, torch.float32, int, str)
    shape : tuple[int | str, ...]
        Expected shape with:
        - Fixed dimensions: positive integers
        - Flexible dimensions: -1
        - Symbolic dimensions: strings (resolved from node attributes)
    description : str
        Human-readable description of the port
    optional : bool
        Whether the port is optional (for inputs)

    Examples
    --------
    >>> # Fixed shape tensor port
    >>> port = PortSpec(dtype=torch.Tensor, shape=(1, 3, 224, 224))

    >>> # Flexible batch dimension
    >>> port = PortSpec(dtype=torch.Tensor, shape=(-1, 3, 224, 224))

    >>> # Symbolic dimension from node attribute
    >>> port = PortSpec(dtype=torch.Tensor, shape=(-1, "num_channels", 224, 224))

    >>> # Scalar port
    >>> port = PortSpec(dtype=float, shape=())
    """

    dtype: Any
    shape: tuple[int | str, ...]
    description: str = ""
    optional: bool = False
