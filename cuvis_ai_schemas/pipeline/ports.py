"""
Port system for typed I/O in cuvis.ai pipelines.

This module provides:
- PortSpec: Specification for input/output ports with type and shape constraints
- InputPort/OutputPort: Proxy objects representing node ports
- DimensionResolver: Utility for resolving symbolic dimensions
- PortCompatibilityError: Exception for incompatible connections
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cuvis_ai_schemas.pipeline.exceptions import PortCompatibilityError


class DimensionResolver:
    """Utility class for resolving symbolic dimensions in port shapes."""

    @staticmethod
    def resolve(
        shape: tuple[int | str, ...],
        node: Any | None,
    ) -> tuple[int, ...]:
        """Resolve symbolic dimensions to concrete values.

        Parameters
        ----------
        shape : tuple[int | str, ...]
            Shape specification with flexible (-1), fixed (int), or symbolic (str) dims.
        node : Any | None
            Node instance to resolve symbolic dimensions from.

        Returns
        -------
        tuple[int, ...]
            Resolved shape with concrete integer values.

        Raises
        ------
        AttributeError
            If symbolic dimension references non-existent node attribute.
        """
        resolved: list[int] = []
        for dim in shape:
            if isinstance(dim, int):
                # Flexible (-1) or fixed (int) dimension
                resolved.append(dim)
                continue

            if isinstance(dim, str):
                # Symbolic dimension - resolve from node
                if node is None:
                    raise ValueError(
                        f"Cannot resolve symbolic dimension '{dim}' without node instance"
                    )
                if not hasattr(node, dim):
                    node_label = getattr(node, "id", None) or node
                    raise AttributeError(
                        f"Node {node_label} has no attribute '{dim}' for dimension resolution"
                    )

                value = getattr(node, dim)
                if not isinstance(value, int):
                    raise TypeError(f"Dimension '{dim}' resolved to {type(value)}, expected int")
                resolved.append(value)
                continue

            raise TypeError(f"Invalid dimension type: {type(dim)}")

        return tuple(resolved)


@dataclass
class PortSpec:
    """Specification for a node input or output port."""

    dtype: Any
    shape: tuple[int | str, ...]
    description: str = ""
    optional: bool = False

    def resolve_shape(self, node: Any) -> tuple[int, ...]:
        """Resolve symbolic dimensions in shape using node attributes.

        Parameters
        ----------
        node : Any
            Node instance to resolve symbolic dimensions from.

        Returns
        -------
        tuple[int, ...]
            Resolved shape with all symbolic dimensions replaced by concrete integer values.

        See Also
        --------
        DimensionResolver.resolve : Underlying resolution logic.
        """
        return DimensionResolver.resolve(self.shape, node)

    def is_compatible_with(
        self,
        other: PortSpec | list[PortSpec],
        source_node: Any | None,
        target_node: Any | None,
    ) -> tuple[bool, str]:
        """Check if this port can connect to another port.

        Parameters
        ----------
        other : PortSpec | list[PortSpec]
            Target port spec. If a list, it's a variadic port - extract the spec.
        source_node : Any | None
            Source node for dimension resolution
        target_node : Any | None
            Target node for dimension resolution

        Returns
        -------
        tuple[bool, str]
            (is_compatible, error_message)
        """

        def _format_dtype(value: Any) -> str:
            """Format a dtype value for display in error messages.

            Parameters
            ----------
            value : Any
                A dtype value (torch.dtype, type, or other).

            Returns
            -------
            str
                Human-readable string representation of the dtype.
            """
            if isinstance(value, torch.dtype):
                return str(value)
            return getattr(value, "__name__", str(value))

        def _is_tensor_related(dtype: Any) -> bool:
            """Check if dtype is torch.Tensor or a specific torch.dtype.

            Parameters
            ----------
            dtype : Any
                The dtype to check.

            Returns
            -------
            bool
                True if dtype is torch.Tensor or a torch.dtype instance.
            """
            return dtype is torch.Tensor or isinstance(dtype, torch.dtype)

        # Handle variadic ports (list-based specs)
        if isinstance(other, list):
            if not other:
                return False, "Variadic port has empty spec list"
            # Extract the actual PortSpec from the list
            other = other[0]

        # Check dtype compatibility with smart tensor handling
        source_is_tensor = _is_tensor_related(self.dtype)
        target_is_tensor = _is_tensor_related(other.dtype)

        if source_is_tensor and target_is_tensor:
            # Both tensor-related types
            # Allow if either is generic torch.Tensor OR both are same dtype
            if not (
                self.dtype is torch.Tensor
                or other.dtype is torch.Tensor
                or self.dtype == other.dtype
            ):
                return False, (
                    f"Dtype mismatch: source has {_format_dtype(self.dtype)}, "
                    f"target expects {_format_dtype(other.dtype)}"
                )
        elif self.dtype != other.dtype:
            # Non-tensor types must match exactly
            return False, (
                f"Dtype mismatch: source has {_format_dtype(self.dtype)}, "
                f"target expects {_format_dtype(other.dtype)}"
            )

        # Resolve shapes
        try:
            source_shape = self.resolve_shape(source_node) if source_node else self.shape
            target_shape = other.resolve_shape(target_node) if target_node else other.shape
        except (AttributeError, ValueError, TypeError) as exc:
            return False, f"Shape resolution failed: {exc}"

        # Check rank compatibility
        if len(source_shape) != len(target_shape):
            return False, (
                f"Shape rank mismatch: source has {len(source_shape)} dimensions, "
                f"target expects {len(target_shape)}"
            )

        # Check dimension-by-dimension compatibility
        for idx, (src_dim, tgt_dim) in enumerate(zip(source_shape, target_shape, strict=True)):
            # -1 means flexible, always compatible
            if src_dim == -1 or tgt_dim == -1:
                continue

            # Both fixed - must match exactly
            if src_dim != tgt_dim:
                return False, (
                    f"Dimension {idx} mismatch: source has size {src_dim}, target expects {tgt_dim}"
                )

        return True, ""


class OutputPort:
    """Proxy object representing a node's output port."""

    def __init__(self, node: Any, name: str, spec: PortSpec) -> None:
        """Initialize an output port proxy.

        Parameters
        ----------
        node : Any
            The node instance that owns this port.
        name : str
            The name of the port on the node.
        spec : PortSpec
            The port specification defining type and shape constraints.
        """
        self.node = node
        self.name = name
        self.spec = spec

    def __repr__(self) -> str:
        """Return string representation of the output port.

        Returns
        -------
        str
            String in format "OutputPort(node_id.port_name)".
        """
        node_id = getattr(self.node, "id", None) or self.node
        return f"OutputPort({node_id}.{self.name})"


class InputPort:
    """Proxy object representing a node's input port."""

    def __init__(self, node: Any, name: str, spec: PortSpec) -> None:
        """Initialize an input port proxy.

        Parameters
        ----------
        node : Any
            The node instance that owns this port.
        name : str
            The name of the port on the node.
        spec : PortSpec
            The port specification defining type and shape constraints.
        """
        self.node = node
        self.name = name
        self.spec = spec

    def __repr__(self) -> str:
        """Return string representation of the input port.

        Returns
        -------
        str
            String in format "InputPort(node_id.port_name)".
        """
        node_id = getattr(self.node, "id", None) or self.node
        return f"InputPort({node_id}.{self.name})"


__all__ = [
    "DimensionResolver",
    "InputPort",
    "OutputPort",
    "PortCompatibilityError",
    "PortSpec",
]
