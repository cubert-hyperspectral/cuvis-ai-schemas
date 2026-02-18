"""Base schema model with shared serialization methods."""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


def _get_pb2() -> ModuleType:
    """Lazy-load the proto module."""
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError as exc:
        msg = "Proto support not installed. Install with: pip install cuvis-ai-schemas[proto]"
        raise ImportError(msg) from exc
    return cuvis_ai_pb2


class BaseSchemaModel(BaseModel):
    """Base model for all cuvis-ai schema configs.

    Provides:
    - Strict validation (``extra="forbid"``, ``validate_assignment=True``)
    - ``to_dict`` / ``from_dict`` / ``to_json`` / ``from_json``
    - Opt-in proto support via ``__proto_message__`` class variable
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    __proto_message__: ClassVar[str] = ""

    # ------------------------------------------------------------------
    # Dict / JSON serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Load from dictionary."""
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str):
        """Load from JSON string."""
        return cls.model_validate_json(payload)

    # ------------------------------------------------------------------
    # Proto serialization (opt-in)
    # ------------------------------------------------------------------

    def to_proto(self) -> Any:
        """Convert to protobuf message.

        Requires ``cuvis-ai-schemas[proto]`` and a ``__proto_message__``
        class variable naming the proto message type.
        """
        if not self.__proto_message__:
            raise NotImplementedError(
                f"{type(self).__name__} has no proto mapping (set __proto_message__ to enable)"
            )
        pb2 = _get_pb2()
        proto_cls = getattr(pb2, self.__proto_message__)
        return proto_cls(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config: Any):
        """Load from protobuf message."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))


__all__ = ["BaseSchemaModel"]
