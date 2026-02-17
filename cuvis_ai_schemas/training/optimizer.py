"""Optimizer configuration schema."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    try:
        from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
    except ImportError:
        cuvis_ai_pb2 = None  # type: ignore[assignment]


class OptimizerConfig(BaseModel):
    """Optimizer configuration with constraints and documentation."""

    name: str = Field(
        default="adamw",
        description="Optimizer type: adamw, sgd, adam",
    )
    lr: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Learning rate",
        json_schema_extra={"minimum": 1e-6},
    )
    weight_decay: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="L2 regularization coefficient",
    )
    momentum: float | None = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Momentum factor (for SGD)",
    )
    betas: tuple[float, float] | None = Field(default=None, description="Adam betas (beta1, beta2)")

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.01,
                }
            ]
        },
    )

    @field_validator("betas")
    @classmethod
    def _validate_betas(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        """Validate that betas is a tuple of exactly 2 floats."""
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("betas must be a tuple of length 2")
        return value

    @field_validator("lr")
    @classmethod
    def _validate_lr(cls, value: float) -> float:
        """Validate that learning rate is non-zero."""
        if value == 0:
            raise ValueError("Learning rate must be non-zero")
        return value

    def to_proto(self) -> Any:
        """Convert to protobuf message (requires proto extra)."""
        try:
            from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
        except ImportError as e:
            raise ImportError(
                "Proto support requires the 'proto' extra: pip install cuvis-ai-schemas[proto]"
            ) from e

        return cuvis_ai_pb2.OptimizerConfig(config_bytes=self.model_dump_json().encode("utf-8"))

    @classmethod
    def from_proto(cls, proto_config):
        """Create from protobuf message (requires proto extra)."""
        return cls.model_validate_json(proto_config.config_bytes.decode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizerConfig:
        """Create from dictionary."""
        return cls.model_validate(data)


__all__ = ["OptimizerConfig"]
