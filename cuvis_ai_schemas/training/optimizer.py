"""Optimizer configuration schema."""

from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator

from cuvis_ai_schemas.base import BaseSchemaModel


class OptimizerConfig(BaseSchemaModel):
    """Optimizer configuration with constraints and documentation."""

    __proto_message__: str = "OptimizerConfig"

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


__all__ = ["OptimizerConfig"]
