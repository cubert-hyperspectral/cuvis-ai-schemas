"""Optimizer configuration schema."""

from __future__ import annotations

from typing import ClassVar

from pydantic import ConfigDict, Field

from cuvis_ai_schemas.base import BaseSchemaModel


class OptimizerConfig(BaseSchemaModel):
    """Optimizer configuration with constraints and documentation."""

    __proto_message__: ClassVar[str] = "OptimizerConfig"

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


__all__ = ["OptimizerConfig"]
