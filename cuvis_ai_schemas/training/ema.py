"""Weight averaging (EMA) of a gradient-training run."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from cuvis_ai_schemas.base import BaseSchemaModel

EmaWeights = Literal["ema", "live"]


class EmaConfig(BaseSchemaModel):
    """Exponential moving average of the trained weights.

    The training runtime keeps an average of every trainable parameter and every
    floating-point buffer of the pipeline. It updates the average after every
    ``update_interval`` optimizer steps and, with ``update_on_epoch_end``, once more at
    the end of every training epoch. The ``n``-th update moves the average towards the
    live weights: ``average = average * d + live * (1 - d)`` with
    ``d = decay * (1 - exp(-n / tau))``, or ``d = decay`` when ``tau`` is 0; the first
    update copies the live weights.

    ``evaluate`` picks the weights that validation, test, the best-checkpoint choice and
    the saved pipeline use: ``ema`` (the average) or ``live`` (the trained weights; the
    average is then only kept in the checkpoints).
    """

    decay: float = Field(
        default=0.999,
        gt=0.0,
        le=1.0,
        description="Weight of the old average per update, in (0, 1]",
    )
    tau: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Warm-up time constant in updates: the decay ramps up as "
            "decay * (1 - exp(-n / tau)); 0 keeps it constant"
        ),
    )
    update_interval: int = Field(default=1, ge=1, description="Optimizer steps between two updates")
    update_on_epoch_end: bool = Field(
        default=False, description="Also update once at the end of every training epoch"
    )
    evaluate: EmaWeights = Field(
        default="ema",
        description="Weights used for validation, test, checkpoint choice and saving",
    )
