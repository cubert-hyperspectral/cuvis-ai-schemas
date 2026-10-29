"""Calibration objective of a training run."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from cuvis_ai_schemas.base import BaseSchemaModel

CalibrationObjective = Literal["f1", "precision", "recall"]
CalibrationLevel = Literal["pixels", "frames"]


class CalibrationConfig(BaseSchemaModel):
    """What the deciders' thresholds are calibrated to on the validation split.

    ``f1`` keeps the thresholds with the best F1, by each decider's own rule. ``precision``
    and ``recall`` take a ``target``: among the candidate thresholds that reach it at the
    chosen ``level``, the calibration keeps the one with the best other metric, and the
    closest one when none reaches it. ``level`` says where precision and recall are counted:
    ``pixels`` counts every pixel, ``frames`` counts a frame as flagged when any of its
    pixels is.
    """

    objective: CalibrationObjective = Field(
        default="f1", description="Metric the thresholds are chosen for"
    )
    target: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Precision or recall to reach, in [0, 1]; required for those objectives",
    )
    level: CalibrationLevel = Field(
        default="pixels", description="Where precision and recall are counted"
    )

    @field_validator("target", mode="before")
    @classmethod
    def _reject_bool_target(cls, value: object) -> object:
        """Refuse a boolean, which pydantic would otherwise read as 0.0 or 1.0."""
        if isinstance(value, bool):
            raise ValueError("target must be a number in [0, 1], not a boolean")
        return value

    @model_validator(mode="after")
    def _target_and_level_fit_the_objective(self) -> CalibrationConfig:
        """F1 takes no target and no level of its own; precision and recall need a target."""
        if self.objective == "f1":
            if self.target is not None:
                raise ValueError(
                    "The f1 objective takes no target; set objective to precision or recall "
                    "to calibrate to a target."
                )
            if self.level != "pixels":
                raise ValueError(
                    "The f1 objective keeps its own rule and takes no level; set objective "
                    "to precision or recall to calibrate at the frame level."
                )
        elif self.target is None:
            raise ValueError(f"The {self.objective} objective needs a target in [0, 1].")
        return self


__all__ = ["CalibrationConfig", "CalibrationLevel", "CalibrationObjective"]
