"""Shared type enums for cuvis-ai ecosystem."""

from enum import StrEnum


class ExecutionStage(StrEnum):
    """Execution stages for node filtering.

    Nodes can specify which stages they should execute in to enable
    stage-aware graph execution (e.g., loss nodes only in training).
    """

    ALWAYS = "always"
    TRAIN = "train"
    VAL = "val"
    VALIDATE = "val"
    TEST = "test"
    INFERENCE = "inference"


class ArtifactType(StrEnum):
    """Types of artifacts with different validation/logging policies.

    Attributes
    ----------
    IMAGE : str
        Image artifact - expects shape (H, W, 1) monocular or (H, W, 3) RGB
    """

    IMAGE = "image"
