"""Monitoring schemas for artifacts and metrics."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cuvis_ai_schemas.enums.types import ArtifactType, ExecutionStage

if TYPE_CHECKING:
    import numpy as np


@dataclass
class Artifact:
    """Artifact for logging visualizations and data to monitoring systems.

    Attributes
    ----------
    name : str
        Name/identifier for the artifact
    value : np.ndarray
        Numpy array containing the artifact data (shape validated by type)
    el_id : int
        Element ID (e.g., batch item index, image index)
    desc : str
        Human-readable description of the artifact
    type : ArtifactType
        Type of artifact, determines validation and logging policy
    stage : ExecutionStage
        Execution stage when artifact was generated
    epoch : int
        Training epoch when artifact was generated
    batch_idx : int
        Batch index when artifact was generated

    Examples
    --------
    >>> import numpy as np
    >>> artifact = Artifact(
    ...     name="heatmap_img_0",
    ...     value=np.random.rand(256, 256, 3),
    ...     el_id=0,
    ...     desc="Anomaly heatmap for first image",
    ...     type=ArtifactType.IMAGE
    ... )
    """

    name: str
    value: "np.ndarray"
    el_id: int
    desc: str
    type: ArtifactType
    stage: ExecutionStage = ExecutionStage.INFERENCE
    epoch: int = 0
    batch_idx: int = 0


@dataclass
class Metric:
    """Metric for logging scalar values to monitoring systems.

    Attributes
    ----------
    name : str
        Name/identifier for the metric
    value : float
        Scalar metric value
    stage : ExecutionStage
        Execution stage when metric was recorded
    epoch : int
        Training epoch when metric was recorded
    batch_idx : int
        Batch index when metric was recorded

    Examples
    --------
    >>> metric = Metric(name="loss/train", value=0.123, stage=ExecutionStage.TRAIN)
    """

    name: str
    value: float
    stage: ExecutionStage = ExecutionStage.INFERENCE
    epoch: int = 0
    batch_idx: int = 0
