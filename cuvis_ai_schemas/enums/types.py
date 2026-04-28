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


class NodeCategory(StrEnum):
    """Node category — graph role / execution contract. Exclusive.

    One value per node. Drives serialization, checkpointing, gradient
    routing, and the palette section a node lands in.
    """

    UNSPECIFIED = "unspecified"
    SOURCE = "source"
    SINK = "sink"
    TRANSFORM = "transform"
    MODEL = "model"
    LOSS = "loss"
    METRIC = "metric"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    REGULARIZER = "regularizer"
    RUNNER = "runner"
    VISUALIZER = "visualizer"
    CONTROL = "control"

    def get_display_name(self) -> str:
        """Title-cased name (e.g. ``'Visualizer'``, ``'Regularizer'``)."""
        return self.value.title()


class NodeTag(StrEnum):
    """Node tag — orthogonal axes (modality, task, lifecycle, properties, backend).

    Zero or more values per node. Combinations are free. Drives search,
    filters, capability checks, and badges.
    """

    UNSPECIFIED = "unspecified"

    # Modality
    IMAGE = "image"
    VIDEO = "video"
    RGB = "rgb"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    POINT_CLOUD = "point_cloud"
    DEPTH = "depth"
    MASK = "mask"
    BBOX = "bbox"
    KEYPOINTS = "keypoints"
    TEXT = "text"
    AUDIO = "audio"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    METADATA = "metadata"
    EMBEDDING = "embedding"

    # Task
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    TRACKING = "tracking"
    REGRESSION = "regression"
    GENERATION = "generation"
    RECONSTRUCTION = "reconstruction"
    DENOISING = "denoising"
    UNMIXING = "unmixing"
    DIM_REDUCTION = "dim_reduction"
    CLUSTERING = "clustering"
    ANOMALY = "anomaly"
    RETRIEVAL = "retrieval"

    # Lifecycle
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"
    AUGMENTATION = "augmentation"
    CALIBRATION = "calibration"
    NORMALIZATION = "normalization"
    TRAINING = "training"
    EVALUATION = "evaluation"
    INFERENCE = "inference"

    # Properties
    LEARNABLE = "learnable"
    DIFFERENTIABLE = "differentiable"
    STOCHASTIC = "stochastic"
    INVERTIBLE = "invertible"
    STREAMING = "streaming"
    BATCHED = "batched"
    STATEFUL = "stateful"

    # Backend
    TORCH = "torch"
    NUMPY = "numpy"
    JAX = "jax"
    ONNX = "onnx"
