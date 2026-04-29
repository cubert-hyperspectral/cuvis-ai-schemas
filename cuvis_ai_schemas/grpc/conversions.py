"""Conversion helpers between Python ``NodeCategory`` / ``NodeTag`` enums and
their proto enum integer wire values.

Lives in cuvis-ai-schemas because both producers (the gRPC populator in
cuvis-ai-core) and consumers (the Qt UI client in cuvis-ai-ui) need the
helpers, and the UI already depends on schemas. Importing this module
requires the ``[proto]`` extra (it imports ``cuvis_ai_pb2``).
"""

from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

_CATEGORY_PY_TO_PROTO: dict[NodeCategory, int] = {
    NodeCategory.UNSPECIFIED: cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED,
    NodeCategory.SOURCE: cuvis_ai_pb2.NODE_CATEGORY_SOURCE,
    NodeCategory.SINK: cuvis_ai_pb2.NODE_CATEGORY_SINK,
    NodeCategory.TRANSFORM: cuvis_ai_pb2.NODE_CATEGORY_TRANSFORM,
    NodeCategory.MODEL: cuvis_ai_pb2.NODE_CATEGORY_MODEL,
    NodeCategory.LOSS: cuvis_ai_pb2.NODE_CATEGORY_LOSS,
    NodeCategory.METRIC: cuvis_ai_pb2.NODE_CATEGORY_METRIC,
    NodeCategory.OPTIMIZER: cuvis_ai_pb2.NODE_CATEGORY_OPTIMIZER,
    NodeCategory.SCHEDULER: cuvis_ai_pb2.NODE_CATEGORY_SCHEDULER,
    NodeCategory.REGULARIZER: cuvis_ai_pb2.NODE_CATEGORY_REGULARIZER,
    NodeCategory.RUNNER: cuvis_ai_pb2.NODE_CATEGORY_RUNNER,
    NodeCategory.VISUALIZER: cuvis_ai_pb2.NODE_CATEGORY_VISUALIZER,
    NodeCategory.CONTROL: cuvis_ai_pb2.NODE_CATEGORY_CONTROL,
}
_CATEGORY_PROTO_TO_PY: dict[int, NodeCategory] = {v: k for k, v in _CATEGORY_PY_TO_PROTO.items()}


_TAG_PY_TO_PROTO: dict[NodeTag, int] = {
    NodeTag.UNSPECIFIED: cuvis_ai_pb2.NODE_TAG_UNSPECIFIED,
    # Modality
    NodeTag.IMAGE: cuvis_ai_pb2.NODE_TAG_IMAGE,
    NodeTag.VIDEO: cuvis_ai_pb2.NODE_TAG_VIDEO,
    NodeTag.RGB: cuvis_ai_pb2.NODE_TAG_RGB,
    NodeTag.MULTISPECTRAL: cuvis_ai_pb2.NODE_TAG_MULTISPECTRAL,
    NodeTag.HYPERSPECTRAL: cuvis_ai_pb2.NODE_TAG_HYPERSPECTRAL,
    NodeTag.POINT_CLOUD: cuvis_ai_pb2.NODE_TAG_POINT_CLOUD,
    NodeTag.DEPTH: cuvis_ai_pb2.NODE_TAG_DEPTH,
    NodeTag.MASK: cuvis_ai_pb2.NODE_TAG_MASK,
    NodeTag.BBOX: cuvis_ai_pb2.NODE_TAG_BBOX,
    NodeTag.KEYPOINTS: cuvis_ai_pb2.NODE_TAG_KEYPOINTS,
    NodeTag.TEXT: cuvis_ai_pb2.NODE_TAG_TEXT,
    NodeTag.AUDIO: cuvis_ai_pb2.NODE_TAG_AUDIO,
    NodeTag.TABULAR: cuvis_ai_pb2.NODE_TAG_TABULAR,
    NodeTag.TIME_SERIES: cuvis_ai_pb2.NODE_TAG_TIME_SERIES,
    NodeTag.METADATA: cuvis_ai_pb2.NODE_TAG_METADATA,
    NodeTag.EMBEDDING: cuvis_ai_pb2.NODE_TAG_EMBEDDING,
    # Task
    NodeTag.CLASSIFICATION: cuvis_ai_pb2.NODE_TAG_CLASSIFICATION,
    NodeTag.SEGMENTATION: cuvis_ai_pb2.NODE_TAG_SEGMENTATION,
    NodeTag.DETECTION: cuvis_ai_pb2.NODE_TAG_DETECTION,
    NodeTag.TRACKING: cuvis_ai_pb2.NODE_TAG_TRACKING,
    NodeTag.REGRESSION: cuvis_ai_pb2.NODE_TAG_REGRESSION,
    NodeTag.GENERATION: cuvis_ai_pb2.NODE_TAG_GENERATION,
    NodeTag.RECONSTRUCTION: cuvis_ai_pb2.NODE_TAG_RECONSTRUCTION,
    NodeTag.DENOISING: cuvis_ai_pb2.NODE_TAG_DENOISING,
    NodeTag.UNMIXING: cuvis_ai_pb2.NODE_TAG_UNMIXING,
    NodeTag.DIM_REDUCTION: cuvis_ai_pb2.NODE_TAG_DIM_REDUCTION,
    NodeTag.CLUSTERING: cuvis_ai_pb2.NODE_TAG_CLUSTERING,
    NodeTag.ANOMALY: cuvis_ai_pb2.NODE_TAG_ANOMALY,
    NodeTag.RETRIEVAL: cuvis_ai_pb2.NODE_TAG_RETRIEVAL,
    # Lifecycle
    NodeTag.PREPROCESSING: cuvis_ai_pb2.NODE_TAG_PREPROCESSING,
    NodeTag.POSTPROCESSING: cuvis_ai_pb2.NODE_TAG_POSTPROCESSING,
    NodeTag.AUGMENTATION: cuvis_ai_pb2.NODE_TAG_AUGMENTATION,
    NodeTag.CALIBRATION: cuvis_ai_pb2.NODE_TAG_CALIBRATION,
    NodeTag.NORMALIZATION: cuvis_ai_pb2.NODE_TAG_NORMALIZATION,
    NodeTag.TRAINING: cuvis_ai_pb2.NODE_TAG_TRAINING,
    NodeTag.EVALUATION: cuvis_ai_pb2.NODE_TAG_EVALUATION,
    NodeTag.INFERENCE: cuvis_ai_pb2.NODE_TAG_INFERENCE,
    # Properties
    NodeTag.LEARNABLE: cuvis_ai_pb2.NODE_TAG_LEARNABLE,
    NodeTag.DIFFERENTIABLE: cuvis_ai_pb2.NODE_TAG_DIFFERENTIABLE,
    NodeTag.STOCHASTIC: cuvis_ai_pb2.NODE_TAG_STOCHASTIC,
    NodeTag.INVERTIBLE: cuvis_ai_pb2.NODE_TAG_INVERTIBLE,
    NodeTag.STREAMING: cuvis_ai_pb2.NODE_TAG_STREAMING,
    NodeTag.BATCHED: cuvis_ai_pb2.NODE_TAG_BATCHED,
    NodeTag.STATEFUL: cuvis_ai_pb2.NODE_TAG_STATEFUL,
    # Backend
    NodeTag.TORCH: cuvis_ai_pb2.NODE_TAG_TORCH,
    NodeTag.NUMPY: cuvis_ai_pb2.NODE_TAG_NUMPY,
    NodeTag.JAX: cuvis_ai_pb2.NODE_TAG_JAX,
    NodeTag.ONNX: cuvis_ai_pb2.NODE_TAG_ONNX,
}
_TAG_PROTO_TO_PY: dict[int, NodeTag] = {v: k for k, v in _TAG_PY_TO_PROTO.items()}


def node_category_to_proto(category: NodeCategory) -> int:
    """Map a Python ``NodeCategory`` to its proto enum integer."""
    return _CATEGORY_PY_TO_PROTO.get(category, cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED)


def proto_to_node_category(proto_value: int) -> NodeCategory:
    """Map a wire integer back to ``NodeCategory``.

    Unknown ints fall back to ``NodeCategory.UNSPECIFIED`` so forward-compat
    clients don't crash on a server that introduces new categories.
    """
    return _CATEGORY_PROTO_TO_PY.get(proto_value, NodeCategory.UNSPECIFIED)


def node_tag_to_proto(tag: NodeTag) -> int:
    """Map a Python ``NodeTag`` to its proto enum integer."""
    return _TAG_PY_TO_PROTO.get(tag, cuvis_ai_pb2.NODE_TAG_UNSPECIFIED)


def proto_to_node_tag(proto_value: int) -> NodeTag | None:
    """Map a wire integer back to ``NodeTag``.

    Unknown ints return ``None`` so consumers can skip them silently —
    important for forward compatibility when a server introduces new tags
    that this client doesn't recognise.
    """
    return _TAG_PROTO_TO_PY.get(proto_value)
