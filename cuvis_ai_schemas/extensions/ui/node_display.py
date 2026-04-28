"""Display metadata (category styles, tag styles) for node taxonomy.

Consumed by both the pipeline visualizer (cuvis-ai-core) and the Qt UI palette
(cuvis-ai-ui), so that no consumer has to maintain its own copy of the styling
tables.

The icon filename for any category is ``f"{category.value}.svg"`` (resolved by
the icon helper introduced in cuvis-ai-core); it is *not* duplicated as a key
here — the enum value is the single source of truth for the filename.
"""

from typing import Any

from cuvis_ai_schemas.enums import NodeCategory, NodeTag

CATEGORY_STYLES: dict[NodeCategory, dict[str, str]] = {
    NodeCategory.UNSPECIFIED: {"fill": "#F8F9FB", "border": "#AAAAAA", "emoji": "⚙️"},
    NodeCategory.SOURCE: {"fill": "#FFF5E6", "border": "#E8A33D", "emoji": "📦"},
    NodeCategory.SINK: {"fill": "#FFF5E6", "border": "#E8A33D", "emoji": "💾"},
    NodeCategory.TRANSFORM: {"fill": "#EAF5EA", "border": "#6FAE5C", "emoji": "🔄"},
    NodeCategory.MODEL: {"fill": "#EAF0FF", "border": "#4E7BD4", "emoji": "🧠"},
    NodeCategory.LOSS: {"fill": "#F6EAFF", "border": "#9A5DD4", "emoji": "📉"},
    NodeCategory.METRIC: {"fill": "#FFF0F0", "border": "#C9615A", "emoji": "📈"},
    NodeCategory.OPTIMIZER: {"fill": "#F6EAFF", "border": "#9A5DD4", "emoji": "⚙️"},
    NodeCategory.SCHEDULER: {"fill": "#F6EAFF", "border": "#9A5DD4", "emoji": "📅"},
    NodeCategory.REGULARIZER: {"fill": "#F6EAFF", "border": "#9A5DD4", "emoji": "🪶"},
    NodeCategory.RUNNER: {"fill": "#FFF7CC", "border": "#DCB43C", "emoji": "▶️"},
    NodeCategory.VISUALIZER: {"fill": "#F6EAFF", "border": "#9A5DD4", "emoji": "📊"},
    NodeCategory.CONTROL: {"fill": "#FFF7CC", "border": "#DCB43C", "emoji": "🔀"},
}


# Tags share a colour family within their sub-namespace so the palette renders
# coherent groups of chips. Modality = blue family, task = green family,
# lifecycle = orange family, properties = grey, backend = small monochrome.
TAG_STYLES: dict[NodeTag, dict[str, str]] = {
    NodeTag.UNSPECIFIED: {"badge_color": "#AAAAAA", "short_label": "?"},
    # Modality (blue family)
    NodeTag.IMAGE: {"badge_color": "#4E7BD4", "short_label": "img"},
    NodeTag.VIDEO: {"badge_color": "#4E7BD4", "short_label": "vid"},
    NodeTag.RGB: {"badge_color": "#4E7BD4", "short_label": "rgb"},
    NodeTag.MULTISPECTRAL: {"badge_color": "#4E7BD4", "short_label": "msi"},
    NodeTag.HYPERSPECTRAL: {"badge_color": "#4E7BD4", "short_label": "hsi"},
    NodeTag.POINT_CLOUD: {"badge_color": "#4E7BD4", "short_label": "pcd"},
    NodeTag.DEPTH: {"badge_color": "#4E7BD4", "short_label": "depth"},
    NodeTag.MASK: {"badge_color": "#4E7BD4", "short_label": "mask"},
    NodeTag.BBOX: {"badge_color": "#4E7BD4", "short_label": "bbox"},
    NodeTag.KEYPOINTS: {"badge_color": "#4E7BD4", "short_label": "kp"},
    NodeTag.TEXT: {"badge_color": "#4E7BD4", "short_label": "text"},
    NodeTag.AUDIO: {"badge_color": "#4E7BD4", "short_label": "audio"},
    NodeTag.TABULAR: {"badge_color": "#4E7BD4", "short_label": "tab"},
    NodeTag.TIME_SERIES: {"badge_color": "#4E7BD4", "short_label": "ts"},
    NodeTag.METADATA: {"badge_color": "#4E7BD4", "short_label": "meta"},
    NodeTag.EMBEDDING: {"badge_color": "#4E7BD4", "short_label": "emb"},
    # Task (green family)
    NodeTag.CLASSIFICATION: {"badge_color": "#6FAE5C", "short_label": "class"},
    NodeTag.SEGMENTATION: {"badge_color": "#6FAE5C", "short_label": "seg"},
    NodeTag.DETECTION: {"badge_color": "#6FAE5C", "short_label": "det"},
    NodeTag.TRACKING: {"badge_color": "#6FAE5C", "short_label": "track"},
    NodeTag.REGRESSION: {"badge_color": "#6FAE5C", "short_label": "reg"},
    NodeTag.GENERATION: {"badge_color": "#6FAE5C", "short_label": "gen"},
    NodeTag.RECONSTRUCTION: {"badge_color": "#6FAE5C", "short_label": "recon"},
    NodeTag.DENOISING: {"badge_color": "#6FAE5C", "short_label": "denoise"},
    NodeTag.UNMIXING: {"badge_color": "#6FAE5C", "short_label": "unmix"},
    NodeTag.DIM_REDUCTION: {"badge_color": "#6FAE5C", "short_label": "dim-red"},
    NodeTag.CLUSTERING: {"badge_color": "#6FAE5C", "short_label": "cluster"},
    NodeTag.ANOMALY: {"badge_color": "#6FAE5C", "short_label": "anom"},
    NodeTag.RETRIEVAL: {"badge_color": "#6FAE5C", "short_label": "retrieval"},
    # Lifecycle (orange family)
    NodeTag.PREPROCESSING: {"badge_color": "#E8A33D", "short_label": "pre"},
    NodeTag.POSTPROCESSING: {"badge_color": "#E8A33D", "short_label": "post"},
    NodeTag.AUGMENTATION: {"badge_color": "#E8A33D", "short_label": "aug"},
    NodeTag.CALIBRATION: {"badge_color": "#E8A33D", "short_label": "calib"},
    NodeTag.NORMALIZATION: {"badge_color": "#E8A33D", "short_label": "norm"},
    NodeTag.TRAINING: {"badge_color": "#E8A33D", "short_label": "train"},
    NodeTag.EVALUATION: {"badge_color": "#E8A33D", "short_label": "eval"},
    NodeTag.INFERENCE: {"badge_color": "#E8A33D", "short_label": "infer"},
    # Properties (grey)
    NodeTag.LEARNABLE: {"badge_color": "#888888", "short_label": "learn"},
    NodeTag.DIFFERENTIABLE: {"badge_color": "#888888", "short_label": "diff"},
    NodeTag.STOCHASTIC: {"badge_color": "#888888", "short_label": "stoch"},
    NodeTag.INVERTIBLE: {"badge_color": "#888888", "short_label": "inv"},
    NodeTag.STREAMING: {"badge_color": "#888888", "short_label": "stream"},
    NodeTag.BATCHED: {"badge_color": "#888888", "short_label": "batched"},
    NodeTag.STATEFUL: {"badge_color": "#888888", "short_label": "stateful"},
    # Backend (small monochrome)
    NodeTag.TORCH: {"badge_color": "#444444", "short_label": "torch"},
    NodeTag.NUMPY: {"badge_color": "#444444", "short_label": "numpy"},
    NodeTag.JAX: {"badge_color": "#444444", "short_label": "jax"},
    NodeTag.ONNX: {"badge_color": "#444444", "short_label": "onnx"},
}


def resolve_display(node: Any) -> dict[str, Any]:
    """Resolve the display spec for a node.

    Reads ``node.get_category()`` and returns a dict carrying the keys
    ``category``, ``fill``, ``border``, ``emoji``, ``label`` (label is always
    ``None`` here; consumers may override).

    Falls back to ``CATEGORY_STYLES[NodeCategory.UNSPECIFIED]`` if the returned
    category is somehow missing from the table — defensive only; every enum
    value has an entry covered by the parametrized completeness test.
    """
    category = node.get_category()
    base = CATEGORY_STYLES.get(category, CATEGORY_STYLES[NodeCategory.UNSPECIFIED])
    return {
        "category": category,
        "fill": base["fill"],
        "border": base["border"],
        "emoji": base["emoji"],
        "label": None,
    }


def is_plugin(node: Any, registry: Any | None = None) -> bool:
    """Return True only when the node class comes from a plugin.

    If ``registry`` is supplied, returns ``True`` iff the class name is listed
    in its ``plugin_registry``. Otherwise returns ``False``. Builtins and
    shipped catalog repos are not flagged — the pill is reserved for genuinely
    external plugin classes.
    """
    if registry is None:
        return False
    return node.__class__.__name__ in getattr(registry, "plugin_registry", {})
