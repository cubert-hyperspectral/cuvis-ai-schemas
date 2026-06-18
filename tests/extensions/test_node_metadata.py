"""Tests for NodeCategory / NodeTag enums, display tables, conversions, and icons."""

import builtins
import runpy
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.extensions.ui.node_display import (
    CATEGORY_STYLES,
    TAG_STYLES,
    is_plugin,
    resolve_display,
)


def test_node_category_members():
    """NodeCategory has the 12 named values plus UNSPECIFIED."""
    expected = {
        "UNSPECIFIED",
        "SOURCE",
        "SINK",
        "TRANSFORM",
        "MODEL",
        "LOSS",
        "METRIC",
        "OPTIMIZER",
        "SCHEDULER",
        "REGULARIZER",
        "RUNNER",
        "VISUALIZER",
        "CONTROL",
    }
    actual = {m.name for m in NodeCategory}
    assert actual == expected
    assert len(NodeCategory) == 13


def test_node_tag_members_match_proto():
    """Every Python NodeTag value has a NODE_TAG_* proto counterpart with the
    same lowercased-name shape."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

    for tag in NodeTag:
        proto_name = f"NODE_TAG_{tag.name}"
        assert hasattr(cuvis_ai_pb2, proto_name), proto_name


@pytest.mark.parametrize("category", list(NodeCategory))
def test_node_category_display_name_title_cased(category):
    """get_display_name() returns the title-cased form."""
    assert category.get_display_name() == category.value.title()


@pytest.mark.parametrize("category", list(NodeCategory))
def test_category_styles_complete(category):
    """Every NodeCategory has a CATEGORY_STYLES entry with fill/border/emoji."""
    style = CATEGORY_STYLES[category]
    assert {"fill", "border", "emoji"} <= set(style.keys())


@pytest.mark.parametrize("tag", list(NodeTag))
def test_tag_styles_complete(tag):
    """Every NodeTag has a TAG_STYLES entry with badge_color/short_label."""
    style = TAG_STYLES[tag]
    assert {"badge_color", "short_label"} <= set(style.keys())


def test_resolve_display_returns_category_style():
    """resolve_display() returns the canonical style for the node category."""

    class Node:
        def get_category(self):
            return NodeCategory.MODEL

    display = resolve_display(Node())
    assert display == {
        "category": NodeCategory.MODEL,
        "fill": CATEGORY_STYLES[NodeCategory.MODEL]["fill"],
        "border": CATEGORY_STYLES[NodeCategory.MODEL]["border"],
        "emoji": CATEGORY_STYLES[NodeCategory.MODEL]["emoji"],
        "label": None,
    }


def test_resolve_display_unknown_category_uses_unspecified_style():
    """Unknown categories keep their value but use the UNSPECIFIED style."""

    class Node:
        def get_category(self):
            return "future-category"

    display = resolve_display(Node())
    assert display["category"] == "future-category"
    assert display["fill"] == CATEGORY_STYLES[NodeCategory.UNSPECIFIED]["fill"]
    assert display["border"] == CATEGORY_STYLES[NodeCategory.UNSPECIFIED]["border"]
    assert display["emoji"] == CATEGORY_STYLES[NodeCategory.UNSPECIFIED]["emoji"]
    assert display["label"] is None


def test_is_plugin_only_flags_registered_plugin_classes():
    """is_plugin() returns true only for classes listed in the plugin registry."""

    class PluginNode:
        pass

    class BuiltinNode:
        pass

    registry = SimpleNamespace(loaded_plugin_nodes={"PluginNode": object()})

    assert is_plugin(PluginNode(), registry)
    assert not is_plugin(BuiltinNode(), registry)
    assert not is_plugin(PluginNode())


def test_grpc_init_handles_missing_proto_helpers(monkeypatch):
    """grpc.__init__ keeps imports optional when generated helpers are missing."""
    original_import = builtins.__import__

    def import_without_conversions(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "cuvis_ai_schemas.grpc.conversions":
            raise ImportError("simulated missing generated proto helpers")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_conversions)
    grpc_init = Path(str(files("cuvis_ai_schemas"))) / "grpc" / "__init__.py"

    namespace = runpy.run_path(str(grpc_init))

    assert namespace["__all__"] == []


def test_node_info_round_trip():
    """NodeInfo serialises with new fields populated and survives a round-trip."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

    msg = cuvis_ai_pb2.NodeInfo(
        class_name="SAM3MaskTracker",
        full_path="cuvis_ai.node.sam3.SAM3MaskTracker",
        source="builtin",
        icon_svg=b"<svg/>",
        category=cuvis_ai_pb2.NODE_CATEGORY_MODEL,
        tags=[
            cuvis_ai_pb2.NODE_TAG_VIDEO,
            cuvis_ai_pb2.NODE_TAG_TRACKING,
            cuvis_ai_pb2.NODE_TAG_INFERENCE,
        ],
    )
    wire = msg.SerializeToString()
    parsed = cuvis_ai_pb2.NodeInfo.FromString(wire)
    assert parsed.class_name == "SAM3MaskTracker"
    assert parsed.icon_svg == b"<svg/>"
    assert parsed.category == cuvis_ai_pb2.NODE_CATEGORY_MODEL
    assert list(parsed.tags) == [
        cuvis_ai_pb2.NODE_TAG_VIDEO,
        cuvis_ai_pb2.NODE_TAG_TRACKING,
        cuvis_ai_pb2.NODE_TAG_INFERENCE,
    ]


def test_node_info_old_payload_uses_proto3_defaults():
    """A NodeInfo serialised with only fields 1–6 deserialises with the new
    fields at their proto3 defaults: empty bytes, UNSPECIFIED, empty list."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2

    old_msg = cuvis_ai_pb2.NodeInfo(
        class_name="LegacyNode",
        full_path="legacy.LegacyNode",
        source="builtin",
    )
    wire = old_msg.SerializeToString()
    parsed = cuvis_ai_pb2.NodeInfo.FromString(wire)
    assert parsed.icon_svg == b""
    assert parsed.category == cuvis_ai_pb2.NODE_CATEGORY_UNSPECIFIED
    assert list(parsed.tags) == []


def test_icons_bundle_exists():
    """The icons folder exists and is importable as a package."""
    icons_dir = files("cuvis_ai_schemas.extensions.ui.icons")
    svgs = sorted(p.name for p in icons_dir.iterdir() if p.name.endswith(".svg"))
    assert len(svgs) == 13


@pytest.mark.parametrize("category", list(NodeCategory))
def test_icon_exists_for_each_category(category):
    """Every NodeCategory value has a corresponding icons/{value}.svg."""
    icon_path = files("cuvis_ai_schemas.extensions.ui.icons") / f"{category.value}.svg"
    assert icon_path.is_file()


@pytest.mark.parametrize(
    "svg_name",
    [f"{c.value}.svg" for c in NodeCategory],
)
def test_icon_size_under_20kb(svg_name):
    """Each bundled SVG is <= 20 KB to keep ListAvailableNodes payloads bounded."""
    icon_path = files("cuvis_ai_schemas.extensions.ui.icons") / svg_name
    size = len(icon_path.read_bytes())
    assert size <= 20 * 1024, f"{svg_name} is {size} bytes (> 20 KB)"


def test_category_conversion_round_trip():
    """node_category_to_proto ↔ proto_to_node_category round-trips every value."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.conversions import (
        node_category_to_proto,
        proto_to_node_category,
    )

    for category in NodeCategory:
        assert proto_to_node_category(node_category_to_proto(category)) is category


def test_tag_conversion_round_trip():
    """node_tag_to_proto ↔ proto_to_node_tag round-trips every value."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.conversions import (
        node_tag_to_proto,
        proto_to_node_tag,
    )

    for tag in NodeTag:
        assert proto_to_node_tag(node_tag_to_proto(tag)) is tag


def test_proto_to_node_category_unknown_fallback():
    """Unknown wire ints map to NodeCategory.UNSPECIFIED for forward compat."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.conversions import proto_to_node_category

    assert proto_to_node_category(99999) is NodeCategory.UNSPECIFIED


def test_proto_to_node_tag_unknown_returns_none():
    """Unknown wire ints return None so consumers can skip them silently."""
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    from cuvis_ai_schemas.grpc.conversions import proto_to_node_tag

    assert proto_to_node_tag(99999) is None


# Worked composition examples from phase01.md §8 — each (category, tags) combo
# must parse cleanly against NodeCategory / NodeTag.
COMPOSITION_EXAMPLES = [
    (
        "SAM3MaskTracker",
        "MODEL",
        [
            "VIDEO",
            "RGB",
            "MASK",
            "TRACKING",
            "SEGMENTATION",
            "INFERENCE",
            "LEARNABLE",
            "BATCHED",
            "TORCH",
        ],
    ),
    (
        "WhiteReferenceCalibration",
        "TRANSFORM",
        ["HYPERSPECTRAL", "CALIBRATION", "PREPROCESSING", "INVERTIBLE", "NUMPY"],
    ),
    ("HSICubeLoader", "SOURCE", ["HYPERSPECTRAL", "METADATA", "STREAMING"]),
    (
        "ForegroundContrastLoss",
        "LOSS",
        ["HYPERSPECTRAL", "RGB", "SEGMENTATION", "TRAINING", "DIFFERENTIABLE", "TORCH"],
    ),
    (
        "PCAReducer",
        "MODEL",
        ["HYPERSPECTRAL", "DIM_REDUCTION", "PREPROCESSING", "LEARNABLE", "INVERTIBLE", "NUMPY"],
    ),
    (
        "RandomSpectralJitter",
        "TRANSFORM",
        ["HYPERSPECTRAL", "AUGMENTATION", "TRAINING", "STOCHASTIC", "TORCH"],
    ),
    ("CosineLRSchedule", "SCHEDULER", ["TRAINING"]),
    ("mIoUMetric", "METRIC", ["SEGMENTATION", "EVALUATION", "MASK"]),
    ("SpectralBandViewer", "VISUALIZER", ["HYPERSPECTRAL", "RGB"]),
    ("TrainLoop", "RUNNER", ["TRAINING"]),
]


@pytest.mark.parametrize("name,category,tags", COMPOSITION_EXAMPLES)
def test_composition_examples_parse(name, category, tags):
    """Every worked composition example resolves to valid enum members."""
    assert NodeCategory[category] is not None, f"{name}: bad category {category}"
    for tag in tags:
        assert NodeTag[tag] is not None, f"{name}: bad tag {tag}"
