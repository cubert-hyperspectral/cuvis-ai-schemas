"""gRPC proto messages and helpers."""

__all__: list[str] = []

try:
    from cuvis_ai_schemas.grpc.conversions import (
        node_category_to_proto,
        node_tag_to_proto,
        proto_to_node_category,
        proto_to_node_tag,
    )

    __all__ = [
        "node_category_to_proto",
        "proto_to_node_category",
        "node_tag_to_proto",
        "proto_to_node_tag",
    ]
except ImportError:
    # Proto files not generated yet or proto extra not installed
    pass
