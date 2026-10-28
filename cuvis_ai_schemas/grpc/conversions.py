"""Conversion helpers between Python ``NodeCategory`` / ``NodeTag`` enums and
their proto enum integer wire values, and between a manifest's ``HparamSpec``
and the ``NodeInfo.hparams`` proto message.

Lives in cuvis-ai-schemas because both producers (the gRPC populator in
cuvis-ai-core) and consumers (the Qt UI client in cuvis-ai-ui) need the
helpers, and the UI already depends on schemas. Importing this module
requires the ``[proto]`` extra (it imports ``cuvis_ai_pb2``).
"""

import json

from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2
from cuvis_ai_schemas.plugin import HparamSpec

# A member without a proto constant (stubs not regenerated yet) is left out of the table and
# maps to UNSPECIFIED, like the hand-written tables did; the CI contract test reports the skew.
_CATEGORY_PY_TO_PROTO: dict[NodeCategory, int] = {
    c: getattr(cuvis_ai_pb2, f"NODE_CATEGORY_{c.name}")
    for c in NodeCategory
    if hasattr(cuvis_ai_pb2, f"NODE_CATEGORY_{c.name}")
}
_CATEGORY_PROTO_TO_PY: dict[int, NodeCategory] = {v: k for k, v in _CATEGORY_PY_TO_PROTO.items()}


_TAG_PY_TO_PROTO: dict[NodeTag, int] = {
    t: getattr(cuvis_ai_pb2, f"NODE_TAG_{t.name}")
    for t in NodeTag
    if hasattr(cuvis_ai_pb2, f"NODE_TAG_{t.name}")
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


def hparam_spec_to_proto(name: str, spec: HparamSpec) -> cuvis_ai_pb2.HparamSpec:
    """Build the ``NodeInfo.hparams`` entry for one declared parameter.

    ``default`` and each choice travel as JSON so their type survives (a float
    default of ``1`` stays ``1``, a string choice stays a string).
    """
    bounds = {
        key: value for key, value in (("min", spec.min), ("max", spec.max)) if value is not None
    }
    return cuvis_ai_pb2.HparamSpec(
        name=name,
        type=spec.type,
        list=spec.is_list,
        default_json="" if spec.default is None else json.dumps(spec.default),
        required=spec.required,
        nullable=spec.nullable,
        choices_json=[json.dumps(choice) for choice in spec.choices],
        description=spec.description,
        **bounds,
    )


def proto_to_hparam_spec(message: cuvis_ai_pb2.HparamSpec) -> HparamSpec:
    """Read a ``NodeInfo.hparams`` entry back into an :class:`HparamSpec`."""
    return HparamSpec.model_validate(
        {
            "type": message.type,
            "list": message.list,
            "default": json.loads(message.default_json) if message.default_json else None,
            "required": message.required,
            "nullable": message.nullable,
            "min": message.min if message.HasField("min") else None,
            "max": message.max if message.HasField("max") else None,
            "choices": [json.loads(choice) for choice in message.choices_json],
            "description": message.description,
        }
    )
