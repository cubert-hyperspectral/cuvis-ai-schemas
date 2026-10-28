"""Round trip of a capability's hparam spec through the NodeInfo proto message."""

from __future__ import annotations

import pytest

from cuvis_ai_schemas.grpc.conversions import hparam_spec_to_proto, proto_to_hparam_spec
from cuvis_ai_schemas.plugin import HparamSpec


@pytest.mark.parametrize(
    "spec",
    [
        HparamSpec(type="int", default=5, min=1, max=31, description="Pooling window"),
        HparamSpec(type="int", required=True, min=1),
        HparamSpec(type="float", default=0.5),
        HparamSpec(type="bool", default=False),
        HparamSpec(type="str", nullable=True),
        HparamSpec(type="enum", choices=["abs", "sq"], default="sq"),
        HparamSpec(type="int", list=True, default=[1, 2]),
    ],
)
def test_round_trip(spec: HparamSpec) -> None:
    message = hparam_spec_to_proto("p", spec)
    assert message.name == "p"
    assert proto_to_hparam_spec(message) == spec


def test_unset_bounds_have_no_presence() -> None:
    message = hparam_spec_to_proto("p", HparamSpec(type="int", default=1, min=0))
    assert message.HasField("min")
    assert not message.HasField("max")


def test_default_keeps_its_json_type() -> None:
    message = hparam_spec_to_proto("p", HparamSpec(type="float", default=1))
    assert message.default_json == "1"
    assert hparam_spec_to_proto("p", HparamSpec(type="str", nullable=True)).default_json == ""
