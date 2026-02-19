"""Tests for pipeline schemas."""

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.pipeline import (
    ConnectionConfig,
    NodeConfig,
    PipelineConfig,
    PipelineMetadata,
    PortSpec,
)


def test_port_spec_creation():
    """Test PortSpec creation."""
    port = PortSpec(dtype="torch.Tensor", shape=(1, 3, 224, 224), description="Input image")
    assert port.dtype == "torch.Tensor"
    assert port.shape == (1, 3, 224, 224)
    assert port.description == "Input image"
    assert port.optional is False


def test_pipeline_metadata():
    """Test PipelineMetadata."""
    metadata = PipelineMetadata(
        name="test_pipeline",
        description="A test pipeline",
        author="Test Author",
        tags=["test", "example"],
    )
    assert metadata.name == "test_pipeline"
    assert "test" in metadata.tags

    # Test dict conversion
    data = metadata.to_dict()
    loaded = PipelineMetadata.from_dict(data)
    assert loaded.name == metadata.name


def test_node_config():
    """Test NodeConfig with direct field names (no aliases)."""
    node = NodeConfig(name="normalizer", class_name="module.Normalizer", hparams={"min": 0})
    assert node.name == "normalizer"
    assert node.class_name == "module.Normalizer"
    assert node.hparams == {"min": 0}


def test_node_config_default_hparams():
    """Test NodeConfig with default empty hparams."""
    node = NodeConfig(name="node_1", class_name="my.package.MyNode")
    assert node.hparams == {}


def test_node_config_rejects_old_aliases():
    """Test that old alias keys are rejected (extra=forbid)."""
    with pytest.raises(ValidationError):
        NodeConfig(name="node_1", **{"class": "my.package.MyNode"})

    with pytest.raises(ValidationError):
        NodeConfig(name="node_1", class_name="my.package.MyNode", **{"params": {"lr": 0.001}})


def test_connection_config():
    """Test ConnectionConfig with source/target format."""
    conn = ConnectionConfig(
        source="normalizer.outputs.cube",
        target="model.inputs.data",
    )
    assert conn.source == "normalizer.outputs.cube"
    assert conn.target == "model.inputs.data"

    # Test convenience properties
    assert conn.from_node == "normalizer"
    assert conn.from_port == "cube"
    assert conn.to_node == "model"
    assert conn.to_port == "data"


def test_connection_config_validation():
    """Test ConnectionConfig source/target validation."""
    with pytest.raises(ValidationError, match="Invalid source"):
        ConnectionConfig(source="bad_format", target="model.inputs.data")

    with pytest.raises(ValidationError, match="Invalid source"):
        ConnectionConfig(source="node.inputs.port", target="model.inputs.data")

    with pytest.raises(ValidationError, match="Invalid target"):
        ConnectionConfig(source="node.outputs.port", target="bad_format")

    with pytest.raises(ValidationError, match="Invalid target"):
        ConnectionConfig(source="node.outputs.port", target="model.outputs.data")


def test_pipeline_config():
    """Test PipelineConfig with typed nodes and connections."""
    pipeline = PipelineConfig(
        name="test_pipeline",
        nodes=[
            NodeConfig(name="normalizer", class_name="module.Normalizer", hparams={"min": 0}),
        ],
        connections=[],
    )
    assert pipeline.name == "test_pipeline"
    assert len(pipeline.nodes) == 1
    assert isinstance(pipeline.nodes[0], NodeConfig)
    assert pipeline.nodes[0].name == "normalizer"


def test_pipeline_config_from_dict():
    """Test PipelineConfig constructed from dict (simulates YAML loading)."""
    data = {
        "name": "test_pipeline",
        "nodes": [
            {"name": "normalizer", "class_name": "module.Normalizer", "hparams": {"min": 0}},
            {"name": "model", "class_name": "module.Model"},
        ],
        "connections": [
            {"source": "normalizer.outputs.cube", "target": "model.inputs.data"},
        ],
    }
    pipeline = PipelineConfig.from_dict(data)
    assert isinstance(pipeline.nodes[0], NodeConfig)
    assert isinstance(pipeline.connections[0], ConnectionConfig)
    assert pipeline.nodes[0].name == "normalizer"
    assert pipeline.connections[0].from_node == "normalizer"
    assert pipeline.connections[0].to_node == "model"


def test_pipeline_config_round_trip():
    """Test dict -> PipelineConfig -> to_dict() -> reload round trip."""
    original = {
        "name": "round_trip_test",
        "nodes": [
            {"name": "node_a", "class_name": "pkg.NodeA", "hparams": {"lr": 0.001}},
            {"name": "node_b", "class_name": "pkg.NodeB"},
        ],
        "connections": [
            {"source": "node_a.outputs.result", "target": "node_b.inputs.data"},
        ],
    }
    pipeline = PipelineConfig.from_dict(original)
    dumped = pipeline.to_dict()
    reloaded = PipelineConfig.from_dict(dumped)

    assert reloaded.name == original["name"]
    assert len(reloaded.nodes) == 2
    assert len(reloaded.connections) == 1
    assert reloaded.nodes[0].name == "node_a"
    assert reloaded.connections[0].source == "node_a.outputs.result"


def test_pipeline_config_json_round_trip():
    """Test JSON serialization round trip."""
    pipeline = PipelineConfig(
        name="json_test",
        nodes=[NodeConfig(name="n1", class_name="pkg.N1")],
        connections=[
            ConnectionConfig(source="n1.outputs.out", target="n2.inputs.in"),
        ],
    )
    json_str = pipeline.to_json()
    loaded = PipelineConfig.from_json(json_str)
    assert loaded.name == pipeline.name
    assert loaded.nodes[0].class_name == "pkg.N1"
    assert loaded.connections[0].source == "n1.outputs.out"


def test_model_dump_uses_field_names():
    """Verify model_dump() uses field names directly (no alias indirection)."""
    node = NodeConfig(name="test", class_name="pkg.Test", hparams={"x": 1})
    dumped = node.model_dump()
    assert "name" in dumped
    assert "class_name" in dumped
    assert "hparams" in dumped
    # Old alias keys must not appear
    assert "id" not in dumped
    assert "class" not in dumped
    assert "params" not in dumped

    conn = ConnectionConfig(source="a.outputs.b", target="c.inputs.d")
    dumped = conn.model_dump()
    assert "source" in dumped
    assert "target" in dumped
    assert "from_node" not in dumped
    assert "to_node" not in dumped
