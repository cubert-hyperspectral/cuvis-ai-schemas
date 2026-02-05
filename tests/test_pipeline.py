"""Tests for pipeline schemas."""

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
    """Test NodeConfig with field aliases."""
    # Test with class_name field
    node1 = NodeConfig(id="node_1", class_name="my.package.MyNode", params={"lr": 0.001})
    assert node1.id == "node_1"
    assert node1.class_name == "my.package.MyNode"

    # Test with 'class' alias
    node2 = NodeConfig(
        id="node_2", **{"class": "my.package.OtherNode", "hparams": {"batch_size": 32}}
    )
    assert node2.class_name == "my.package.OtherNode"
    assert node2.params == {"batch_size": 32}


def test_connection_config():
    """Test ConnectionConfig."""
    conn = ConnectionConfig(
        from_node="node_1",
        from_port="output",
        to_node="node_2",
        to_port="input",
    )
    assert conn.from_node == "node_1"
    assert conn.to_port == "input"


def test_pipeline_config():
    """Test PipelineConfig with nodes and connections."""
    pipeline = PipelineConfig(
        name="test_pipeline",
        nodes=[{"id": "node_1", "class": "MyNode", "params": {}}],
        connections=[],
        frozen_nodes=["node_1"],
    )
    assert pipeline.name == "test_pipeline"
    assert len(pipeline.nodes) == 1
    assert "node_1" in pipeline.frozen_nodes

    # Test JSON serialization
    json_str = pipeline.to_json()
    loaded = PipelineConfig.from_json(json_str)
    assert loaded.name == pipeline.name
