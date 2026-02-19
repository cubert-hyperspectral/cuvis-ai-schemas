"""Tests for BaseSchemaModel serialization including proto support."""

from unittest.mock import patch

import pytest

from cuvis_ai_schemas.base import BaseSchemaModel, _get_pb2
from cuvis_ai_schemas.pipeline import PipelineConfig, PipelineMetadata
from cuvis_ai_schemas.training import DataConfig, OptimizerConfig


def test_get_pb2_returns_module():
    """Test _get_pb2 returns the proto module."""
    pb2 = _get_pb2()
    assert hasattr(pb2, "DataConfig")


def test_get_pb2_import_error():
    """Test _get_pb2 raises ImportError when proto not installed."""
    with patch.dict(
        "sys.modules", {"cuvis_ai_schemas.grpc": None, "cuvis_ai_schemas.grpc.v1": None}
    ):
        with pytest.raises(ImportError, match="Proto support not installed"):
            _get_pb2()


def test_to_proto_round_trip():
    """Test to_proto -> from_proto round-trip for DataConfig."""
    original = DataConfig(cu3s_file_path="/data/test.cu3s", batch_size=16)
    proto = original.to_proto()
    restored = DataConfig.from_proto(proto)
    assert restored.cu3s_file_path == "/data/test.cu3s"
    assert restored.batch_size == 16


def test_to_proto_not_implemented():
    """Test to_proto raises NotImplementedError when __proto_message__ is empty."""

    class NoProtoModel(BaseSchemaModel):
        value: int = 0

    with pytest.raises(NotImplementedError, match="has no proto mapping"):
        NoProtoModel(value=1).to_proto()


def test_to_proto_invalid_message_name():
    """Test to_proto raises ValueError for misspelled proto message."""
    from typing import ClassVar

    class BadProtoModel(BaseSchemaModel):
        __proto_message__: ClassVar[str] = "NonExistentProtoMessage"
        value: int = 0

    with pytest.raises(ValueError, match="not found in cuvis_ai_pb2"):
        BadProtoModel(value=1).to_proto()


def test_optimizer_proto_round_trip():
    """Test OptimizerConfig proto round-trip."""
    original = OptimizerConfig(name="sgd", lr=0.01, weight_decay=0.001)
    proto = original.to_proto()
    restored = OptimizerConfig.from_proto(proto)
    assert restored.name == "sgd"
    assert restored.lr == 0.01


def test_pipeline_config_save_and_load(tmp_path):
    """Test PipelineConfig save_to_file and load_from_file."""
    from cuvis_ai_schemas.pipeline import ConnectionConfig, NodeConfig

    config = PipelineConfig(
        name="test",
        nodes=[NodeConfig(name="n1", class_name="pkg.N1")],
        connections=[ConnectionConfig(source="n1.outputs.out", target="n2.inputs.in")],
    )
    path = tmp_path / "pipeline.yaml"
    config.save_to_file(path)
    loaded = PipelineConfig.load_from_file(path)
    assert loaded.name == "test"
    assert loaded.nodes[0].class_name == "pkg.N1"


def test_pipeline_metadata_to_proto_import_error():
    """Test PipelineMetadata.to_proto raises ImportError when proto missing."""
    meta = PipelineMetadata(name="test")
    with patch.dict(
        "sys.modules", {"cuvis_ai_schemas.grpc": None, "cuvis_ai_schemas.grpc.v1": None}
    ):
        with pytest.raises(ImportError, match="Proto support not installed"):
            meta.to_proto()


def test_pipeline_metadata_to_proto():
    """Test PipelineMetadata.to_proto field-by-field mapping."""
    meta = PipelineMetadata(
        name="test_pipe",
        description="A test",
        author="tester",
        tags=["a", "b"],
    )
    proto = meta.to_proto()
    assert proto.name == "test_pipe"
    assert proto.description == "A test"
    assert proto.author == "tester"
    assert list(proto.tags) == ["a", "b"]
