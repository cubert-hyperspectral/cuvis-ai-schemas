from pathlib import Path

from cuvis_ai_schemas.grpc.v1 import cuvis_ai_pb2


def test_no_legacy_validate_training_config_rpc() -> None:
    """Ensure old ValidateTrainingConfig RPC is not reintroduced."""
    proto_path = Path("proto/cuvis_ai_schemas/grpc/v1/cuvis_ai.proto")
    proto_content = proto_path.read_text()

    assert "ValidateTrainingConfigRequest" not in proto_content
    assert "ValidateTrainingConfigResponse" not in proto_content
    assert "rpc ValidateTrainingConfig" not in proto_content


def test_input_batch_has_explicit_video_fields() -> None:
    """InputBatch should expose first-class RGB video inputs."""
    fields = cuvis_ai_pb2.InputBatch.DESCRIPTOR.fields_by_name

    assert "rgb_image" in fields
    assert "frame_id" in fields
    assert fields["rgb_image"].number == 9
    assert fields["frame_id"].number == 10
