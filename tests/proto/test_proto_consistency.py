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


def test_bounding_box_object_id_transport_is_reserved() -> None:
    """BoundingBox transport should preserve a dedicated optional object_id field."""
    bbox_fields = cuvis_ai_pb2.BoundingBox.DESCRIPTOR.fields_by_name
    input_fields = cuvis_ai_pb2.InputBatch.DESCRIPTOR.fields_by_name

    assert "element_id" in bbox_fields
    assert bbox_fields["element_id"].number == 1
    assert "object_id" in bbox_fields
    assert bbox_fields["object_id"].number == 6
    assert "bboxes" in input_fields
    assert input_fields["bboxes"].number == 4


def test_pipeline_discovery_uses_explicit_lookup_and_resolved_paths() -> None:
    """Pipeline discovery should distinguish stable lookup keys from resolved paths."""
    info_fields = cuvis_ai_pb2.PipelineInfo.DESCRIPTOR.fields_by_name
    request_fields = cuvis_ai_pb2.GetPipelineInfoRequest.DESCRIPTOR.fields_by_name

    assert "pipeline_path" in info_fields
    assert "resolved_path" in info_fields
    assert "path" not in info_fields
    assert info_fields["pipeline_path"].number == 1
    assert info_fields["resolved_path"].number == 2
    assert "pipeline_path" in request_fields
    assert request_fields["pipeline_path"].number == 1
