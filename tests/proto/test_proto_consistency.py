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


def test_train_status_has_terminal_cancelled_value() -> None:
    """TrainStatus must carry the terminal CANCELLED value for cooperative stop."""
    values = cuvis_ai_pb2.TrainStatus.DESCRIPTOR.values_by_name

    assert "TRAIN_STATUS_CANCELLED" in values
    assert values["TRAIN_STATUS_CANCELLED"].number == 4


def test_stop_train_messages_shape() -> None:
    """StopTrain carries a session id in and an accepted flag + message out."""
    request_fields = cuvis_ai_pb2.StopTrainRequest.DESCRIPTOR.fields_by_name
    response_fields = cuvis_ai_pb2.StopTrainResponse.DESCRIPTOR.fields_by_name

    assert set(request_fields) == {"session_id"}
    assert request_fields["session_id"].number == 1
    assert set(response_fields) == {"accepted", "message"}
    assert response_fields["accepted"].number == 1
    assert response_fields["message"].number == 2


def test_stop_train_rpc_on_both_services() -> None:
    """StopTrain must exist on the public service and the child runtime alike."""
    services = cuvis_ai_pb2.DESCRIPTOR.services_by_name

    for service_name in ("CuvisAIService", "RunRuntime"):
        methods = services[service_name].methods_by_name
        assert "StopTrain" in methods
        assert methods["StopTrain"].input_type.name == "StopTrainRequest"
        assert methods["StopTrain"].output_type.name == "StopTrainResponse"
