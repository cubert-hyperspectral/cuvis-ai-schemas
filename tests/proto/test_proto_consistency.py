from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

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


def test_profiling_rpcs_on_both_services() -> None:
    """Profiling state lives on the child's pipeline, so both services carry the RPCs."""
    services = cuvis_ai_pb2.DESCRIPTOR.services_by_name

    for service_name in ("CuvisAIService", "RunRuntime"):
        methods = services[service_name].methods_by_name
        for rpc, request, response in (
            ("SetProfiling", "SetProfilingRequest", "SetProfilingResponse"),
            ("GetProfilingSummary", "GetProfilingSummaryRequest", "GetProfilingSummaryResponse"),
        ):
            assert rpc in methods
            assert methods[rpc].input_type.name == request
            assert methods[rpc].output_type.name == response


def test_node_profiling_stats_flops_per_call_has_presence() -> None:
    """flops_per_call is optional field 11: a counted 0 is set, a missing count is unset."""
    field = cuvis_ai_pb2.NodeProfilingStats.DESCRIPTOR.fields_by_name["flops_per_call"]

    assert field.number == 11
    assert field.has_presence
    assert not cuvis_ai_pb2.NodeProfilingStats(node_name="N").HasField("flops_per_call")
    assert cuvis_ai_pb2.NodeProfilingStats(node_name="N", flops_per_call=0).HasField(
        "flops_per_call"
    )


def _node_profiling_stats_without_flops() -> type[Message]:
    """NodeProfilingStats as a peer built before field 11 sees it (enum stage as int32)."""
    message = descriptor_pb2.DescriptorProto()
    cuvis_ai_pb2.NodeProfilingStats.DESCRIPTOR.CopyToProto(message)
    kept = [f for f in message.field if f.name != "flops_per_call"]
    del message.field[:]
    message.field.extend(kept)
    del message.oneof_decl[:]
    for field in message.field:
        field.ClearField("oneof_index")
        if field.type == descriptor_pb2.FieldDescriptorProto.TYPE_ENUM:
            field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT32
            field.ClearField("type_name")
    old_file = descriptor_pb2.FileDescriptorProto(
        name="old_profiling.proto", package="old", syntax="proto3", message_type=[message]
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(old_file)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("old.NodeProfilingStats"))


@pytest.mark.parametrize("flops", [0, 123_456_789_012])
def test_flops_per_call_survives_a_peer_without_field_11(flops: int) -> None:
    """An older orchestrator parent re-serializes the child's reply with field 11 intact."""
    new = cuvis_ai_pb2.NodeProfilingStats(node_name="N", count=2, flops_per_call=flops)

    old = _node_profiling_stats_without_flops().FromString(new.SerializeToString())
    back = cuvis_ai_pb2.NodeProfilingStats.FromString(old.SerializeToString())

    assert back.HasField("flops_per_call")
    assert back.flops_per_call == flops
    assert back.count == 2


def test_restore_train_run_accepts_target_session() -> None:
    """RestoreTrainRun can restore into a client-prepared session (plugin catalog)."""
    fields = cuvis_ai_pb2.RestoreTrainRunRequest.DESCRIPTOR.fields_by_name

    assert "session_id" in fields
    assert fields["session_id"].number == 4


def test_clear_plugin_cache_messages_kept_without_their_rpc() -> None:
    """cuvis-ai-core 0.17.x names both messages at import, so they outlive the removed RPC."""
    request_fields = cuvis_ai_pb2.ClearPluginCacheRequest.DESCRIPTOR.fields_by_name
    response_fields = cuvis_ai_pb2.ClearPluginCacheResponse.DESCRIPTOR.fields_by_name

    assert set(request_fields) == {"plugin_name"}
    assert request_fields["plugin_name"].number == 1
    assert set(response_fields) == {"cleared_count"}
    assert response_fields["cleared_count"].number == 1
    for service in cuvis_ai_pb2.DESCRIPTOR.services_by_name.values():
        assert "ClearPluginCache" not in service.methods_by_name


def test_trainer_type_has_calibration_value() -> None:
    """TrainerType carries the standalone post-fit phase (calibrate and evaluate, no fit)."""
    values = cuvis_ai_pb2.TrainerType.DESCRIPTOR.values_by_name

    assert "TRAINER_TYPE_CALIBRATION" in values
    assert values["TRAINER_TYPE_CALIBRATION"].number == 3


def test_train_response_carries_a_typed_event() -> None:
    """Every post-fit response reports its lifecycle hook as data, never as message text."""
    fields = cuvis_ai_pb2.TrainResponse.DESCRIPTOR.fields_by_name

    assert "event" in fields
    assert fields["event"].number == 6
    assert fields["event"].message_type.name == "TrainEvent"

    event = cuvis_ai_pb2.TrainEvent.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in event.items()} == {
        "hook": 1,
        "split": 2,
        "progress": 3,
        "checkpoint": 4,
        "calibration": 5,
        "evaluation": 6,
        "calibration_pending": 7,
        "epochs": 8,
        "node": 9,
    }
    assert event["hook"].enum_type.name == "TrainHook"
    assert event["split"].enum_type.name == "ExecutionStage"
    assert event["progress"].message_type.name == "FrameProgress"
    assert event["checkpoint"].message_type.name == "CheckpointRestored"
    assert event["calibration"].message_type.name == "CalibrationOutcome"
    assert event["evaluation"].message_type.name == "EvaluationOutcome"
    assert event["epochs"].message_type.name == "EpochProgress"
    assert event["node"].type == event["node"].TYPE_STRING


def test_epoch_progress_shape() -> None:
    """The epoch counter travels typed: current is one-based, total 0 means unknown."""
    fields = cuvis_ai_pb2.EpochProgress.DESCRIPTOR.fields_by_name

    assert {name: field.number for name, field in fields.items()} == {"current": 1, "total": 2}
    assert fields["current"].type == fields["current"].TYPE_INT32
    assert fields["total"].type == fields["total"].TYPE_INT32


def test_train_hook_values_are_pinned() -> None:
    """A client switches on the hook number, so the numbers are part of the contract.

    The values 13 to 19 report the fit itself (the statistical pass per node, the epoch start,
    the train and validation batches), so every RUNNING progress response carries an event.
    """
    values = cuvis_ai_pb2.TrainHook.DESCRIPTOR.values_by_name

    assert {name: value.number for name, value in values.items()} == {
        "TRAIN_HOOK_UNSPECIFIED": 0,
        "TRAIN_HOOK_FIT_START": 1,
        "TRAIN_HOOK_EPOCH_END": 2,
        "TRAIN_HOOK_VALIDATION_END": 3,
        "TRAIN_HOOK_FIT_END": 4,
        "TRAIN_HOOK_CHECKPOINT_RESTORED": 5,
        "TRAIN_HOOK_CALIBRATION_START": 6,
        "TRAIN_HOOK_CALIBRATION_PROGRESS": 7,
        "TRAIN_HOOK_CALIBRATION_END": 8,
        "TRAIN_HOOK_EVALUATION_START": 9,
        "TRAIN_HOOK_EVALUATION_PROGRESS": 10,
        "TRAIN_HOOK_EVALUATION_END": 11,
        "TRAIN_HOOK_POST_FIT_END": 12,
        "TRAIN_HOOK_STATISTICAL_NODE_START": 13,
        "TRAIN_HOOK_STATISTICAL_PROGRESS": 14,
        "TRAIN_HOOK_STATISTICAL_NODE_END": 15,
        "TRAIN_HOOK_EPOCH_START": 16,
        "TRAIN_HOOK_TRAIN_PROGRESS": 17,
        "TRAIN_HOOK_VALIDATION_START": 18,
        "TRAIN_HOOK_VALIDATION_PROGRESS": 19,
    }


def test_calibration_outcome_is_structured() -> None:
    """The calibration result travels as enums, counts and per-decider records, not a sentence."""
    outcome = cuvis_ai_pb2.CalibrationOutcome.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in outcome.items()} == {
        "split": 1,
        "applicable": 2,
        "phase_skip": 3,
        "phase_detail": 4,
        "frames": 5,
        "anomalous_frames": 6,
        "deciders": 7,
    }
    assert outcome["split"].enum_type.name == "ExecutionStage"
    assert outcome["phase_skip"].enum_type.name == "CalibrationPhaseSkip"
    assert outcome["deciders"].message_type.name == "DeciderCalibration"
    assert outcome["deciders"].label == outcome["deciders"].LABEL_REPEATED

    decider = cuvis_ai_pb2.DeciderCalibration.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in decider.items()} == {
        "name": 1,
        "class_name": 2,
        "status": 3,
        "skip": 4,
        "skip_detail": 5,
        "thresholds": 6,
        "metrics": 7,
        "thresholds_present": 8,
    }
    assert decider["status"].enum_type.name == "DeciderCalibrationStatus"
    assert decider["skip"].enum_type.name == "DeciderSkip"
    threshold_entry = decider["thresholds"].message_type.fields_by_name
    assert threshold_entry["value"].message_type.name == "ThresholdChange"

    change = cuvis_ai_pb2.ThresholdChange.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in change.items()} == {"before": 1, "after": 2}
    assert change["before"].has_presence
    assert not change["after"].has_presence

    phase = cuvis_ai_pb2.CalibrationPhaseSkip.DESCRIPTOR.values_by_name
    assert {name: value.number for name, value in phase.items()} == {
        "CALIBRATION_PHASE_SKIP_UNSPECIFIED": 0,
        "CALIBRATION_PHASE_SKIP_NONE": 1,
        "CALIBRATION_PHASE_SKIP_NO_SPLIT": 2,
        "CALIBRATION_PHASE_SKIP_INFERENCE_FAILED": 3,
        "CALIBRATION_PHASE_SKIP_AMBIGUOUS_MASK": 4,
        "CALIBRATION_PHASE_SKIP_SINGLE_CLASS": 5,
    }
    status = cuvis_ai_pb2.DeciderCalibrationStatus.DESCRIPTOR.values_by_name
    assert {name: value.number for name, value in status.items()} == {
        "DECIDER_CALIBRATION_STATUS_UNSPECIFIED": 0,
        "DECIDER_CALIBRATION_STATUS_CALIBRATED": 1,
        "DECIDER_CALIBRATION_STATUS_SKIPPED": 2,
    }
    skip = cuvis_ai_pb2.DeciderSkip.DESCRIPTOR.values_by_name
    assert {name: value.number for name, value in skip.items()} == {
        "DECIDER_SKIP_UNSPECIFIED": 0,
        "DECIDER_SKIP_NONE": 1,
        "DECIDER_SKIP_NOTHING_WIRED": 2,
        "DECIDER_SKIP_NO_SCORES": 3,
        "DECIDER_SKIP_CALIBRATE_FAILED": 4,
        "DECIDER_SKIP_NO_REPORT": 5,
    }


def test_checkpoint_progress_and_evaluation_outcomes_shape() -> None:
    """The restore, the frame progress and each evaluation pass are typed as well."""
    checkpoint = cuvis_ai_pb2.CheckpointRestored.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in checkpoint.items()} == {
        "restored": 1,
        "path": 2,
        "epoch": 3,
        "monitor": 4,
        "score": 5,
        "reason": 6,
    }
    assert checkpoint["score"].has_presence

    progress = cuvis_ai_pb2.FrameProgress.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in progress.items()} == {"done": 1, "total": 2}

    evaluation = cuvis_ai_pb2.EvaluationOutcome.DESCRIPTOR.fields_by_name
    assert {name: field.number for name, field in evaluation.items()} == {
        "split": 1,
        "status": 2,
        "frames": 3,
        "metrics": 4,
        "error": 5,
    }
    assert evaluation["split"].enum_type.name == "ExecutionStage"
    assert evaluation["status"].enum_type.name == "EvaluationStatus"
    values = cuvis_ai_pb2.EvaluationStatus.DESCRIPTOR.values_by_name
    assert {name: value.number for name, value in values.items()} == {
        "EVALUATION_STATUS_UNSPECIFIED": 0,
        "EVALUATION_STATUS_EVALUATED": 1,
        "EVALUATION_STATUS_NO_SPLIT": 2,
        "EVALUATION_STATUS_FAILED": 3,
    }
