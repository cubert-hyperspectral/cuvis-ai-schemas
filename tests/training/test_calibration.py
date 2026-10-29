"""Tests for the calibration block of a training run."""

import json

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training import (
    CalibrationConfig,
    DataConfig,
    TrainingConfig,
    TrainRunConfig,
)


def _run(**kwargs) -> TrainRunConfig:
    """A minimal trainrun; keyword arguments set further fields."""
    return TrainRunConfig(name="run", data=DataConfig(data_module="cu3s"), **kwargs)


def test_defaults_calibrate_to_f1():
    block = CalibrationConfig()
    assert (block.objective, block.target, block.level) == ("f1", None, "pixels")


@pytest.mark.parametrize("objective", ["precision", "recall"])
@pytest.mark.parametrize("level", ["pixels", "frames"])
def test_precision_and_recall_take_a_target_at_either_level(objective, level):
    block = CalibrationConfig(objective=objective, target=0.9, level=level)
    assert (block.objective, block.target, block.level) == (objective, 0.9, level)


@pytest.mark.parametrize("objective", ["precision", "recall"])
def test_precision_and_recall_without_a_target_are_refused(objective):
    with pytest.raises(ValidationError, match="needs a target"):
        CalibrationConfig(objective=objective)


def test_f1_with_a_target_is_refused():
    with pytest.raises(ValidationError, match="takes no target"):
        CalibrationConfig(objective="f1", target=0.9)


def test_f1_at_the_frame_level_is_refused():
    """F1 keeps its own rule, so a level other than the default would be ignored."""
    with pytest.raises(ValidationError, match="level"):
        CalibrationConfig(objective="f1", level="frames")


@pytest.mark.parametrize("target", [0.0, 1.0, 1])
def test_the_bounds_and_an_integer_are_valid_targets(target):
    assert CalibrationConfig(objective="precision", target=target).target == float(target)


@pytest.mark.parametrize("target", [-0.01, 1.01, float("nan"), float("inf"), True, "0.9x"])
def test_a_target_outside_zero_and_one_or_not_a_number_is_refused(target):
    with pytest.raises(ValidationError):
        CalibrationConfig(objective="recall", target=target)


@pytest.mark.parametrize(("field", "value"), [("objective", "accuracy"), ("level", "objects")])
def test_an_unknown_objective_or_level_is_refused(field, value):
    with pytest.raises(ValidationError):
        CalibrationConfig(**{field: value})


def test_unknown_keys_are_refused():
    with pytest.raises(ValidationError):
        CalibrationConfig.from_dict({"objective": "f1", "metric": "f1"})


def test_a_strategy_is_registered_for_the_round_trip_properties():
    from cuvis_ai_schemas.testing import MODEL_STRATEGIES

    assert CalibrationConfig in MODEL_STRATEGIES


def test_a_trainrun_without_the_block_serializes_as_before():
    """No ``calibration`` key anywhere, so a core that predates the block reads the run."""
    run = _run()
    assert run.calibration is None
    assert "calibration" not in run.to_dict()
    assert "calibration" not in json.loads(run.to_json())


def test_a_trainrun_without_the_block_writes_no_calibration_key(tmp_path):
    path = tmp_path / "trainrun.yaml"
    _run().save_to_file(path)
    assert "calibration" not in path.read_text(encoding="utf-8")


def test_a_trainrun_carries_the_block_through_dict_and_json():
    run = _run(calibration=CalibrationConfig(objective="recall", target=0.95, level="frames"))
    assert run.to_dict()["calibration"] == {
        "objective": "recall",
        "target": 0.95,
        "level": "frames",
    }
    assert TrainRunConfig.from_dict(run.to_dict()) == run
    assert TrainRunConfig.from_json(run.to_json()) == run


def test_a_trainrun_carries_the_block_through_a_yaml_file(tmp_path):
    run = _run(
        calibration=CalibrationConfig(objective="precision", target=0.9),
        training=TrainingConfig(accelerator="cpu"),
    )
    path = tmp_path / "trainrun.yaml"
    run.save_to_file(path)
    assert "calibration:" in path.read_text(encoding="utf-8")
    assert TrainRunConfig.load_from_file(path) == run


def test_a_trainrun_carries_the_block_through_the_proto_bytes():
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    with_block = _run(calibration=CalibrationConfig(objective="recall", target=0.9))
    assert TrainRunConfig.from_proto(with_block.to_proto()) == with_block
    without = _run()
    assert b"calibration" not in without.to_proto().config_bytes
    assert TrainRunConfig.from_proto(without.to_proto()) == without


def test_a_trainrun_with_an_unknown_key_in_the_block_is_refused():
    with pytest.raises(ValidationError):
        TrainRunConfig.from_dict(
            {
                "name": "run",
                "data": {"data_module": "cu3s"},
                "calibration": {"objective": "recall", "target": 0.9, "goal": 1},
            }
        )
