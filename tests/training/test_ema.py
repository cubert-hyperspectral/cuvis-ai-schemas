"""Tests for the weight-averaging (EMA) block of a training config."""

import json

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training import (
    DataConfig,
    EmaConfig,
    TrainingConfig,
    TrainRunConfig,
)
from cuvis_ai_schemas.training.config import _ORCHESTRATION_FIELDS


def test_defaults():
    block = EmaConfig()
    assert (block.decay, block.tau, block.update_interval) == (0.999, 0.0, 1)
    assert block.update_on_epoch_end is False
    assert block.evaluate == "ema"


def test_rfdetr_style_block():
    block = EmaConfig(decay=0.993, tau=100, update_on_epoch_end=True)
    assert (block.decay, block.tau, block.update_on_epoch_end) == (0.993, 100.0, True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decay", 0.0),
        ("decay", 1.01),
        ("tau", -1.0),
        ("update_interval", 0),
        ("evaluate", "best"),
    ],
)
def test_out_of_range_values_are_refused(field, value):
    with pytest.raises(ValidationError):
        EmaConfig(**{field: value})


def test_unknown_keys_are_refused():
    with pytest.raises(ValidationError):
        EmaConfig.from_dict({"decay": 0.99, "momentum": 0.9})


def test_ema_is_a_runtime_field_never_passed_to_lightning():
    assert "ema" in _ORCHESTRATION_FIELDS
    assert "ema" not in TrainingConfig._LIGHTNING_FIELDS
    config = TrainingConfig(ema=EmaConfig())
    assert "ema" not in config.to_lightning_kwargs()


def test_a_strategy_is_registered_for_the_round_trip_properties():
    from cuvis_ai_schemas.testing import MODEL_STRATEGIES

    assert EmaConfig in MODEL_STRATEGIES


def test_a_config_without_the_block_serializes_as_before():
    """No ``ema`` key anywhere, so a reader that predates the field reads the config."""
    config = TrainingConfig()
    assert config.ema is None
    assert "ema" not in config.to_dict()
    assert "ema" not in json.loads(config.to_json())


def test_a_trainrun_without_the_block_writes_no_ema_key(tmp_path):
    run = TrainRunConfig(name="run", data=DataConfig(data_module="cu3s"), training=TrainingConfig())
    path = tmp_path / "trainrun.yaml"
    run.save_to_file(path)
    assert "ema" not in path.read_text(encoding="utf-8")


def test_the_block_round_trips_through_dict_json_and_yaml(tmp_path):
    block = EmaConfig(decay=0.993, tau=100, update_interval=2, evaluate="live")
    config = TrainingConfig(ema=block)
    assert config.to_dict()["ema"] == {
        "decay": 0.993,
        "tau": 100.0,
        "update_interval": 2,
        "update_on_epoch_end": False,
        "evaluate": "live",
    }
    assert TrainingConfig.from_dict(config.to_dict()) == config
    assert TrainingConfig.from_json(config.to_json()) == config
    run = TrainRunConfig(name="run", data=DataConfig(data_module="cu3s"), training=config)
    path = tmp_path / "trainrun.yaml"
    run.save_to_file(path)
    assert "ema:" in path.read_text(encoding="utf-8")
    assert TrainRunConfig.load_from_file(path) == run


def test_the_block_round_trips_through_the_proto_bytes():
    pytest.importorskip("cuvis_ai_schemas.grpc.v1.cuvis_ai_pb2")
    with_block = TrainingConfig(ema=EmaConfig(decay=0.99))
    assert TrainingConfig.from_proto(with_block.to_proto()) == with_block
    without = TrainingConfig()
    assert b'"ema"' not in without.to_proto().config_bytes
    assert TrainingConfig.from_proto(without.to_proto()) == without
