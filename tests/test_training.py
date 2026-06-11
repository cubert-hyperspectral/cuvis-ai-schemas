"""Tests for training schemas."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training import (
    DataConfig,
    DataSplitConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
    create_callbacks_from_config,
)
from cuvis_ai_schemas.training.callbacks import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
)
from cuvis_ai_schemas.training.run import TrainRunConfig


def test_optimizer_config():
    """Test OptimizerConfig."""
    opt = OptimizerConfig(name="adamw", lr=0.001, weight_decay=0.01)
    assert opt.name == "adamw"
    assert opt.lr == 0.001
    assert opt.weight_decay == 0.01

    # Test lr validation
    with pytest.raises(ValidationError):
        OptimizerConfig(name="adam", lr=0)


def test_scheduler_config():
    """Test SchedulerConfig."""
    scheduler = SchedulerConfig(
        name="cosine",
        warmup_epochs=5,
        min_lr=1e-6,
        t_max=100,
    )
    assert scheduler.name == "cosine"
    assert scheduler.warmup_epochs == 5


def test_trainer_config():
    """Test TrainerConfig."""
    trainer = TrainerConfig(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
    )
    assert trainer.max_epochs == 100
    assert trainer.accelerator == "gpu"


def test_training_config():
    """Test TrainingConfig with trainer field syncing."""
    config = TrainingConfig(
        seed=42,
        max_epochs=50,
        batch_size=32,
        optimizer=OptimizerConfig(name="adamw", lr=0.001),
    )

    # Verify trainer fields are synced
    assert config.trainer.max_epochs == 50
    assert config.max_epochs == 50

    # Test JSON serialization
    json_str = config.to_json()
    loaded = TrainingConfig.from_json(json_str)
    assert loaded.seed == config.seed
    assert loaded.max_epochs == config.max_epochs


def test_training_config_trainer_syncs_gradient_clip():
    """Test _sync_trainer_fields for gradient_clip_val."""
    # Trainer provides gradient_clip_val, top-level not set
    config = TrainingConfig(
        trainer=TrainerConfig(gradient_clip_val=1.0),
    )
    assert config.gradient_clip_val == 1.0

    # Top-level provides gradient_clip_val, synced to trainer
    config = TrainingConfig(gradient_clip_val=0.5)
    assert config.trainer.gradient_clip_val == 0.5


def test_training_config_trainer_syncs_accumulate_grad():
    """Test _sync_trainer_fields for accumulate_grad_batches."""
    config = TrainingConfig(
        trainer=TrainerConfig(accumulate_grad_batches=4),
    )
    assert config.accumulate_grad_batches == 4


def test_training_config_callbacks_synced_to_trainer():
    """Test that callbacks are synced to trainer config."""
    cb = CallbacksConfig(early_stopping=[EarlyStoppingConfig(monitor="val_loss")])
    config = TrainingConfig(callbacks=cb)
    assert config.trainer.callbacks is cb


def test_training_config_to_dict_config_fallback():
    """Test to_dict_config falls back to model_dump when omegaconf unavailable."""
    config = TrainingConfig(max_epochs=10)
    result = config.to_dict_config()
    assert isinstance(result, dict)
    assert result["max_epochs"] == 10


def test_training_config_from_dict_config():
    """Test from_dict_config with a plain dict."""
    data = {"seed": 123, "max_epochs": 25, "batch_size": 8}
    config = TrainingConfig.from_dict_config(data)
    assert config.seed == 123
    assert config.max_epochs == 25


def test_data_config():
    """Test DataConfig: module selection + splits + module-specific params."""
    data = DataConfig(
        data_module="cu3s",
        splits=DataSplitConfig(train_ids=[0, 2, 3], val_ids=[1], test_ids=[1]),
        batch_size=16,
        num_workers=2,
        params={"cu3s_file_path": "/path/to/data.cu3s", "processing_mode": "Reflectance"},
    )
    assert data.data_module == "cu3s"
    assert data.splits is not None
    assert data.splits.train_ids == [0, 2, 3]
    assert data.batch_size == 16
    assert data.num_workers == 2
    assert data.params["cu3s_file_path"] == "/path/to/data.cu3s"


def test_data_config_defaults():
    """DataConfig is constructible with defaults (no required module-specific fields)."""
    data = DataConfig()
    assert data.data_module == "cu3s"
    assert data.splits is None
    assert data.batch_size == 1
    assert data.params == {}


def test_data_split_config_accepts_str_selectors():
    """Split selectors may be str keys (e.g. TIFF stems) as well as ints."""
    splits = DataSplitConfig(train_ids=["scrap_01", "scrap_07"], predict_ids=[0, 1])
    assert splits.train_ids == ["scrap_01", "scrap_07"]
    assert splits.predict_ids == [0, 1]


def test_create_callbacks_from_config_none():
    """Test create_callbacks_from_config with None returns empty list."""
    assert create_callbacks_from_config(None) == []


def test_create_callbacks_from_config_all_types():
    """Test create_callbacks_from_config creates all callback types."""
    config = CallbacksConfig(
        early_stopping=[EarlyStoppingConfig(monitor="val_loss", patience=5)],
        checkpoint=ModelCheckpointConfig(monitor="val_loss", save_top_k=1),
        learning_rate_monitor=LearningRateMonitorConfig(logging_interval="epoch"),
    )
    callbacks = create_callbacks_from_config(config)
    assert len(callbacks) == 3

    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    types = {type(cb) for cb in callbacks}
    assert EarlyStopping in types
    assert ModelCheckpoint in types
    assert LearningRateMonitor in types


def test_create_callbacks_from_config_import_error():
    """Test create_callbacks_from_config raises ImportError when lightning missing."""
    config = CallbacksConfig(early_stopping=[EarlyStoppingConfig(monitor="val_loss")])

    with patch.dict(
        "sys.modules", {"pytorch_lightning": None, "pytorch_lightning.callbacks": None}
    ):
        with pytest.raises(ImportError, match="pytorch_lightning is required"):
            create_callbacks_from_config(config)


def test_trainrun_save_and_load(tmp_path):
    """Test TrainRunConfig save_to_file and load_from_file round-trip."""
    config = TrainRunConfig(
        name="test_run",
        data=DataConfig(data_module="cu3s", params={"cu3s_file_path": "/data/test.cu3s"}),
    )
    path = tmp_path / "trainrun.yaml"
    config.save_to_file(path)
    loaded = TrainRunConfig.load_from_file(path)
    assert loaded.name == "test_run"
    assert loaded.data.data_module == "cu3s"
    assert loaded.data.params["cu3s_file_path"] == "/data/test.cu3s"
