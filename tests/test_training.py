"""Tests for training schemas."""

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training import (
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
    TrainingConfig,
)


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


def test_data_config():
    """Test DataConfig."""
    data = DataConfig(
        cu3s_file_path="/path/to/data.cu3s",
        train_split=0.8,
        val_split=0.1,
        batch_size=16,
        processing_mode="Reflectance",
    )
    assert data.cu3s_file_path == "/path/to/data.cu3s"
    assert data.train_split == 0.8
    assert data.batch_size == 16
