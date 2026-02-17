"""Training configuration schemas for cuvis-ai."""

from cuvis_ai_schemas.pipeline.config import PipelineConfig, PipelineMetadata
from cuvis_ai_schemas.training.callbacks import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
    create_callbacks_from_config,
)
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.data import DataConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.run import TrainRunConfig
from cuvis_ai_schemas.training.scheduler import SchedulerConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig

__all__ = [
    # Callbacks
    "CallbacksConfig",
    "EarlyStoppingConfig",
    "LearningRateMonitorConfig",
    "ModelCheckpointConfig",
    "create_callbacks_from_config",
    # Config
    "TrainingConfig",
    # Data
    "DataConfig",
    # Optimizer
    "OptimizerConfig",
    # Scheduler
    "SchedulerConfig",
    # Trainer
    "TrainerConfig",
    # Run
    "PipelineConfig",
    "PipelineMetadata",
    "TrainRunConfig",
]
