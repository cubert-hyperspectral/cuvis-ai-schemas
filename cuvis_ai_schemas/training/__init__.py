"""Training configuration schemas for cuvis-ai."""

from cuvis_ai_schemas.training.callbacks import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
    ModelCheckpointConfig,
)
from cuvis_ai_schemas.training.config import TrainingConfig
from cuvis_ai_schemas.training.data import DataConfig
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.run import (
    PipelineConfig,
    PipelineMetadata,
    TrainRunConfig,
)
from cuvis_ai_schemas.training.scheduler import SchedulerConfig
from cuvis_ai_schemas.training.trainer import TrainerConfig

__all__ = [
    # Callbacks
    "CallbacksConfig",
    "EarlyStoppingConfig",
    "LearningRateMonitorConfig",
    "ModelCheckpointConfig",
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
