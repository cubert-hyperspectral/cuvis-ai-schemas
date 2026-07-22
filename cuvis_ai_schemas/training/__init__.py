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
from cuvis_ai_schemas.training.data import (
    DEFAULT_CONSTRAINT_SEVERITY,
    Constraint,
    ConstraintKind,
    ConstraintSeverity,
    DataConfig,
    DataSplitConfig,
    SampleRef,
    Selector,
    SelectorKind,
    default_constraints,
)
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.run import TrainRunConfig
from cuvis_ai_schemas.training.scheduler import SchedulerConfig

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
    "DataSplitConfig",
    "SampleRef",
    "Selector",
    "SelectorKind",
    "Constraint",
    "ConstraintKind",
    "ConstraintSeverity",
    "DEFAULT_CONSTRAINT_SEVERITY",
    "default_constraints",
    # Optimizer
    "OptimizerConfig",
    # Scheduler
    "SchedulerConfig",
    # Run
    "PipelineConfig",
    "PipelineMetadata",
    "TrainRunConfig",
]
