"""Test that all schema modules can be imported and instantiated correctly."""


def test_enums_import():
    """Test that enum types can be imported."""
    from cuvis_ai_schemas.enums import ArtifactType, ExecutionStage

    assert ArtifactType is not None
    assert ExecutionStage is not None


def test_execution_schemas_import():
    """Test that execution schemas can be imported."""
    from cuvis_ai_schemas.execution import Artifact, Context, Metric

    assert Artifact is not None
    assert Context is not None
    assert Metric is not None


def test_plugin_schemas_import():
    """Test that plugin schemas can be imported."""
    from cuvis_ai_schemas.plugin import (
        GitPluginConfig,
        LocalPluginConfig,
        PluginManifest,
    )

    assert GitPluginConfig is not None
    assert LocalPluginConfig is not None
    assert PluginManifest is not None


def test_pipeline_schemas_import():
    """Test that pipeline schemas can be imported."""
    from cuvis_ai_schemas.pipeline import (
        ConnectionConfig,
        NodeConfig,
        PipelineConfig,
        PipelineMetadata,
        PortSpec,
    )

    assert ConnectionConfig is not None
    assert NodeConfig is not None
    assert PipelineConfig is not None
    assert PipelineMetadata is not None
    assert PortSpec is not None


def test_training_schemas_import():
    """Test that training schemas can be imported."""
    from cuvis_ai_schemas.training import (
        CallbacksConfig,
        DataConfig,
        OptimizerConfig,
        PipelineConfig,
        PipelineMetadata,
        SchedulerConfig,
        TrainerConfig,
        TrainingConfig,
        TrainRunConfig,
    )

    assert CallbacksConfig is not None
    assert DataConfig is not None
    assert OptimizerConfig is not None
    assert PipelineConfig is not None
    assert PipelineMetadata is not None
    assert SchedulerConfig is not None
    assert TrainerConfig is not None
    assert TrainingConfig is not None
    assert TrainRunConfig is not None


def test_ui_extensions_import():
    """Test that UI extensions can be imported."""
    from cuvis_ai_schemas.extensions.ui import DTYPE_COLORS, PortDisplaySpec

    assert DTYPE_COLORS is not None
    assert PortDisplaySpec is not None


def test_context_instantiation():
    """Test that Context can be instantiated correctly."""
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    ctx = Context(stage=ExecutionStage.TRAIN, epoch=1)
    assert ctx.epoch == 1
    assert ctx.stage == ExecutionStage.TRAIN


def test_optimizer_config_instantiation():
    """Test that OptimizerConfig can be instantiated correctly."""
    from cuvis_ai_schemas.training import OptimizerConfig

    opt = OptimizerConfig(name="adamw", lr=0.001)
    assert opt.lr == 0.001
    assert opt.name == "adamw"


def test_port_spec_instantiation():
    """Test that PortSpec can be instantiated correctly."""
    from cuvis_ai_schemas.pipeline import PortSpec

    port = PortSpec(dtype="torch.Tensor", shape=(1, 3, 224, 224))
    assert port.shape == (1, 3, 224, 224)
    assert port.dtype == "torch.Tensor"
