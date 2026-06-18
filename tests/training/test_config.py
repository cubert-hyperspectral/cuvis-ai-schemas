"""Tests for training schemas."""

import sys
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cuvis_ai_schemas.training import (
    DataConfig,
    DataSplitConfig,
    OptimizerConfig,
    SampleRef,
    SchedulerConfig,
    Selector,
    SelectorKind,
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


def test_sync_matrix_explicit_top_level_wins():
    """An explicitly-set top-level value is pushed to the trainer, for all 3 fields."""
    config = TrainingConfig(
        max_epochs=50,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        trainer=TrainerConfig(max_epochs=7, gradient_clip_val=2.0, accumulate_grad_batches=2),
    )
    assert (config.max_epochs, config.trainer.max_epochs) == (50, 50)
    assert (config.gradient_clip_val, config.trainer.gradient_clip_val) == (1.0, 1.0)
    assert (config.accumulate_grad_batches, config.trainer.accumulate_grad_batches) == (4, 4)


def test_sync_matrix_trainer_value_pulled_up_when_top_unset():
    """A non-None trainer value is pulled up when the top-level field is unset."""
    config = TrainingConfig(
        trainer=TrainerConfig(max_epochs=7, gradient_clip_val=2.0, accumulate_grad_batches=2),
    )
    assert config.max_epochs == 7
    assert config.gradient_clip_val == 2.0
    assert config.accumulate_grad_batches == 2


def test_sync_matrix_explicit_none_clears_trainer_gradient_clip():
    """Explicit top-level gradient_clip_val=None now clears a stale trainer value.

    This is the unified-semantics fix: previously the trainer kept its value
    because an explicit top-level None was not pushed down.
    """
    config = TrainingConfig(
        gradient_clip_val=None,
        trainer=TrainerConfig(gradient_clip_val=2.0),
    )
    assert config.gradient_clip_val is None
    assert config.trainer.gradient_clip_val is None


def test_to_dict_config_without_omegaconf_returns_plain_dict():
    """Without omegaconf importable, to_dict_config falls back to a plain dict."""
    config = TrainingConfig(max_epochs=10)
    with patch.dict(sys.modules, {"omegaconf": None}):
        result = config.to_dict_config()
    assert isinstance(result, dict)
    assert result["max_epochs"] == 10


def test_to_dict_config_with_omegaconf_returns_dictconfig():
    """With omegaconf installed, to_dict_config returns a DictConfig (core's path)."""
    omegaconf = pytest.importorskip("omegaconf")
    config = TrainingConfig(max_epochs=10)
    result = config.to_dict_config()
    assert omegaconf.OmegaConf.is_config(result)
    assert not isinstance(result, dict)
    assert result["max_epochs"] == 10
    assert omegaconf.OmegaConf.to_container(result)["max_epochs"] == 10


def test_training_config_from_dict_config():
    """Test from_dict_config with a plain dict."""
    data = {"seed": 123, "max_epochs": 25, "batch_size": 8}
    config = TrainingConfig.from_dict_config(data)
    assert config.seed == 123
    assert config.max_epochs == 25


def test_from_dict_config_with_dictconfig():
    """from_dict_config converts an OmegaConf DictConfig before validating."""
    omegaconf = pytest.importorskip("omegaconf")
    dict_config = omegaconf.OmegaConf.create({"seed": 7, "max_epochs": 11})
    config = TrainingConfig.from_dict_config(dict_config)
    assert config.seed == 7
    assert config.max_epochs == 11


def test_from_dict_config_with_non_dict_mapping():
    """from_dict_config coerces a non-dict mapping into a dict before validating."""
    from types import MappingProxyType

    config = TrainingConfig.from_dict_config(MappingProxyType({"seed": 9, "batch_size": 4}))
    assert config.seed == 9
    assert config.batch_size == 4


def test_data_config():
    """Test DataConfig: module selection + selector splits + module-specific params."""
    data = DataConfig(
        data_module="cu3s",
        splits=DataSplitConfig(
            train=[Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[0, 2, 3])],
            val=[Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[1])],
            test=[Selector(kind=SelectorKind.CATEGORIES, any_of=["scrap"])],
        ),
        batch_size=16,
        num_workers=2,
        params={"cu3s_file_path": "/path/to/data.cu3s", "processing_mode": "Reflectance"},
    )
    assert data.data_module == "cu3s"
    assert data.splits is not None
    assert data.splits.train[0].source == "a.cu3s"
    assert data.splits.train[0].ids == [0, 2, 3]
    assert data.splits.leakage_check == "error"
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


def test_sample_ref_uid_content_derived():
    """SampleRef derives a stable content uid + group + stem when not supplied."""
    r = SampleRef(source="/data/a.cu3s", index=3)
    assert r.uid == "/data/a.cu3s#3"
    assert r.group == "/data/a.cu3s"
    assert r.stem == "a"
    # a distinct COCO id appends to the uid
    r2 = SampleRef(source="/data/a.cu3s", index=3, label_id=5)
    assert r2.uid == "/data/a.cu3s#3#5"
    # whole-file sample (no measurement index)
    r3 = SampleRef(source="/data/a.cu3s")
    assert r3.uid == "/data/a.cu3s"
    # an explicit uid is respected
    r4 = SampleRef(source="/data/a.cu3s", index=0, uid="row-7")
    assert r4.uid == "row-7"


def test_selector_file_indices_requires_source():
    """file_indices must carry an explicit source and non-empty ids."""
    Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s", ids=[0, "1-3"])
    with pytest.raises(ValidationError):
        Selector(kind=SelectorKind.FILE_INDICES, ids=[0, 1])  # no source
    with pytest.raises(ValidationError):
        Selector(kind=SelectorKind.FILE_INDICES, source="a.cu3s")  # no ids


def test_selector_dir_indices_forbids_source():
    """dir_indices addresses the file list; it must not set source."""
    Selector(kind=SelectorKind.DIR_INDICES, ids=[0, 2])
    with pytest.raises(ValidationError):
        Selector(kind=SelectorKind.DIR_INDICES, source="a.cu3s", ids=[0])


def test_selector_rejects_wrong_field_for_kind():
    """A selector may only set the fields valid for its kind (structure-only)."""
    with pytest.raises(ValidationError):
        Selector(kind=SelectorKind.STEMS, stems=["x"], pattern="y*")  # pattern not for stems
    with pytest.raises(ValidationError):
        Selector(kind=SelectorKind.ALL, paths=["a"])  # all takes nothing


def test_selector_kinds_construct():
    """The simple kinds construct with their own field."""
    Selector(kind=SelectorKind.FILES, paths=["a.cu3s", "b.cu3s"])
    Selector(kind=SelectorKind.STEMS, stems=["scrap_01"])
    Selector(kind=SelectorKind.GLOB, pattern="scrap_*")
    Selector(kind=SelectorKind.TAG, any_of=["normal"])
    Selector(kind=SelectorKind.ALL)


def test_selector_set_ops():
    """union/except/intersect compose nested selectors; except/intersect need >= 2."""
    inner = [
        Selector(kind=SelectorKind.TAG, any_of=["normal"]),
        Selector(kind=SelectorKind.FILES, paths=["a.cu3s"]),
    ]
    Selector(kind=SelectorKind.UNION, of=inner)
    Selector(kind=SelectorKind.EXCEPT, of=inner)
    with pytest.raises(ValidationError):
        Selector(kind=SelectorKind.INTERSECT, of=inner[:1])  # needs >= 2


@pytest.mark.parametrize(
    "kind",
    [
        SelectorKind.DIR_INDICES,
        SelectorKind.FILES,
        SelectorKind.STEMS,
        SelectorKind.GLOB,
        SelectorKind.TAG,
        SelectorKind.CATEGORIES,
        SelectorKind.UNION,
        SelectorKind.EXCEPT,
        SelectorKind.INTERSECT,
    ],
)
def test_selector_kind_requires_its_field(kind):
    """Each non-ALL kind rejects construction without its required field."""
    with pytest.raises(ValidationError):
        Selector(kind=kind)


def test_data_split_config_defaults_and_old_shape_rejected():
    """Defaults: leakage_check=error, empty predict; the old flat shape is rejected."""
    s = DataSplitConfig()
    assert s.leakage_check == "error"
    assert s.predict == []
    with pytest.raises(ValidationError):
        DataSplitConfig(train_ids=[0, 1])  # extra=forbid: the flat field is gone


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


def test_trainrun_pipeline_is_a_reference():
    """``pipeline`` is a path-string reference to a pipeline YAML, round-tripped verbatim."""
    config = TrainRunConfig(
        name="ref_run",
        pipeline="../pipeline/anomaly/adaclip/adaclip_baseline.yaml",
        data=DataConfig(data_module="cu3s"),
    )
    assert config.pipeline == "../pipeline/anomaly/adaclip/adaclip_baseline.yaml"
    assert TrainRunConfig.from_dict(config.to_dict()).pipeline == config.pipeline


def test_trainrun_rejects_inline_pipeline():
    """An inline pipeline mapping is rejected with a fix-it hint pointing at the reference."""
    with pytest.raises(ValueError, match="no longer supported"):
        TrainRunConfig(
            name="inline_run",
            pipeline={"nodes": [], "connections": []},
            data=DataConfig(data_module="cu3s"),
        )


def test_trainrun_rejects_non_string_pipeline():
    """A non-string, non-mapping pipeline value is rejected as not a path reference."""
    with pytest.raises(ValueError, match="must be a path string"):
        TrainRunConfig(name="bad", pipeline=123, data=DataConfig())
