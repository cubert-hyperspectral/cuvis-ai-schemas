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


def test_training_config_flat_lightning_fields():
    """TrainingConfig carries the pl.Trainer kwargs flat (no nested trainer)."""
    config = TrainingConfig(
        seed=42,
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        optimizer=OptimizerConfig(name="adamw", lr=0.001),
    )
    assert config.max_epochs == 50
    assert config.accelerator == "gpu"
    assert config.precision == "16-mixed"
    assert not hasattr(config, "trainer")

    # JSON round-trip is stable on the flat shape
    loaded = TrainingConfig.from_json(config.to_json())
    assert loaded.seed == config.seed
    assert loaded.max_epochs == config.max_epochs
    assert loaded.accelerator == config.accelerator


def test_to_lightning_kwargs_is_the_trainer_allowlist():
    """to_lightning_kwargs returns exactly the raw pl.Trainer kwargs, nothing else."""
    config = TrainingConfig(
        seed=7,
        max_epochs=20,
        accelerator="gpu",
        optimizer=OptimizerConfig(name="adamw"),
        scheduler=SchedulerConfig(name="cosine", t_max=10),
        callbacks=CallbacksConfig(early_stopping=[EarlyStoppingConfig(monitor="val_loss")]),
        gradient_clip_val=1.0,
    )
    kwargs = config.to_lightning_kwargs()

    # Orchestration fields never leak into pl.Trainer(**kwargs)
    for orchestration in ("seed", "optimizer", "scheduler", "callbacks"):
        assert orchestration not in kwargs

    # Every returned key is a real Lightning field, and set fields survive
    assert set(kwargs) <= set(TrainingConfig._LIGHTNING_FIELDS)
    assert kwargs["max_epochs"] == 20
    assert kwargs["accelerator"] == "gpu"
    assert kwargs["gradient_clip_val"] == 1.0
    # exclude_none drops unset optionals
    assert "devices" not in kwargs


def test_old_nested_trainer_shape_is_rejected():
    """The pre-fold nested-trainer / dead-field shape fails under extra=forbid."""
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"seed": 42, "trainer": {"max_epochs": 20}})
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"max_epochs": 20, "batch_size": 32})
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({"max_epochs": 20, "num_workers": 4})


def test_lightning_field_defaults():
    """The pl.Trainer-derived fields keep their documented defaults."""
    config = TrainingConfig()
    assert config.max_epochs == 100
    assert config.accelerator == "auto"
    assert config.precision == "32-true"
    assert config.devices is None
    assert config.callbacks is None


@pytest.mark.parametrize("precision", ["16-mixed", "bf16", 16, 32])
def test_precision_accepts_str_or_int(precision):
    """precision is a str | int union."""
    assert TrainingConfig(precision=precision).precision == precision


@pytest.mark.parametrize("devices", [1, "auto", None])
def test_devices_accepts_int_str_or_none(devices):
    """devices is an int | str | None union."""
    assert TrainingConfig(devices=devices).devices == devices


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_epochs": 0},
        {"accumulate_grad_batches": 0},
        {"log_every_n_steps": 0},
        {"check_val_every_n_epoch": 0},
        {"gradient_clip_val": -1.0},
        {"val_check_interval": -0.5},
    ],
)
def test_out_of_range_lightning_values_rejected(kwargs):
    """Field range constraints reject out-of-range Lightning values."""
    with pytest.raises(ValidationError):
        TrainingConfig(**kwargs)


def test_callbacks_survive_dict_round_trip():
    """A nested CallbacksConfig survives a dict round-trip on the flat config."""
    config = TrainingConfig(
        callbacks=CallbacksConfig(early_stopping=[EarlyStoppingConfig(monitor="val_loss")])
    )
    assert TrainingConfig.from_dict(config.to_dict()) == config


def test_extra_field_rejected():
    """Unknown fields are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        TrainingConfig(unknown="x")


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
    data = {"seed": 123, "max_epochs": 25, "accelerator": "cpu"}
    config = TrainingConfig.from_dict_config(data)
    assert config.seed == 123
    assert config.max_epochs == 25
    assert config.accelerator == "cpu"


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

    config = TrainingConfig.from_dict_config(MappingProxyType({"seed": 9, "max_epochs": 4}))
    assert config.seed == 9
    assert config.max_epochs == 4


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


def test_trainrun_with_training_round_trips_flat(tmp_path):
    """A trainrun carrying a training block round-trips without a nested ``trainer`` key."""
    config = TrainRunConfig(
        name="grad_run",
        data=DataConfig(data_module="cu3s", params={"cu3s_file_path": "/data/test.cu3s"}),
        training=TrainingConfig(seed=7, max_epochs=5, accelerator="cpu"),
    )
    path = tmp_path / "trainrun.yaml"
    config.save_to_file(path)

    assert "trainer:" not in path.read_text(encoding="utf-8")
    loaded = TrainRunConfig.load_from_file(path)
    assert loaded.training is not None
    assert loaded.training.max_epochs == 5
    assert not hasattr(loaded.training, "trainer")


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
