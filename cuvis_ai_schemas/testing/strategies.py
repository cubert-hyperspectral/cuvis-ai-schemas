"""Hypothesis strategies for cuvis-ai-schemas models.

One importable source of truth for generating valid schema instances, so the
schemas suite and downstream repos (cuvis-ai, cuvis-ai-core) do not each
hand-roll generators. Requires the optional ``testing`` extra::

    pip install cuvis-ai-schemas[testing]

The public surface is :data:`MODEL_STRATEGIES` (a model class -> strategy
registry), :func:`model_strategy`, and the individual ``*_strategy`` factories
for composition. Models with opaque or heavily-unioned fields (e.g.
``Artifact.value`` numpy arrays, ``ModelCheckpointConfig.train_time_interval``)
are intentionally absent; add them when a tractable strategy is worth it.
"""

from __future__ import annotations

try:
    from hypothesis import strategies as st
except ImportError as exc:  # pragma: no cover - covered by the guard test
    msg = (
        "cuvis_ai_schemas.testing requires the optional 'testing' extra. "
        "Install with: pip install cuvis-ai-schemas[testing]"
    )
    raise ImportError(msg) from exc

from hypothesis.strategies import SearchStrategy

from cuvis_ai_schemas.base import BaseSchemaModel
from cuvis_ai_schemas.pipeline.config import (
    ConnectionConfig,
    NodeConfig,
    PipelineConfig,
    PipelineMetadata,
)
from cuvis_ai_schemas.plugin.manifest_capabilities import (
    GitPluginSource,
    LocalPluginSource,
    NodePortSpec,
    PluginCapabilities,
    PluginCapabilityEntry,
)
from cuvis_ai_schemas.training.callbacks import (
    CallbacksConfig,
    EarlyStoppingConfig,
    LearningRateMonitorConfig,
)
from cuvis_ai_schemas.training.data import (
    DataConfig,
    DataSplitConfig,
    SampleRef,
    Selector,
    SelectorKind,
)
from cuvis_ai_schemas.training.optimizer import OptimizerConfig
from cuvis_ai_schemas.training.scheduler import SchedulerConfig

# ---------------------------------------------------------------------------
# Primitive building blocks (bounded so generation stays fast and JSON-safe)
# ---------------------------------------------------------------------------
#: A valid, short Python identifier (plugin names, node names, port names).
identifiers: SearchStrategy[str] = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,15}", fullmatch=True)

#: A fully-qualified dotted class path of >= 2 identifier segments.
fqcns: SearchStrategy[str] = st.lists(identifiers, min_size=2, max_size=4).map(".".join)

#: A short free-text string with no surprises.
short_text: SearchStrategy[str] = st.text(max_size=24)

#: JSON-round-trippable finite floats in the unit interval.
unit_floats: SearchStrategy[float] = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)

#: A path segment (no dots) for the ``node.outputs.port`` connection format.
_segments: SearchStrategy[str] = st.from_regex(r"[A-Za-z0-9_]{1,10}", fullmatch=True)

#: JSON-safe hyperparameter values.
_json_scalars: SearchStrategy[object] = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.booleans(),
    short_text,
)


def node_port_spec_strategy() -> SearchStrategy[NodePortSpec]:
    """Strategy for :class:`NodePortSpec` (wire port spec)."""
    return st.builds(
        NodePortSpec,
        dtype=st.sampled_from(["", "float32", "uint8", "int64", "bool"]),
        shape=st.lists(st.integers(min_value=-1, max_value=8), max_size=4),
        optional=st.booleans(),
        description=short_text,
        variadic=st.booleans(),
    )


def plugin_capability_entry_strategy() -> SearchStrategy[PluginCapabilityEntry]:
    """Strategy for :class:`PluginCapabilityEntry` (node or data_module)."""
    node = st.builds(
        PluginCapabilityEntry,
        class_name=fqcns,
        kind=st.just("node"),
        category=st.sampled_from(["unspecified", "transform", "source", "sink", "model"]),
        tags=st.lists(identifiers, max_size=3),
        icon_svg=short_text,
        input_specs=st.dictionaries(identifiers, node_port_spec_strategy(), max_size=2),
        output_specs=st.dictionaries(identifiers, node_port_spec_strategy(), max_size=2),
        doc_summary=short_text,
    )
    data_module = st.builds(
        PluginCapabilityEntry,
        class_name=fqcns,
        kind=st.just("data_module"),
        data_module_name=identifiers,
        extras=st.lists(identifiers, max_size=3),
    )
    return st.one_of(node, data_module)


def git_plugin_source_strategy() -> SearchStrategy[GitPluginSource]:
    """Strategy for a git-sourced :class:`GitPluginSource` manifest."""
    return st.builds(
        GitPluginSource,
        name=identifiers,
        repo=st.just("https://github.com/cubert-hyperspectral/plugin.git"),
        tag=st.from_regex(r"v[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2}", fullmatch=True),
        capabilities=st.lists(plugin_capability_entry_strategy(), min_size=1, max_size=3),
        package_name=st.one_of(st.none(), identifiers),
    )


def local_plugin_source_strategy() -> SearchStrategy[LocalPluginSource]:
    """Strategy for a local-path :class:`LocalPluginSource` manifest."""
    return st.builds(
        LocalPluginSource,
        name=identifiers,
        path=st.from_regex(r"[A-Za-z0-9_./-]{1,24}", fullmatch=True),
        capabilities=st.lists(plugin_capability_entry_strategy(), min_size=1, max_size=3),
        package_name=st.one_of(st.none(), identifiers),
    )


def plugin_capabilities_strategy() -> SearchStrategy[PluginCapabilities]:
    """Strategy for an install-stripped :class:`PluginCapabilities` set."""
    return st.builds(
        PluginCapabilities,
        plugin_name=identifiers,
        plugin_version=st.from_regex(r"[0-9]\.[0-9]\.[0-9]", fullmatch=True),
        capabilities=st.lists(plugin_capability_entry_strategy(), min_size=1, max_size=3),
    )


def _leaf_selector_strategy() -> SearchStrategy[Selector]:
    """Strategy for non-recursive (non set-op) :class:`Selector` kinds."""
    ids = st.lists(
        st.one_of(st.integers(min_value=0, max_value=20), identifiers), min_size=1, max_size=3
    )
    return st.one_of(
        st.builds(
            Selector,
            kind=st.just(SelectorKind.FILES),
            paths=st.lists(identifiers, min_size=1, max_size=3),
        ),
        st.builds(Selector, kind=st.just(SelectorKind.FILE_INDICES), source=identifiers, ids=ids),
        st.builds(Selector, kind=st.just(SelectorKind.DIR_INDICES), ids=ids),
        st.builds(
            Selector,
            kind=st.just(SelectorKind.STEMS),
            stems=st.lists(identifiers, min_size=1, max_size=3),
        ),
        st.builds(Selector, kind=st.just(SelectorKind.GLOB), pattern=identifiers),
        st.builds(
            Selector,
            kind=st.just(SelectorKind.TAG),
            any_of=st.lists(identifiers, min_size=1, max_size=3),
        ),
        st.builds(
            Selector,
            kind=st.just(SelectorKind.CATEGORIES),
            any_of=st.lists(identifiers, min_size=1, max_size=3),
        ),
        st.builds(Selector, kind=st.just(SelectorKind.ALL)),
    )


def selector_strategy() -> SearchStrategy[Selector]:
    """Strategy for any :class:`Selector`, including nested set operations."""
    return st.recursive(
        _leaf_selector_strategy(),
        lambda children: st.one_of(
            st.builds(
                Selector,
                kind=st.just(SelectorKind.UNION),
                of=st.lists(children, min_size=1, max_size=3),
            ),
            st.builds(
                Selector,
                kind=st.just(SelectorKind.EXCEPT),
                of=st.lists(children, min_size=2, max_size=3),
            ),
            st.builds(
                Selector,
                kind=st.just(SelectorKind.INTERSECT),
                of=st.lists(children, min_size=2, max_size=3),
            ),
        ),
        max_leaves=4,
    )


def sample_ref_strategy() -> SearchStrategy[SampleRef]:
    """Strategy for :class:`SampleRef` (identity fields left to the validator)."""
    return st.builds(
        SampleRef,
        source=identifiers,
        index=st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
        label_id=st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
        annotation=st.one_of(st.none(), identifiers),
        tags=st.lists(short_text, max_size=3),
        category_ids=st.lists(st.integers(min_value=0, max_value=50), max_size=3),
    )


def data_split_config_strategy() -> SearchStrategy[DataSplitConfig]:
    """Strategy for :class:`DataSplitConfig` (per-stage selector lists)."""
    split = st.lists(selector_strategy(), max_size=2)
    return st.builds(
        DataSplitConfig,
        splits_path=st.one_of(st.none(), identifiers),
        leakage_check=st.sampled_from(["error", "warn", "off"]),
        universe_hash=st.one_of(st.none(), identifiers),
        train=split,
        val=split,
        test=split,
        predict=split,
    )


def data_config_strategy() -> SearchStrategy[DataConfig]:
    """Strategy for :class:`DataConfig`."""
    return st.builds(
        DataConfig,
        data_module=identifiers,
        splits=st.one_of(st.none(), data_split_config_strategy()),
        batch_size=st.integers(min_value=1, max_value=64),
        num_workers=st.integers(min_value=0, max_value=8),
        params=st.dictionaries(identifiers, _json_scalars, max_size=3),
    )


def connection_config_strategy() -> SearchStrategy[ConnectionConfig]:
    """Strategy for :class:`ConnectionConfig` in the ``node.{outputs,inputs}.port`` form."""
    return st.builds(
        ConnectionConfig,
        source=st.tuples(_segments, _segments).map(lambda t: f"{t[0]}.outputs.{t[1]}"),
        target=st.tuples(_segments, _segments).map(lambda t: f"{t[0]}.inputs.{t[1]}"),
    )


def node_config_strategy() -> SearchStrategy[NodeConfig]:
    """Strategy for :class:`NodeConfig`."""
    return st.builds(
        NodeConfig,
        name=identifiers,
        class_name=fqcns,
        hparams=st.dictionaries(identifiers, _json_scalars, max_size=3),
    )


def pipeline_metadata_strategy() -> SearchStrategy[PipelineMetadata]:
    """Strategy for :class:`PipelineMetadata` (version pinned for determinism)."""
    return st.builds(
        PipelineMetadata,
        name=identifiers,
        description=short_text,
        created=short_text,
        tags=st.lists(short_text, max_size=3),
        author=short_text,
        cuvis_ai_version=st.from_regex(r"[0-9]\.[0-9]\.[0-9]", fullmatch=True),
    )


def pipeline_config_strategy() -> SearchStrategy[PipelineConfig]:
    """Strategy for :class:`PipelineConfig`."""
    return st.builds(
        PipelineConfig,
        plugins=st.one_of(st.none(), st.lists(identifiers, max_size=3)),
        nodes=st.lists(node_config_strategy(), max_size=3),
        connections=st.lists(connection_config_strategy(), max_size=3),
        metadata=st.one_of(st.none(), pipeline_metadata_strategy()),
    )


def optimizer_config_strategy() -> SearchStrategy[OptimizerConfig]:
    """Strategy for :class:`OptimizerConfig`."""
    return st.builds(
        OptimizerConfig,
        name=st.sampled_from(["adamw", "sgd", "adam"]),
        lr=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
        weight_decay=unit_floats,
        momentum=st.one_of(st.none(), unit_floats),
        betas=st.one_of(st.none(), st.tuples(unit_floats, unit_floats)),
    )


def scheduler_config_strategy() -> SearchStrategy[SchedulerConfig]:
    """Strategy for :class:`SchedulerConfig`."""
    return st.builds(
        SchedulerConfig,
        name=st.one_of(st.none(), st.sampled_from(["cosine", "step", "plateau"])),
        warmup_epochs=st.integers(min_value=0, max_value=10),
        min_lr=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        t_max=st.one_of(st.none(), st.integers(min_value=1, max_value=100)),
        step_size=st.one_of(st.none(), st.integers(min_value=1, max_value=50)),
        gamma=st.one_of(st.none(), unit_floats),
        patience=st.integers(min_value=0, max_value=20),
        cooldown=st.integers(min_value=0, max_value=10),
    )


def early_stopping_config_strategy() -> SearchStrategy[EarlyStoppingConfig]:
    """Strategy for :class:`EarlyStoppingConfig`."""
    return st.builds(
        EarlyStoppingConfig,
        monitor=identifiers,
        patience=st.integers(min_value=1, max_value=20),
        mode=st.sampled_from(["min", "max"]),
        min_delta=unit_floats,
    )


def learning_rate_monitor_config_strategy() -> SearchStrategy[LearningRateMonitorConfig]:
    """Strategy for :class:`LearningRateMonitorConfig`."""
    return st.builds(
        LearningRateMonitorConfig,
        logging_interval=st.sampled_from(["step", "epoch"]),
        log_momentum=st.booleans(),
        log_weight_decay=st.booleans(),
    )


def callbacks_config_strategy() -> SearchStrategy[CallbacksConfig]:
    """Strategy for :class:`CallbacksConfig` (checkpoint left at default)."""
    return st.builds(
        CallbacksConfig,
        early_stopping=st.lists(early_stopping_config_strategy(), max_size=2),
        learning_rate_monitor=st.one_of(st.none(), learning_rate_monitor_config_strategy()),
    )


#: Registry mapping each supported model class to its Hypothesis strategy.
MODEL_STRATEGIES: dict[type[BaseSchemaModel], SearchStrategy[BaseSchemaModel]] = {
    NodePortSpec: node_port_spec_strategy(),
    PluginCapabilityEntry: plugin_capability_entry_strategy(),
    GitPluginSource: git_plugin_source_strategy(),
    LocalPluginSource: local_plugin_source_strategy(),
    PluginCapabilities: plugin_capabilities_strategy(),
    Selector: selector_strategy(),
    SampleRef: sample_ref_strategy(),
    DataSplitConfig: data_split_config_strategy(),
    DataConfig: data_config_strategy(),
    ConnectionConfig: connection_config_strategy(),
    NodeConfig: node_config_strategy(),
    PipelineMetadata: pipeline_metadata_strategy(),
    PipelineConfig: pipeline_config_strategy(),
    OptimizerConfig: optimizer_config_strategy(),
    SchedulerConfig: scheduler_config_strategy(),
    EarlyStoppingConfig: early_stopping_config_strategy(),
    LearningRateMonitorConfig: learning_rate_monitor_config_strategy(),
    CallbacksConfig: callbacks_config_strategy(),
}


def model_strategy(model_cls: type[BaseSchemaModel]) -> SearchStrategy[BaseSchemaModel]:
    """Return the registered Hypothesis strategy for ``model_cls``.

    Raises
    ------
    KeyError
        If ``model_cls`` has no registered strategy (it may carry opaque or
        heavily-unioned fields; register one in this module when needed).
    """
    try:
        return MODEL_STRATEGIES[model_cls]
    except KeyError as exc:
        msg = (
            f"No Hypothesis strategy registered for {model_cls.__name__}. "
            "Add one to cuvis_ai_schemas.testing.strategies.MODEL_STRATEGIES."
        )
        raise KeyError(msg) from exc
