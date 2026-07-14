# Changelog

## 0.8.0 - 2026-07-09

- Folded `TrainerConfig` into `TrainingConfig` (breaking): deleted `cuvis_ai_schemas/training/trainer.py` and the `TrainerConfig` export; the 14 `pytorch_lightning.Trainer` keyword fields (`max_epochs`, `accelerator`, `devices`, `default_root_dir`, `precision`, `accumulate_grad_batches`, `enable_progress_bar`, `enable_checkpointing`, `log_every_n_steps`, `val_check_interval`, `check_val_every_n_epoch`, `gradient_clip_val`, `deterministic`, `benchmark`) plus `callbacks` now live flat on `TrainingConfig`. Removed the nested `trainer` field, the duplicated top-level `max_epochs` / `gradient_clip_val` / `accumulate_grad_batches` fields, the `_sync_trainer_fields` validator, and the dead `batch_size` / `num_workers` fields. A trainrun/training YAML or JSON with a `trainer:` / `batch_size` / `num_workers` key now fails `extra="forbid"` validation. The proto message name (`TrainingConfig`) is unchanged.
- Added `TrainingConfig.to_lightning_kwargs()`: returns the explicit allowlist of the 14 raw `pl.Trainer` keyword arguments (via `model_dump(include=..., exclude_none=True)`), excluding the orchestration fields (`seed`, `optimizer`, `scheduler`, `callbacks`) so they never reach `pl.Trainer(**kwargs)`.

## 0.7.0 - 2026-06-23

- Added shared-memory tensor transport to the gRPC contract. `Tensor` now carries a `oneof payload`, so a tensor travels either inline (`bytes raw_data`, field 3, unchanged and wire-compatible) or by reference to a shared-memory block via the new `ShmRef` message (`name`, `byte_offset` (reserved, always 0), `byte_size`). Same-host producers and consumers can hand off large cubes without copying them through the gRPC channel; readers that only understand `raw_data` keep working because its field number is unchanged.
- Raised the `proto` extra floors to `grpcio>=1.76.0` / `grpcio-tools>=1.76.0` and regenerated the `cuvis_ai_pb2` stubs for the new `ShmRef` message and `Tensor.shm_ref` field.

## 0.6.0 - 2026-06-18

- Changed `torch` to a lazy, guarded import so the pure-pydantic surface imports without it: importing `PipelineConfig` (or anything from `cuvis_ai_schemas.pipeline`) no longer pulls in `torch`, restoring the minimal core-dependency promise. `PortSpec.is_compatible_with` imports `torch` on first call and raises a clear `cuvis-ai-schemas[torch]` install hint when the extra is absent; the public `PortSpec` / `DimensionResolver` / `InputPort` / `OutputPort` surface is unchanged.
- Added `cuvis_ai_schemas.testing` (new `[testing]` extra): a shipped home for Hypothesis strategies (`model_strategy`, the per-model factories, and the `MODEL_STRATEGIES` registry over `BaseSchemaModel` subclasses) and dependency-free round-trip assertions (`assert_dict_roundtrip` / `assert_json_roundtrip` / `assert_dict_json_roundtrip`), so downstream repos reuse one set of property-test helpers instead of reinventing them. Importing it without the extra raises a clear install hint.
- Changed `TrainingConfig` trainer-field syncing so the three overlapping fields (`max_epochs`, `gradient_clip_val`, `accumulate_grad_batches`) follow one uniform rule: an explicitly-set top-level value wins, including an explicit `gradient_clip_val=None`, which now clears a stale `trainer.gradient_clip_val` (previously the asymmetric special-case left it in place).
- Changed the plugin manifest schema to a "capabilities" family (breaking): collapsed `catalog.py` / `plugin/config.py` / `plugin/manifest.py` into `plugin/manifest_capabilities.py`; renamed `CatalogPortSpec` → `NodePortSpec`, `CatalogNodeEntry` → `PluginCapabilityEntry`, `CatalogPluginEntry` → `PluginCapabilities`, `PluginConfig` → `PluginManifest` (source variants `GitPluginSource` / `LocalPluginSource`), and `provides` → `capabilities`. A manifest is now one bare plugin with a required explicit `name` (never the filename), not a `{plugins: {<name>: ...}}` map; old-shape manifests fail validation. Module loaders are `load_plugin_manifest` + `write_plugin_manifest` (plural loader removed).
- Added `kind` / `data_module_name` / `extras` to `PluginCapabilityEntry`: `kind: node | data_module` (default `node`); a `data_module` entry declares a globally-unique `data_module_name` and may list pip `extras`, a `node` entry sets neither.
- Changed proto `LoadPlugins` → singular `LoadPlugin` and `PluginInfo.provides` → `capabilities` (breaking, regenerated stubs): one `LoadPlugin` registers one plugin (`config_bytes` is a single manifest, not a list); the response is singular (`registered_plugin` + `error`). `InitializeSession.resolved_plugins_json` stays a JSON list.
- Changed `DataConfig` to be module-agnostic (breaking): replaced the cu3s-specific fields (`cu3s_file_path`, `annotation_json_path`, `train_ids` / `val_ids` / `test_ids`, `train_split` / `val_split`, `shuffle`, `processing_mode`) with `{data_module, splits, batch_size, num_workers, params}`. `data_module` selects a registered DataModule by `DATA_MODULE_NAME`; module-specific arguments ride in `params`.
- Changed `DataSplitConfig` to composable selectors over an attributed sample universe (breaking): now `{splits_path, leakage_check, universe_hash, train/val/test/predict: list[Selector]}`. Added `SampleRef` (content-derived `uid`), `SelectorKind` (`files / file_indices / dir_indices / stems / glob / tag / categories / all / union / except / intersect`), and `Selector`. Stages union in order, empty `predict` means all samples; `leakage_check` (`error` / `warn` / `off`) guards train/val/test disjointness at setup. Replaces the flat `train_ids/val_ids/test_ids/predict_ids` id-lists.
- Changed `TrainRunConfig.pipeline` to a path reference, not an inline config (breaking): `PipelineConfig | None` → `str | None`, resolved relative to the trainrun file (a bare name resolves on the pipeline search path). Inline mappings and Hydra `@pipeline` compositions are rejected with a fix-it hint.
- Added `LoadPipelineRequest.data_module` (proto, additive): optional `string data_module = 3`, the data-module name, so the orchestrator resolves that module's plugin and pip extras when it composes the child env at `LoadPipeline`. Carries only the name, not a `DataConfig`.
- Changed `PipelineMetadata.cuvis_ai_version` to auto-stamp the installed version: it was a hardcoded `"0.1.0"`; `__version__` now derives from the installed distribution (`importlib.metadata` / setuptools-scm) and the field's `default_factory` stamps it, preserving an already-recorded value on load.
- Hardened internals with no public-API change: dropped two unreachable `OptimizerConfig` validators, de-duplicated the `ConnectionConfig` endpoint parsing, the `Selector` structure validator (now table-driven), and the manifest empty-string checks, and reorganised the test suite to mirror the package layout (run with `--import-mode=importlib`).

## 0.5.2 - 2026-06-10

- The `dep_compat` workflow now fetches the shared cuvis-ai-core audit script from `main` instead of a feature branch (since merged). The script ships on cuvis-ai-core's `main` and in every release tag from v0.7.0 onward, so the audit gate runs against a live, maintained ref and the latest floor rules. CI-only change; no package or schema changes.

## 0.5.1 - 2026-06-09

- `extensions/ui/node_display.is_plugin` now reads the registry's `loaded_plugin_nodes` attribute (was `plugin_registry`), matching cuvis-ai-core's renamed node-class map. The plugin pill is still driven purely by class-name membership; only the attribute name changed. Pairs with cuvis-ai-core's matching rename (core floors `cuvis-ai-schemas[proto]>=0.5.1`).

## 0.5.0 - 2026-06-08

- Added `PipelineConfig.plugins`: an optional `list[str]` of **bare plugin names**. The `_validate_plugins` validator rejects object / inline / tag-override entries and non-identifier names: a plugin is declared by name and resolves to a manifest yaml in the plugins directory.
- Added catalog models in new `cuvis_ai_schemas/catalog.py`: `CatalogPortSpec`, `CatalogNodeEntry` (fully-qualified `class_name`), and `CatalogPluginEntry` (`from_manifest_entry`): the static node catalog the server reads to populate the palette without importing any plugin Python. `dtype` is permissive (empty string = generic-tensor marker, mapped to `D_TYPE_UNSPECIFIED`); `category` / `tags` are plain strings so newer readers degrade gracefully. `CatalogNodeEntry.class_name` is validated as a fully-qualified dotted path of Python identifiers, so malformed forms such as `pkg.`, `.Node`, or `pkg..Node` are rejected on both the authoring and server-load paths.
- Added the plugin-config contract here so schemas is the single source of truth (cuvis-ai-core drops its fork): `_BasePluginConfig` gains an optional `package_name` author override, and `PluginConfig = GitPluginConfig | LocalPluginConfig` is exported. `provides` carries `CatalogNodeEntry` items directly, so the install list and the node catalog are the same list.
- **Breaking (proto):** renamed `LoadPluginsResponse.loaded_plugins` → `registered_plugins` (wire tag 1 retained). Reflects register-only `LoadPlugins` in cuvis-ai-core: entries are registered as catalog metadata, not installed; materialisation moves to `LoadPipeline` via the pipeline's `plugins:` field. Regenerated the Python stubs.
- Added an internal `RunRuntime` service section to `cuvis_ai.proto` (parent ↔ child runtime). Private `InitializeSession` hands the child its parent-owned session id, search paths, the resolved plugin dict, and FS paths; the parent forwards `LoadPipeline` / `LoadPipelineWeights` / `RestoreTrainRun` / `Inference` / `Train` / `CloseSession`, the persistence + introspection RPCs (`SavePipeline`, `SaveTrainRun`, `GetPipelineInputs` / `GetPipelineOutputs` / `GetPipelineVisualization`, `SetTrainRunConfig`, `GetTrainStatus`), `StopRun` (graceful cancel with a grace window), and `HealthCheck`. Reuses the public message types; the public `CuvisAIService` surface is untouched. Regenerated stubs via `buf generate`.
- **Breaking (proto):** a node port now maps to exactly one `PortSpec`. Added `bool variadic` to the `PortSpec` message, removed the `PortSpecList` message, and changed `NodeInfo.input_specs` / `NodeInfo.output_specs` from `map<string, PortSpecList>` to `map<string, PortSpec>`. Regenerated the Python stubs.
- Added `PortSpec.variadic` (default `False`) to the dataclass and `CatalogPortSpec.variadic`; `variadic` marks an *input* port for fan-in (the node receives a list) and is honored for input ports only. `extensions/ui/port_display.py` surfaces a `[Variadic]` tooltip tag.
- Changed `PortSpec.is_compatible_with` to take a single `PortSpec` (dropped the `PortSpec | list[PortSpec]` overload) and `CatalogNodeEntry.input_specs` / `output_specs` to `dict[str, CatalogPortSpec]`; removed the single-spec→list coercion.
- Collapsed plugin manifests so `provides` carries the node list directly (`class_name` = fully-qualified class), and restricted pipeline `plugins:` to bare manifest names.
- Tightened `ConnectionConfig` `source` / `target` validation to reject empty node or port segments (e.g. `.outputs.port` or `node.outputs.`), not just the wrong-arity case.
- Pinned dependency floors (`pydantic>=2.12.5`, `pyyaml>=6.0.3`; `uv.lock` in sync) and added a `dep_compat` workflow that audits the floors against the shared cuvis-ai-core audit script.

## 0.4.1 - 2026-05-04

- Widened `requires-python` from `<3.12` to `<3.14`; package now installs on Python 3.11, 3.12, and 3.13. No API or schema changes.
- Added Trove classifiers for Python 3.12 and 3.13.

## 0.4.0 - 2026-04-28

- Added `NodeCategory` proto enum (12 named values + `UNSPECIFIED`) capturing each node's exclusive graph role / execution contract (source, sink, transform, model, loss, metric, optimizer, scheduler, regularizer, runner, visualizer, control).
- Added `NodeTag` proto enum (~50 values, ID-range sub-namespaced into modality/task/lifecycle/properties/backend) for orthogonal axes a node can carry in any combination.
- Added `NodeInfo.icon_svg` (field 7, `bytes`), `NodeInfo.category` (field 8, `NodeCategory`), and `NodeInfo.tags` (field 9, `repeated NodeTag`); old `NodeInfo` payloads continue to deserialize cleanly under proto3 defaults.
- Added Python `NodeCategory` and `NodeTag` `StrEnum` mirrors at `cuvis_ai_schemas.enums`, with `NodeCategory.get_display_name()` returning the title-cased label.
- Added `cuvis_ai_schemas.extensions.ui.node_display` exposing `CATEGORY_STYLES`, `TAG_STYLES`, `resolve_display(node)`, and `is_plugin(node, registry)` as the canonical display module shared by the pipeline visualizer and the Qt UI palette.
- Added `cuvis_ai_schemas.grpc.conversions` with `node_category_to_proto` / `proto_to_node_category` and `node_tag_to_proto` / `proto_to_node_tag` helpers; unknown wire ints fall back to `UNSPECIFIED` (categories) or `None` (tags) for forward compatibility.
- Added 13 default monochrome SVG icons under `cuvis_ai_schemas/extensions/ui/icons/` (one per `NodeCategory` value), bundled into the wheel via a new `[tool.setuptools.package-data]` block.

## 0.3.0 - 2026-03-31

- Added per-node pipeline profiling transport with `SetProfiling` / `GetProfilingSummary` RPCs and `NodeProfilingStats`.
- Added optional `BoundingBox.object_id` transport for scheduled SAM3 bbox prompting while preserving `element_id` as the batch-element index.
- Changed discovery proto contracts to use canonical `pipeline_path` in `PipelineInfo` and `GetPipelineInfoRequest`, and explicit `resolved_path` for the concrete server-side file location, removing legacy `name`/`tags`/`has_weights` fields.
- Added explicit video-stream inputs to `InputBatch` (`mesu_index`, `rgb_image`, `frame_id`) and regenerated gRPC Python stubs.
- Removed top-level `PipelineConfig.name` in favor of metadata-backed naming and updated pipeline/base-model/proto consistency tests.
- Updated proto generation toolchain pins in `buf.gen.yaml`, protobuf dependency constraints in `pyproject.toml`, and GitHub Actions workflow dependencies for artifacts and Codecov.

## 0.2.0 - 2026-02-27

- Added `BaseSchemaModel` base class with unified serialization (`to_dict`, `from_dict`, `to_json`, `from_json`) and optional proto support via `__proto_message__`
- Added `-> Self` return type annotations on `BaseSchemaModel` classmethods (`from_dict`, `from_json`, `from_proto`)
- Added `create_callbacks_from_config()` utility for training callback instantiation (with guarded `pytorch_lightning` import)
- Added `Context.to_dict()` / `Context.from_dict()` and plugin serialization helpers
- Added `buf.yaml` and `buf.gen.yaml` for proto stub generation via `buf`
- Added breaking/lint rules to buf config
- Changed proto source location from `proto/cuvis_ai/` to `proto/cuvis_ai_schemas/` to match Python package namespace
- Changed all config models to inherit from `BaseSchemaModel`, removing 3 duplicate `_BaseConfig` classes
- Changed `NodeConfig`: renamed `id` → `name`, `params` → `hparams`, dropped `class` alias (now `class_name` only)
- Changed `ConnectionConfig`: replaced `from_node`/`from_port`/`to_node`/`to_port` fields with compact `source`/`target` dotted-path format (`"node.outputs.port"`)
- Changed `PipelineConfig` to fully typed Pydantic model with JSON-safe `save_to_file` serialization
- Fixed `Pipelinees` typo in proto source — renamed to `Pipelines` (message types, field names, and RPC method)
- Removed `VALIDATE` duplicate from `ExecutionStage` enum
- Removed obsolete `PipelineConfig.frozen_nodes` field
- Removed empty `discovery` module

## 0.1.4 - 2026-02-10

- Added changelog validation in release workflow
- Added security tooling: pip-audit, bandit, detect-secrets, cyclonedx-bom, pip-licenses
- Added CI security job with parallel pip-audit, detect-secrets, and bandit
- Added CI build-and-validate job with SBOM generation and license report
- Added Dependabot configuration for Python and GitHub Actions dependencies
- Added .secrets.baseline and bandit configuration
- Added version-tag sanity check in release workflow
- Added SBOM and license report attached to GitHub Releases
- Changed upload-artifact to v6 and download-artifact to v7 in release workflow
- Changed CI workflow with tags-ignore, concurrency control, and updated actions
- Changed release workflow to streamlined build-TestPyPI-PyPI-Release pipeline
- Changed license format to plain SPDX string for setuptools 77+
- Removed redundant License classifier superseded by SPDX field
- Removed dependency-groups section (twine moved to dev extras)

## 0.1.3 - 2026-02-07

- Added continuous integration workflow with test, typecheck, and lint
- Added Codecov integration with 80% coverage target
- Added README badges for PyPI, CI, codecov, license, and Python version
- Added PyPI metadata enhancements with keywords and classifiers
- Fixed pipeline port system with restored complete port compatibility checking
- Fixed CI branch triggers from develop to staging

## 0.1.2 - 2026-02-05

- Added automated CI/CD pipeline for PyPI releases via GitHub Actions
- Added tag-triggered release workflow for v*.*.* tags
- Added automated testing and type checking in CI
- Added TestPyPI publishing for pre-production verification
- Added production PyPI publishing with manual approval gate
- Added automatic GitHub Release creation with changelog extraction
- Added PyPI Trusted Publishing via OIDC
- Changed pytest to include coverage reporting
- Improved GitHub Release notes to parse CHANGELOG.md automatically
- Fixed workflow changelog extraction for version-specific release notes

## 0.1.0 - 2026-02-04

- Added initial release of cuvis-ai-schemas package
- Added pipeline structure schemas: PipelineConfig, NodeConfig, ConnectionConfig, PortSpec
- Added plugin system schemas: PluginManifest, GitPluginConfig, LocalPluginConfig
- Added training configuration schemas: TrainingConfig, DataConfig, OptimizerConfig
- Added execution context schemas: Context, ExecutionStage, Artifact, Metric
- Added discovery/metadata schemas: NodeInfo, PluginInfo, PipelineInfo
- Added gRPC proto definitions and generated Python stubs
- Added type conversion helpers for proto/Python interop
- Added UI extensions for port display with PortDisplaySpec and DTYPE_COLORS
- Added optional dependencies structure (proto, torch, numpy, lightning, full)
- Added comprehensive test suite with development tooling
- Added Pydantic validation, JSON/YAML/Proto serialization support
