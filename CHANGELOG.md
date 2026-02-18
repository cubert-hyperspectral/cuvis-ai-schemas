# Changelog

## [Unreleased]

- Added `BaseSchemaModel` base class with unified serialization (`to_dict`, `from_dict`, `to_json`, `from_json`) and optional proto support
- Added `create_callbacks_from_config` utility for training callback instantiation
- Added `Context` dataclass and plugin serialization support
- Changed all config models to inherit from `BaseSchemaModel`, removing 3 duplicate `_BaseConfig` classes
- Changed `NodeConfig.params` to `hparams` for consistency with runtime layer
- Changed `PipelineConfig` to fully typed Pydantic model with JSON-safe serialization
- Removed `VALIDATE` duplicate from `ExecutionStage` enum
- Removed obsolete `PipelineConfig.frozen_nodes` field
- Removed empty `discovery` module
- Reformatted CHANGELOG to concise single-list style
- Updated release workflow changelog extraction for new heading format

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
