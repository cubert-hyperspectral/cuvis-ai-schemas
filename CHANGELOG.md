# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Changelog validation in release workflow (hard-fails if no entry for tag version)

## [0.1.4] - 2026-02-10

### Added
- Security tooling: pip-audit, bandit, detect-secrets, cyclonedx-bom, pip-licenses
- CI security job running pip-audit, detect-secrets, and bandit in parallel
- CI build-and-validate job with SBOM generation and license report
- Dependabot configuration for Python dependencies and GitHub Actions
- `.secrets.baseline` for detect-secrets
- `[tool.bandit]` configuration in pyproject.toml
- Version-tag sanity check in release workflow
- SBOM and license report attached to GitHub Releases

### Changed
- CI workflow: added `tags-ignore`, concurrency control, updated actions (checkout@v6, setup-uv@v7, codecov@v5)
- Release workflow: removed redundant test re-runs, streamlined to build → TestPyPI → PyPI → GitHub Release
- License format migrated to plain SPDX string (`license = "Apache-2.0"`) for setuptools 77+

### Removed
- Redundant `License ::` classifier (superseded by SPDX license field)
- `[dependency-groups]` section (twine moved into dev extras)

## [0.1.3] - 2026-02-07

### Added
- Continuous integration workflow (test, typecheck, lint)
- Codecov integration with `codecov.yml` configuration (80% target)
- Codecov upload in both CI and release workflows
- README badges (PyPI version, CI status, codecov, license, Python version)
- PyPI metadata enhancements (keywords, expanded classifiers)

### Fixed
- Pipeline port system: restored complete port compatibility checking
- CI branch triggers changed from `develop` to `staging`

## [0.1.2] - 2026-02-05

### Added
- Automated CI/CD pipeline for PyPI releases via GitHub Actions
- Tag-triggered release workflow (triggers on `v*.*.*` tags)
- Automated testing and type checking in CI (pytest, mypy, ruff)
- Automated package building and validation with twine
- TestPyPI publishing for pre-production verification
- Production PyPI publishing with manual approval gate
- Automatic GitHub Release creation with changelog extraction
- GitHub Environments configuration (testpypi, pypi)
- PyPI Trusted Publishing (OIDC) for secure authentication

### Changed
- Enhanced pytest execution to include coverage reporting (`--cov --cov-report=term-missing`)
- Improved GitHub Release notes generation to parse CHANGELOG.md automatically
- Added `skip-existing: true` to TestPyPI publish step to prevent re-run failures

### Fixed
- Workflow now properly extracts version-specific release notes from CHANGELOG.md
- Release notes now include installation instructions with examples

## [0.1.0] - 2026-02-04

### Added
- Initial release of cuvis-ai-schemas package
- Core schema definitions extracted from cuvis-ai-core and cuvis-ai-ui
- Pipeline structure schemas (PipelineConfig, NodeConfig, ConnectionConfig, PortSpec)
- Plugin system schemas (PluginManifest, GitPluginConfig, LocalPluginConfig)
- Training configuration schemas (TrainingConfig, DataConfig, OptimizerConfig, etc.)
- Execution context schemas (Context, ExecutionStage, Artifact, Metric)
- Discovery/metadata schemas (NodeInfo, PluginInfo, PipelineInfo)
- gRPC proto definitions and generated Python stubs
- Type conversion helpers for proto/Python interop
- UI extensions for port display (PortDisplaySpec, DTYPE_COLORS)
- Optional dependencies structure (proto, torch, numpy, lightning, full)
- Comprehensive test suite
- Development tools (ruff, pytest, mypy configuration)
- Git hooks for pre-commit validation
- Complete documentation (README, API docs)

### Features
- Lightweight core dependencies (pydantic + pyyaml only)
- Optional extras for specific features
- Full Pydantic validation for all schemas
- JSON and YAML serialization support
- Proto serialization for gRPC communication
- Field aliases for backward compatibility
- Type-safe schema definitions
- UV package manager support
- PyTorch CUDA index configuration

### Schema Reconciliation
- Unified PipelineConfig from core and UI implementations
- Reconciled PortSpec with UI color extensions
- Centralized proto definitions (single source of truth)
- Consistent serialization patterns across all schemas

### Documentation
- Installation guide with examples
- Usage examples for all schema categories
- Development setup instructions
- Architecture overview
- Ecosystem integration guide
- Migration guide for consumers

---

**Legend:**
- `Added`: New features
- `Changed`: Changes in existing functionality
- `Deprecated`: Soon-to-be removed features
- `Removed`: Removed features
- `Fixed`: Bug fixes
- `Security`: Security fixes
