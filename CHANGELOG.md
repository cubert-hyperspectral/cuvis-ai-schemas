# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [Unreleased]

### Planned
- Additional schema validators
- Performance optimizations
- Extended documentation
- More usage examples

---

**Legend:**
- `Added`: New features
- `Changed`: Changes in existing functionality
- `Deprecated`: Soon-to-be removed features
- `Removed`: Removed features
- `Fixed`: Bug fixes
- `Security`: Security fixes
