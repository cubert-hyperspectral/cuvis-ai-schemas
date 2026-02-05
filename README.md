# cuvis-ai-schemas

Lightweight schema definitions for the cuvis-ai ecosystem.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

## Overview

`cuvis-ai-schemas` is a centralized, dependency-light package of schema definitions used across the cuvis-ai ecosystem. It enables type-safe communication between services without heavy runtime requirements.

Key points:
- Minimal deps (pydantic + pyyaml)
- Full Pydantic validation
- Optional extras for proto, torch, numpy, lightning

## Installation

```bash
uv add cuvis-ai-schemas
uv add "cuvis-ai-schemas[proto]"
uv add "cuvis-ai-schemas[full]"
```

Extras:
- `proto`: gRPC and protobuf support
- `torch`: PyTorch dtype handling (validation only)
- `numpy`: NumPy array support
- `lightning`: PyTorch Lightning training configs
- `full`: All features
- `dev`: Development dependencies

## Usage

```python
from cuvis_ai_schemas.pipeline import PipelineConfig, NodeConfig

pipeline = PipelineConfig(
    nodes=[NodeConfig(id="node_1", class_name="DataLoader", params={"batch_size": 32})],
    connections=[],
)

pipeline_json = pipeline.to_json()
pipeline = PipelineConfig.from_json(pipeline_json)
```

## Development

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check cuvis_ai_schemas/ tests/
uv run ruff format cuvis_ai_schemas/ tests/
uv run mypy cuvis_ai_schemas/
```

## Contributing

Contributions are welcome. Please:
1. Ensure tests pass
2. Run ruff format and ruff check
3. Keep type hints and update docs as needed

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
