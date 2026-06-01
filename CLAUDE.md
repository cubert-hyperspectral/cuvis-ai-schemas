# cuvis-ai-schemas

The dependency-light **schema/contract layer** of the Cuvis.AI ecosystem. Pure Pydantic v2
models that define the wire and disk formats every other repo speaks: `PipelineConfig`,
`NodeConfig`, `PortSpec`, `PluginRef`, `TrainerConfig`, and the execution types
(`Context`, `Artifact`, `Metric`), plus enums and the generated gRPC v1 stubs. Because
`cuvis-ai-core` and `cuvis-ai` both depend on this package, **changes here ripple the
widest** — treat schema edits as breaking-by-default and check downstream consumers.

## Part of the Cuvis.AI ecosystem

`cuvis-ai-schemas` (contracts, this repo) → `cuvis-ai-core` (framework: Node base class,
pipeline engine, gRPC, training) → `cuvis-ai` (node/operator library + CLIs) → plugins.
`cuvis-ai-cookbook` holds runnable examples; `cuvis-ai-agentic-skills` is a Claude Code
plugin; `dev-docs` holds internal ticket docs.

## Layout

- `cuvis_ai_schemas/base.py` — `BaseSchemaModel`, the root of every schema.
- `cuvis_ai_schemas/pipeline/` — `PipelineConfig`, `NodeConfig`, `PortSpec`, ports, connections.
- `cuvis_ai_schemas/plugin/` — `PluginRef` family (catalog / inline-git / inline-local).
- `cuvis_ai_schemas/training/` — `TrainerConfig`, optimizer/scheduler/data/callbacks configs.
- `cuvis_ai_schemas/execution/` — `Context`, `Artifact`, `Metric`, `InputStream`.
- `cuvis_ai_schemas/enums/` — `NodeCategory`, `NodeTag`, `ExecutionStage`.
- `cuvis_ai_schemas/extensions/` — UI icons (SVG) + extension config.
- `cuvis_ai_schemas/grpc/v1/` — **generated** protobuf stubs (`*_pb2`, `*_pb2_grpc`).

## Build & test

- Install (dev): `uv sync --all-extras --dev`. Extras: `[proto]`, `[torch]`, `[numpy]`,
  `[lightning]`, `[full]`. Use `uv`, never bare `pip`.
- Tests: `uv run pytest tests/ -v` (`xfail_strict=true` — an unexpected pass is a failure).

## Key abstraction

`BaseSchemaModel` (`base.py`) extends `pydantic.BaseModel` with `to_dict`/`from_dict`,
`to_json`/`from_json`, `to_proto`/`from_proto`. Config is strict: `extra="forbid"`,
`validate_assignment=True` — unknown fields raise, so keep configs exact.

## Conventions

- ruff line length **100**, double quotes; interrogate docstring coverage **95%** (strict).
- `grpc/v1/*_pb2*` are **generated — never hand-edit**; regenerate from the proto source.
- A `.githooks` pre-commit runs `ruff format` + `ruff check --fix`.
- No Jira IDs / "Phase N" / migration tags in shipped code, comments, or docstrings.
- No Claude/AI mentions or `Co-Authored-By` trailers in commit messages.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **cuvis-ai-schemas** (980 symbols, 1568 relationships, 1 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/cuvis-ai-schemas/context` | Codebase overview, check index freshness |
| `gitnexus://repo/cuvis-ai-schemas/clusters` | All functional areas |
| `gitnexus://repo/cuvis-ai-schemas/processes` | All execution flows |
| `gitnexus://repo/cuvis-ai-schemas/process/{name}` | Step-by-step execution trace |

## Cross-Repo Groups

This repository is listed under GitNexus **group(s): cuvis-ai-group** (see `~/.gitnexus/groups/`). For cross-repo analysis, use MCP tools `impact`, `query`, and `context` with `repo` set to `@<groupName>` or `@<groupName>/<memberPath>` (paths match keys in that group’s `group.yaml`). Use `group_list` / `group_sync` for membership and sync. From the terminal: `npx gitnexus group list`, `npx gitnexus group sync <name>`, `npx gitnexus group impact <name> --target <symbol> --repo <group-path>`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
