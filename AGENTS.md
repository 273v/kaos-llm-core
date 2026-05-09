# Coding Agent Guidance

## Scope

This file is the canonical repository-local guidance for coding agents
working in `kaos-llm-core`. Keep changes focused, public, and suitable
for this standalone repository. Follow [CONTRIBUTING.md](CONTRIBUTING.md)
and the detailed standards in [docs/standards](docs/standards/) rather
than duplicating those rules here.

## Project Identity

- Distribution: `kaos-llm-core`.
- Import package: `kaos_llm_core`.
- Python: 3.13 or newer.
- Runtime shape: pure Python package with CLI entry points
  `kaos-llm-core` and `kaos-llm-core-serve`.
- Core public surface: typed signatures, programs, codecs, routers,
  metrics, optimizers, traces, batch execution, MCP tool integration,
  and starter helpers.

## Setup

Use `uv` for local environments and dependency management:

```bash
uv sync --group dev
uvx pre-commit install
```

Do not hand-edit generated artifacts, lockfiles, release metadata, or
build outputs unless the task explicitly concerns them.

## Local Checks

For code changes, run the local quality gate documented in
[CONTRIBUTING.md](CONTRIBUTING.md):

```bash
uv run ruff format --check kaos_llm_core tests
uv run ruff check kaos_llm_core tests
uv run ty check kaos_llm_core tests
uv run pytest tests/unit/ -q --no-cov
```

Use `ty`, not mypy. When packaging metadata, README rendering, or
release behavior changes, also run the documented build and
`twine check --strict` gate.

## Architecture Rules

- Treat public exports from `kaos_llm_core.__all__`, documented modules,
  CLI behavior, MCP tool schemas, JSON output, environment variables,
  Pydantic models, and serialized trace or envelope shapes as public
  contracts.
- Keep provider access routed through `kaos-llm-client`. Do not add
  direct provider SDK calls or transport-specific behavior in core
  program logic.
- Keep import-time behavior cheap and deterministic: no network calls,
  credential reads beyond explicit settings resolution, filesystem
  scans, or provider initialization at import time.
- Keep optional features behind extras and lazy imports.
- Keep examples and docs on public APIs only.
- Prefer small, typed extension points over broad inheritance or
  internal re-export growth.
- Follow the package design rules in
  [docs/standards/python-design-and-architecture.md](docs/standards/python-design-and-architecture.md)
  and the quality rules in
  [docs/standards/code-quality-standards.md](docs/standards/code-quality-standards.md).

## Typed LLM Programming Principles

- Signatures, programs, codecs, optimizers, metrics, invocation records,
  traces, envelopes, and batch records are user-facing contracts. Keep
  field names, prompt/message structure, schema formats, and error
  shapes stable unless the task intentionally changes public behavior.
- Preserve deterministic local behavior where possible. Unit tests,
  alpha extractors, codec round trips, metric calculations, routing
  decisions, cache keys, envelope hashes, and batch resume identifiers
  should not depend on live services, wall-clock timing, or global
  mutable state unless explicitly designed to do so.
- Keep prompt, schema, message, and multimodal payload formats stable
  and covered by tests when changed.
- Make optimizer and metric behavior reproducible with explicit inputs,
  budgets, seeds, and recorded examples where the implementation
  supports them.
- Never leak secrets, provider payloads containing credentials, local
  paths, or private inputs through traces, errors, logs, CLI JSON, MCP
  responses, or exported artifacts.

## Testing

- Unit tests must stay deterministic and offline.
- Live and network tests are opt-in only. Tests that call provider APIs,
  public network services, or billable endpoints must be marked
  appropriately and must not run as part of the default unit gate.
- Credentials must never be committed, printed, logged, captured in
  traces, stored in fixtures, or included in failure messages.
- New public APIs need tests through their real public entry points.
- Bug fixes need focused regression tests.
- Fixture and CI rules live in
  [docs/standards/tests-fixtures-ci.md](docs/standards/tests-fixtures-ci.md).

## Security

- Do not commit secrets, `.env` files, private keys, customer data, or
  unknown-license fixtures.
- Bound untrusted input by size, recursion, time, token, row, page, and
  path limits where relevant.
- Keep path, URL, archive, subprocess, credential, and external-service
  checks intact.
- Report suspected vulnerabilities through [SECURITY.md](SECURITY.md),
  not public issues.

## Commits, PRs, And Releases

- Use conventional commit style and sign commits with DCO
  (`git commit -s`).
- Keep PRs to one logical change and document what changed, why, how it
  was tested, and whether public API, CLI behavior, schema output,
  fixtures, package metadata, or release artifacts changed.
- Update `CHANGELOG.md` for user-visible public API, CLI, schema,
  package metadata, security behavior, or deprecation changes.
- Do not move public tags or force-push shared branches.
- Follow [docs/standards/engineering-process.md](docs/standards/engineering-process.md)
  for branch, PR, tag, release, and hotfix expectations.
