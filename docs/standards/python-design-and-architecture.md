# Python Design And Architecture Standards

These standards apply to Python code in `kaos-llm-core`.

`kaos-llm-core` publishes the `kaos_llm_core` import package. This package publishes CLI entry point(s): `kaos-llm-core`, `kaos-llm-core-serve`.

## Package Shape

- Keep the import package name aligned with the distribution name:
  `kaos-llm-core` publishes import package `kaos_llm_core`.
- Declare the public API in `kaos_llm_core.__all__` when the package has
  a top-level public surface.
- Keep `py.typed` in the wheel for typed packages.
- Keep import-time work minimal: no network calls, filesystem scans,
  provider initialization, logging setup, or expensive model loads at
  import time.
- Use absolute imports for package code.
- Keep optional dependencies behind extras and lazy imports.
- Prefer a small top-level package surface that re-exports stable,
  documented names only.

## Public API

Treat all of these as public API once released:

- Names exported from `kaos_llm_core.__all__`.
- Documented modules, classes, functions, dataclasses, Pydantic models,
  protocols, and type aliases.
- CLI commands, flags, `--json` output, and exit behavior.
- MCP tools, schemas, resource names, prompts, and error shapes.
- Pydantic models used for external input/output.
- Environment-variable names and documented configuration fields.

Changing or removing public API requires a changelog entry and a version
bump consistent with the package's stability policy.

## Dependency Boundaries

- Keep base dependencies small and justified.
- Put optional integrations in named extras. Public extras currently declared: `otel`, `vision`.
- Do not import optional dependencies at module import time.
- Do not make tests pass by relying on undeclared transitive
  dependencies.
- Do not use private APIs from dependencies unless the risk is recorded
  and covered by tests.
- Centralize dependency-specific adapters so provider or parser changes
  do not leak through the package.

## Data Modeling

- Use frozen, slotted dataclasses for small internal value objects and
  result records.
- Use Pydantic for external boundaries: configuration, JSON-like inputs,
  API payloads, MCP schemas, and serialized results.
- Keep parsing and validation at boundaries. Internal functions should
  receive typed, normalized values.
- Prefer explicit result types over loosely shaped dictionaries.
- Avoid returning ambiguous tuples from public APIs.

## Functions And Classes

- Prefer functions for stateless transformations.
- Use classes when there is persistent state, lifecycle management,
  shared configuration, caches, backends, clients, or an explicit
  protocol.
- Keep constructors cheap. Use explicit `connect`, `load`, `from_*`, or
  factory methods for expensive setup.
- Avoid inheritance unless the abstraction is stable and tested through
  multiple implementations.
- Prefer protocols or small composition points over deep class
  hierarchies.

## Configuration

- Define typed settings for package configuration.
- Read environment variables and config files at the edge, not deep in
  algorithmic code.
- Represent secrets with secret-aware types where available.
- Do not print, log, serialize, or include secrets in exception strings.
- Configuration resolution order must be documented when more than one
  source exists.

## Error Handling

- Define package-specific exception types for user-facing failure modes.
- Error messages should explain what failed, why it likely failed, and
  what the caller can do next.
- Do not expose stack traces, credentials, internal paths, or provider
  payloads in user-facing errors.
- Preserve original exceptions with exception chaining when debugging
  context matters.
- Validate untrusted inputs early and fail with bounded, predictable
  errors.

## Async And Concurrency

- Use async APIs for external I/O: HTTP, provider calls, subprocesses,
  and database calls.
- Use synchronous APIs for simple CPU-bound transformations unless the
  package already exposes an async surface.
- Bound concurrency with semaphores or worker limits.
- Apply timeouts to external calls.
- Offload blocking or CPU-heavy work from event loops.
- Make cancellation safe: clean up files, subprocesses, and client
  sessions.

## Files, Paths, And Inputs

- Accept `str` and `PathLike` inputs where file paths are part of the
  public API.
- Normalize paths at boundaries.
- Do not follow symlinks, traverse directories, or read arbitrary files
  unless the API explicitly permits it.
- Put size, row, token, page, recursion, and time limits on untrusted
  inputs.
- Prefer streaming for large files and corpora.

## CLI Design

- Every CLI command must support `--help`.
- Commands that produce machine-consumable output should support
  `--json`.
- Exit codes must be stable and documented for common failure modes.
- CLI errors should be concise and actionable.
- CLI examples in README and docs must be tested or manually verified
  before release.

## Documentation Expectations

- README quick starts must be runnable from a fresh environment.
- Examples should use public APIs only.
- Advanced docs belong under `docs/`.
- Any advertised provider, file format, parser, integration, or extra
  must have at least one test at the appropriate tier.
