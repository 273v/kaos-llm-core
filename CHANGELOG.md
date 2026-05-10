# Changelog

All notable changes to `kaos-llm-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Cross-platform import + sandbox preexec.** Two regressions surfaced
  by the new macOS-arm64 / Windows-x64 CI test legs:
  - `kaos_llm_core.programs.program_of_thought` did `import resource`
    at module top level. `resource` is POSIX-only, so on Windows
    *every* test that transitively imported `kaos_llm_core` failed to
    collect with `ModuleNotFoundError: No module named 'resource'`.
    The import is now gated behind `os.name == "posix"` and stored as
    a private `_resource` symbol; the same gate skips passing
    `preexec_fn` to `subprocess.run` on Windows (where the kwarg
    raises). The Windows sandbox is correspondingly weaker — only the
    wall-clock timeout + `-I -S` isolation apply; the production
    boundary continues to be POSIX-only as documented in the module
    header.
  - `_apply_rlimits` called `resource.setrlimit(RLIMIT_AS, ...)`
    unconditionally. macOS doesn't support `RLIMIT_AS` and the call
    raises `OSError(EINVAL)`, which propagated up as
    `subprocess.SubprocessError: Exception occurred in preexec_fn`
    and killed every sandbox test on the macOS-arm64 leg. RLIMIT_AS
    is now skipped on Darwin (`sys.platform == "darwin"`); RLIMIT_CPU
    + RLIMIT_FSIZE still apply.

  Regression coverage:
  `tests/unit/test_program_of_thought_crossplatform.py` — three
  tests (module-import shape, Darwin RLIMIT_AS skip, Linux all-three).
- **Tests: KLLC-02 / KLLC-03 absolute-path fixtures now use a
  platform-aware absolute path.** Four tests in
  ``tests/unit/test_security_regressions.py`` and one in
  ``tests/unit/test_tools.py`` asserted that production code rejects
  absolute paths like ``/etc/cron.d/escape-test`` with an error
  containing the word ``"absolute"``. On Windows, a leading ``/``
  without a drive letter is drive-relative — not absolute —
  ``Path("/etc/...").is_absolute()`` returns ``False`` there, so
  the security guard never fired and the tests collected a
  not-found / no-runtime-context error message instead of the
  expected ``"absolute"`` one. New ``_abs_escape_path`` helper
  emits a platform-appropriate absolute path
  (``/etc/<components>`` on POSIX,
  ``C:\\Windows\\System32\\drivers\\etc\\<components>`` on
  Windows). The security guards continue to do the rejection; only
  the test fixtures change. Files:
  ``tests/unit/test_security_regressions.py``,
  ``tests/unit/test_tools.py``.

## [0.1.0a3] — 2026-05-07

Page-level VLM programs relocated from `kaos_pdf.vision` to keep the
extraction → LLM dependency direction one-directional. Closes audit-01
PDF-001 (the published `kaos-pdf[vision]` extra had `kaos-pdf`
depending up on `kaos-llm-core`, which inverts the documented DAG).

### Added

- `kaos_llm_core.vision` subpackage with three async page programs:
  - `describe_page(image, *, model, instruction)` → `PageDescription`
  - `classify_page(image, *, model)` → `PageClassification`
    (10 categories: text, table, chart, form, signature_page, exhibit,
    photo, diagram, blank, mixed)
  - `ocr_page(image, *, model)` → `PageOCRResult` — VLM-based OCR,
    the high-accuracy complement to Tesseract
  All three accept a `KaosImage` (from kaos-content) and return a
  frozen-slots dataclass. Default model is `anthropic:claude-haiku-4-5`.
- New `[vision]` extra pulling `kaos-content[images]` (Pillow + numpy)
  so the `KaosImage` input type works out of the box.

### Migration

Replace:

    from kaos_pdf.vision import describe_page, classify_page, ocr_page

with:

    from kaos_llm_core.vision import describe_page, classify_page, ocr_page

The function signatures, default model, and return shapes are unchanged.
`kaos_pdf.vision` and the `kaos-pdf[vision]` extra are removed in the
next `kaos-pdf` release.

## [0.1.0a2] — 2026-05-07

audit-01 follow-ups. No public-API or behavior changes; all changes are
documentation, internal error-message guidance, and tooling configuration.

### Added

- Public namespace `__all__` declarations on the previously-implicit
  `kaos_llm_core.integrations` and `kaos_llm_core.integrations.mcp`
  package roots (both empty — these are pure namespace packages). Codifies
  that nothing is intentionally re-exported through those `__init__.py`s,
  which keeps `from kaos_llm_core.integrations import *` deterministic
  and import-tooling-friendly.
- `live` pytest marker registered in `[tool.pytest.ini_options].markers`
  for tests that issue real, billable LLM API calls. The two existing
  `@pytest.mark.live` integration tests now collect cleanly under
  `--strict-markers` (they previously errored at collection time).
- `missing_batch_id_error()` and `unknown_batch_error(batch_id)` helpers
  in `kaos_llm_core.integrations.mcp._batch_helpers` so every batch tool
  reports the same what/how/alternative recovery hint when the input is
  missing or the batch can't be found.

### Changed

- VFS-resolution error messages in
  `kaos_llm_core.integrations.mcp._batch_helpers._resolve_vfs_to_disk`
  and `resolve_output_dir` now include a specific recovery hint
  (configure `VFS_BACKEND=disk`, run through `kaos-llm-core-serve`, or
  fall back to an inline `list` input source) instead of just naming
  the failure.
- `kaos-llm-core-batch-status` and `kaos-llm-core-batch-results` use the
  new shared error helpers, replacing terse single-sentence errors with
  the standard recovery-hint pattern.
- `kaos-llm-core-batch-results` `format` validation now lists the three
  valid values inline so callers don't have to look at the schema.

## [0.1.0a1] — 2026-05-07

First public alpha. Apache-2.0. Earlier internal versions were proprietary.

### Added

- **Signatures** — typed I/O contracts as Pydantic models, with grounding
  primitives (`Span`, `Claim`, `Answer[T]`, `Cited[T]`, `GroundedAnswer[T]`,
  `InsufficientEvidence`) and `ExtractionSchema` for schema-driven extraction.
  First-class multimodal field types (`Image`, `Audio`, `Document`) routed
  through `kaos-llm-client`'s native attachment mechanism.
- **Calls + Programs** — `Call` (single LLM invocation), `Program` (composed),
  `ChainOfThought`, `Judge`, `Ensemble`, `Grounded`, `RAG`, `ReAct`, `Refine`,
  `BestOfN`, `MultiChainComparison`, `ProgramOfThought` (subprocess-sandboxed
  code-as-reasoning, opt-in via `allow_code_execution=True`).
- **Invocation runtime contract** — every Call/Program execution returns an
  `Invocation` (output, trace, usage, error, context, client, model, extras).
  `Call.invoke()` exposes the full bundle; `Call.__call__()` returns just
  the output.
- **Codecs** — `JSONCodec`, `ChatCodec`, `XMLCodec` for bidirectional
  Signature ↔ provider-message translation.
- **Routers** — `CascadeRouter`, `RuleRouter` for multi-model selection.
- **Optimizers** — `BootstrapOptimizer`, `InstructionOptimizer`,
  `HyperparameterOptimizer`, `CodecOptimizer`, `ModelOptimizer`,
  `ReflectiveOptimizer`, `CoOptimizer`, `MiproLiteOptimizer`,
  `MiproV2Optimizer` (full DSPy MIPROv2 port with categorical TPE +
  GroundedInstructionProposer + minibatch + full-eval promotion).
  Cost attribution flows through `TrialRunner` so composite optimizers
  enforce a single `BudgetTracker` cap cumulatively.
- **Cache** — two-tier semantic cache: tier 1 exact-input hash (O(1));
  tier 2 embedding-similarity. Optional crash-safe JSONL disk persistence
  with replay-on-init.
- **Observability** — `ExecutionTrace` (hierarchical), per-model cost
  estimation via `cost.MODEL_PRICING`, JSONL trace export.
- **Batch runner** — `batch_run()` library primitive with streaming JSONL
  log, deterministic content-addressed `custom_id`s, resume contract,
  three error policies, and per-VFS SQLite metadata store (WAL, multi-process
  safe).
- **Higher-level "starter" API** — `text()`, `extract()`, `classify()`,
  `summarize()` (and `_sync` variants) for one-shot scripts and exploration.
- **CLI** — `kaos-llm-core` for direct invocation, `kaos-llm-core-serve`
  for the MCP server (stdio or streamable HTTP).
- **MCP server** — 29 tools exposed via `kaos-llm-core-serve`:
  `kaos-llm-core-call`, `-reason`, `-judge`, `-ensemble`, `-evaluate`,
  `-optimize`, `-cost-report`, `-react`, `-refine`, `-best-of-n`,
  `-save-load`, `-optimize-codec`, `-optimize-model`, `-pareto`,
  `-recipe-tune`, `-metric`, `-analyze-trial`, `-program-execute`,
  `-batch-create`, `-batch-run`, `-batch-status`, `-batch-results`,
  `-mipro-v2`, plus 6 alpha extractors (date, entity, money, number,
  percent, duration) routed through `kaos-nlp-core`.
- **Typed settings** — `KaosLLMCoreSettings` (`ModuleSettings` subclass) with
  `KAOS_LLM_CORE_` env prefix. Honors global `KAOS_PROFILE` as a fallback
  when `KAOS_LLM_CORE_PROFILE` is unset.
- **Error hierarchy** — `KaosLLMCoreError`, `SignatureError`, `CodecError`,
  `CallError`, `ValidationRetryExhaustedError`. All inherit `KaosCoreError`.
  Python 3.13 + 3.14 support.

### Security

- **`ProgramOfThought` requires explicit opt-in** at both construction
  (`allow_code_execution=True`) and envelope validation. The default
  refuses to execute code.
- **Subprocess sandbox** for `ProgramOfThought` uses POSIX rlimits,
  tempdir-cwd isolation, and a wall-clock timeout.
- **All iterative programs hard-cap iterations** — `LoopRunner` enforces
  `max_iterations`; `ReAct`, `Refine`, `BestOfN`, `MultiChainComparison`
  all carry sane defaults.
- **TrialRunner cost cap** — composite optimizers (`CoOptimizer`) inject
  one shared `BudgetTracker` so the cumulative `Budget` cap holds across
  stages, not per stage.
- API keys flow through `kaos-llm-client` as `SecretStr` end-to-end.
- **KLLC-01** — `_resolve_codec_class` (called from `Program.load()` →
  `Call.set_learnable_state()`) now refuses to import any module not
  under `kaos_llm_core.codecs.`, *before* calling
  `importlib.import_module`. A malicious saved-state JSON envelope can
  no longer trigger import side effects of arbitrary modules on
  `PYTHONPATH`. Regression test:
  `tests/unit/test_security_regressions.py::test_kllc_01_*`.
- **KLLC-02** — `_resolve_vfs_to_disk` and `resolve_output_dir` (in
  `integrations/mcp/_batch_helpers.py`) now reject absolute disk paths
  outright. Previously an MCP caller of `kaos-llm-core-batch-create`
  could escape the workspace by passing an absolute path like
  `/etc/cron.d/`; all paths must now flow through
  `runtime.vfs.resolve_disk_path` for VFS containment. Regression test:
  `tests/unit/test_security_regressions.py::test_kllc_02_*`.
- **KLLC-03** — `kaos-llm-core-save-load` (load + round-trip modes) now
  rejects absolute disk paths and routes VFS-relative paths through
  `runtime.vfs.resolve_disk_path`. Closes an information-disclosure
  vector where a caller could probe the filesystem via the JSON-decode
  error message. Regression test:
  `tests/unit/test_security_regressions.py::test_kllc_03_*`.

### Fixed

- **KLLC-05** — `_build_eval_dataset` (used by every MCP tool that
  accepts `examples`) now validates that each list element is a `dict`
  with an agent-friendly `ToolResult.create_error(...)`, instead of
  raising an opaque `TypeError`.
- **KLLC-07** — `ReAct`, `Refine`, `BestOfN`, and `MultiChainComparison`
  now carry hard upper bounds on their iteration / sample count
  (`MAX_ITERATIONS = 1000` / `100`, `MAX_N = 100`). Protects direct
  Python-API consumers from typo-driven runaway API spend
  (e.g. `max_iterations=10000` instead of `100`). Regression test:
  `tests/unit/test_security_regressions.py::test_kllc_07_*`.

### Changed

- **KLLC-04** — `KaosLLMCoreSettings.trace_path` removed. The field
  was never consumed by the observability layer; the docstring promised
  a `~/.cache/kaos/llm-core/traces` default that no code honored. Trace
  export sinks are passed explicitly at the call site
  (see `observability.export`).

### Documentation

- Added `tests/fixtures/cross-doc-corpus/README.md` and
  `tests/fixtures/privilege-sample/README.md` documenting fixture
  provenance (KLLC-06). Both directories contain synthetic content;
  the READMEs make that explicit so casual readers cannot mistake the
  files for real legal documents.

### License

- This release is the first to ship under the Apache License 2.0. Earlier
  internal versions were proprietary.

### Notes

- The `[mcp]`, `[pdf]`, and `[tabular]` extras (cross-module input bridges
  for the batch runner) are not declared in this release because the
  underlying `kaos-mcp`, `kaos-pdf`, and `kaos-tabular` packages are not
  yet on PyPI. They will be re-added in `0.1.0a2` once those siblings ship.
  Until then, install those packages from source if you need the bridges.

[Unreleased]: https://github.com/273v/kaos-llm-core/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-llm-core/releases/tag/v0.1.0a1
