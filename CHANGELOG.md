# Changelog

All notable changes to `kaos-llm-core` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

## [0.1.0a14] — 2026-05-17

### Changed

- **kaos-core floor raised to `>=0.1.0a10`** to pick up the URI
  contract redesign (bare names route through
  `context.default_vfs_namespace`; `file://` and `vfs://` schemes).
  See `kaos-modules/docs/plans/uri-contract-redesign.md`. Pass-through
  for kaos-llm-core internals (no synthetic bare-name resolver calls).

## [0.1.0a13] — 2026-05-17

### Fixed

- **`extract_corpus()` no longer silently returns empty results when
  `output_dir` is missing.** Previously, if the caller passed an
  `output_dir` that did not exist on the local filesystem (most often
  a session-VFS path like `sessions/<sid>/files/output` plumbed through
  by an agent), `batch_run()` would create the directory and write
  `items.jsonl`, but the hydration step's `Path(output_dir).exists()`
  check would still see the path the caller passed and — in the failure
  paths covered by this fix — return a vacuously-empty
  `CorpusExtractionResult` with no error signal. Indistinguishable from
  a corpus that genuinely had no matches. `extract_corpus()` now raises
  the new `ExtractCorpusError` (subclass of `KaosLLMCoreError`) with an
  agent-friendly what / how / alternative message before any LLM call.
  See `kaos-modules/docs/plans/vfs-blind-tools-audit-and-fix-plan.md`
  §5.1. A `TODO(vfs-aware)` marker tracks the optional follow-up to
  accept a `KaosContext` and route via VFS materialisation.

### Added

- **`ExtractCorpusError`** in `kaos_llm_core.programs.extract`. Raised
  by `extract_corpus()` when `output_dir` is missing or not a directory.
  Inherits `KaosLLMCoreError` → `KaosCoreError`, so agent-side tool
  wrappers that already translate `KaosCoreError` to
  `ToolResult.create_error()` pick it up without changes.

## [0.1.0a12] — 2026-05-15

Lands the Phase-8 GLiNER half from
`docs/summarization-classification-plan.md` §4.2.4. The companion
`kaos-nlp-transformers` 0.2.0a7 release ships the actual ONNX-backed
`GLiNERExtractor` whose `extract()` method satisfies the new
`NerExtractor` Protocol at runtime — cross-tested against the real
`onnx-community/gliner_medium-v2.1` model and verified to produce
sigmoid scores matching the upstream PyTorch reference (0.9935 /
0.9772 on the canonical "Barack Obama was born in Hawaii." input).

### Added

- **`GLiNERExtract` Program** in `kaos_llm_core.programs.ner`. The
  zero-shot NER counterpart to `ZeroShotNLIClassifier` — LLM-free,
  deterministic, runs locally through an injected `NerExtractor`
  Protocol implementation. Accepts a `labels` list, a per-Program
  `threshold` (default 0.5) plus `max_width` / `flat_ner` /
  `dup_label` / `multi_label` tuning, and returns `Entities`
  carrying byte-offset `EntitySpan` records sorted by source
  position.
- **`NerExtractor` + `EntityResult` Protocols** in
  `kaos_llm_core.programs.ner`. Both `@runtime_checkable`; the
  canonical implementation lives in
  `kaos_nlp_transformers.GLiNERExtractor` (which ships in
  `kaos-nlp-transformers` 0.2.0a7+).
- **`Entities` and `EntitySpan` result types** in
  `kaos_llm_core.results`. `EntitySpan` is the lightweight
  `(start, end, text, label, score)` record emitted by
  `GLiNERExtract`; `Entities` is the per-text wrapper carrying the
  list, the queried labels, and program/extractor metadata.
- Both new result types re-exported from the top-level
  `kaos_llm_core.__all__`.

## [0.1.0a11] — 2026-05-15

Audit-driven follow-up to the 0.1.0a10 plan release. Re-running the
plan §11 endgame snippet against the actual public API surfaced three
load-bearing gaps; this release closes them and adds a contract-test
guard so future drift triggers a unit-test failure rather than a
plan-doc lie.

### Added

- **`resolve_aggregator`** in `kaos_llm_core.composition`. Maps short
  names (``"vote"``, ``"majority"``, ``"union"``, ``"intersection"``,
  ``"weighted"``, ``"max_score"``) to the matching `Aggregator` class
  and passes through instances unchanged. Raises `CallError` on
  unknown names with the allowed-name list embedded in the message.

- **`classify_doc(aggregator=…)` accepts strings** (audit G1). The
  plan §11 endgame snippet — ``classify(doc, …, aggregator="union",
  …)`` — now runs against the actual API:
  ``classify_doc(doc, …, aggregator="union", …)``.

- **`summarize_doc(query=…, embedder=…)`** (audit G3, plan §7.1).
  When `query` is supplied, `summarize_doc` routes through
  :class:`QueryFocusedSummary` regardless of `long_strategy`. `CallError`
  when `query` is set without `embedder`. CLI / MCP exposure waits
  for an embedder-kind registry (parallel to the tokenizer-registry
  task P2-2); the query route is in-process only at 0.1.0a11.

- **`tests/unit/test_plan_endgame_snippet.py`** — verbatim contract
  test for the plan §11 snippets (classification + no-LLM
  prototype + cited summary). Future drift between the plan's
  promise and the shipped API now produces a test failure.

### Changed

- **`QueryFocusedSummary` accepts `budget=`** (audit G2). The single
  abstractive call is bracketed by a pre-call `tracker.exhausted()`
  gate and a post-call `tracker.consume()` from `Invocation.usage`.
  Exhaustion returns an empty Summary tagged with
  `metadata["partial"] = True` and `metadata["budget.exhausted"]`.

- **`HybridSummary` accepts `budget=`** (audit G2). Same shape as
  the QueryFocusedSummary wiring; the extractive pre-filter is
  free, so only the abstractive stage charges the tracker.

- **Plan §11 endgame snippet** updated to use the actual
  ``classify_doc`` / ``summarize_doc`` names + the new symmetric
  no-LLM path with ``supervision="prototype"``.

### Notes

- ``cache=`` is deliberately NOT exposed on `QueryFocusedSummary` /
  `HybridSummary`: the `ChunkCache` Protocol is chunk-id-keyed for
  the chunked-reducer Programs, and these single-call Programs
  don't chunk. Per-(doc, query) reuse is the caller's
  responsibility for now; revisit if usage patterns demand it.

## [0.1.0a10] — 2026-05-15

Plan-driven release closing `docs/summarization-classification-plan.md`
§8.6 items A, B, C, D, E, F. The plan's pyramid is now fully built on
the `kaos-llm-core` side; only the `kaos-nlp-transformers` NLI
checkpoint registration (plan §8 Phase 8 lower half) remains as a
separate follow-up.

### Added

- **First live quality-harness run** (plan §8.6 item A) against
  `anthropic:claude-haiku-4-5`. Three new artifacts under
  `docs/benchmarks/`:

  - `quality-cuad-grounding.json` — 25 (contract × clause) cells,
    100 % verified-claim rate, 68 % cells with ≥1 verified span, 0
    refusals, 20 % cells with a gold-clause match.
  - `quality-billsum-rouge.json` — 60 cells (20 bills × 3 programs).
    `AbstractiveSummary` ROUGE-1 / 2 / L = 0.48 / 0.21 / 0.31;
    `HierarchicalSummary` 0.46 / 0.20 / 0.28; `MapReduceSummary`
    0.46 / 0.20 / 0.28. 0 errors.
  - `quality-ledgar-f1.json` — 100 clauses, 100-class taxonomy,
    68 classes seen. F1-macro 0.54, F1-micro / accuracy 0.66
    (top of the harness's expected 0.30–0.55 band).

### Fixed

- `tests/quality/test_cuad_grounding.py` referenced
  `summary.refused`, which is not a field on the :class:`Summary`
  Pydantic model — the refusal state lives in
  `metadata["cited.refused"]`. Fixed; the live CUAD harness now
  runs clean.
- `tests/quality/test_billsum_rouge.py` called
  `rouge_score.rouge_scorer.RougeScorer` without explicitly
  importing the `rouge_scorer` submodule. Fixed with a one-line
  `from rouge_score import rouge_scorer` after the
  `importorskip`.

### Added

- **Phase 6 retrieval-augmented Programs** (plan §6.1, §6.2,
  §8.6 item D). Four new Programs in `kaos-llm-core` 0.1.0a10:

  - `kaos_llm_core.programs.summarize.QueryFocusedSummary` —
    segment + embed + cosine-rank (via the SIMD
    `cosine_one_to_many_normalized` fast path) → top-k → routed
    through `CitedSummary` by default. Accepts the same
    `Embedder` protocol as `PrototypeClassify`.

  - `kaos_llm_core.programs.summarize.ClusteredSummary` — long-doc
    summarizer specialising `_LongDocBase` with the new `Cluster`
    reducer. Keeps the cache + budget wiring from §8.5.

  - `kaos_llm_core.programs.summarize.HybridSummary` —
    extractive top-k pre-filter then `CitedSummary` (default) or
    `AbstractiveSummary` over the picks. `method="hybrid"`.

  - `kaos_llm_core.programs.classify.RetrievalClassify` — kNN over
    a labeled corpus + weighted majority vote, optional LLM
    tie-break via a caller-supplied `tie_break` Program. Closes
    plan §11's no-LLM-many-shot endgame.

- **`Cluster` reducer** (`kaos_llm_core.composition.Cluster`,
  plan §5.1 Phase-2 leftover). Spherical k-means in cosine space
  with deterministic seed. `k="auto"` resolves to
  `max(2, min(round(sqrt(n)), 8))`. Required by
  `ClusteredSummary` and exposed for direct use through
  `kaos_llm_core.composition`. The `ClusterEmbedder` protocol
  mirrors the existing `Embedder` shape.

- **Phase 8 zero-shot NLI classifier** (plan §4.2.3, §8 Phase 8,
  §8.6 item F — `kaos-llm-core` half):

  - `kaos_llm_core.programs.classify.ZeroShotNLIClassifier`
    formulates each label as a natural-language hypothesis via
    `hypothesis_template.format(label.prompt_text)` (default
    `"This text is about {}."`) and picks the label maximising
    `P(entailment)`. Optional `min_score` abstention floor.
  - `NLIScorer` / `NLIScore` Protocols mirror the
    `Ranker` / `Embedder` pattern; any object producing three
    probabilities per hypothesis satisfies the contract. The
    canonical `kaos_nlp_transformers.NliModel` is a separate
    follow-up.
  - `classify_doc` now accepts five `supervision` modes
    (`zero_shot`, `few_shot`, `prototype`, `retrieval`, `nli`)
    with per-mode required kwargs (`examples=` / `embedder=` /
    `corpus=` / `nli_scorer=`). Each branch validates its
    requirements with a `CallError`.

- **Phase 7 declarative surfaces** (plan §7.1 + §7.2 + §7.3,
  §8.6 item E) in three layers:

  - `kaos_llm_core.starter.summarize_doc` and
    `kaos_llm_core.starter.classify_doc` (+ `_sync` wrappers) — the
    §7.1 declarative façade. Returns the full `Summary[str]` /
    `Classification` Pydantic objects with `long_strategy="auto"`
    rules (12 000-char threshold, single ↔ tree/chunk),
    `cited=True` routing through `CitedSummary`, plus
    `cache=`/`budget=`/`chunker=` passthrough to the long-doc
    Programs.  The simpler one-shot `summarize` / `classify`
    (plain-string return) are unchanged for back-compat; the new
    `*_doc` functions are the canonical Phase-7 surface.

  - CLI subcommands `kaos-llm-core summarize <file>` and
    `kaos-llm-core classify <file> --labels labels.json`. Read
    from disk or stdin (``-``). Flags: `--strategy`, `--cited`,
    `--supervision`, `--model`, `--budget-tokens`, `--budget-usd`,
    `--pretty`, `--cost`. JSON output by default; `--pretty` for
    human-readable + optional per-result cost line.

  - MCP tools `KaosLLMCoreSummarizeTool` (`kaos-llm-core-summarize`)
    and `KaosLLMCoreClassifyTool` (`kaos-llm-core-classify`)
    wrapping the starter façade. Complement (do not replace) the
    pre-existing `KaosLLMCoreProgramExecuteTool`. Bumps the
    program-tools registration count 24 → 26 and the full bulk
    count 30 → 32. `classify` accepts either a flat list of label
    name strings or a single serialized `LabelSet` in the
    `labels` array (the second form lets agents pass multi-label
    or hierarchical taxonomies through MCP).

- **Chunk-result cache + per-Program budget enforcement** (audit
  P1-7 + P1-8, plan §5.3 + §8.5 + §8.6 item C). Three new
  surfaces in `kaos_llm_core.cache.chunk`:
  - `ChunkCacheKey` — public on-disk key shape
    `(chunk_id, program_name, model_hint)`.
  - `ChunkCache` — async Protocol for caller-supplied
    implementations.
  - `InMemoryChunkCache` — default process-local FIFO with
    bounded `max_entries` (default 10k) and hit/miss counters.

  Wired through `_LongDocBase` (= `HierarchicalSummary` /
  `MapReduceSummary` / `RefineSummary`) and `ChunkedClassify`
  via new `cache: ChunkCache | None` / `budget: Budget | None`
  constructor params. On a cache hit the per-chunk Program
  invocation is skipped; on budget exhaustion processing halts
  and the aggregated result carries `metadata["partial"] = True`
  plus `metadata["budget.exhausted"]`. The aggregated result
  also reports `metadata["cache.hits"]`,
  `metadata["chunks.processed"]`, and (when a budget tracker
  ran) `metadata["budget.cost_usd"]` /
  `metadata["budget.tokens"]`.

  The pre-existing `SemanticCache` (Call-level, embedding-keyed)
  is unchanged and complements `ChunkCache` (Program-level,
  chunk-id-keyed); both can coexist.
- **`ExtractiveSummary` Program** (Phase-5 leftover from
  `docs/summarization-classification-plan.md` §6.1) at
  `kaos_llm_core.programs.summarize.ExtractiveSummary`. Wraps any
  object conforming to the new local ``Ranker`` protocol (canonical
  implementation: ``kaos_nlp_transformers.extraction.ExtractiveRanker``)
  to produce no-LLM extractive summaries. Forward signature:
  ``await program(text, query=None, top_k=None, parent_id=None)``;
  returns a ``Summary[str]`` with ``method="extractive"``, picks
  re-ordered by source offset for narrative readability, and
  per-pick ``rank``/``score`` preserved in
  ``metadata["picks"]``. Zero LLM cost — the Program holds no
  ``Call`` children and ``Program.invoke`` builds a childless trace
  with zero token usage.
- **`PrototypeClassify` Program** (Phase-5 leftover from plan §6.2)
  at `kaos_llm_core.programs.classify.PrototypeClassify`. Embeds the
  input plus each ``Label.prompt_text`` via the supplied ``Embedder``
  (protocol matching ``kaos_nlp_transformers.EmbeddingModel.embed``),
  scores via the SIMD-dispatched
  ``kaos_nlp_core.similarity.cosine_one_to_many_normalized`` fast
  path, and applies an argmax (exclusive ``LabelSet``) or threshold
  rule (multi-label ``LabelSet``). Label prototypes are embedded
  once and cached. Optional ``min_score`` floor abstains when the
  top cosine falls below the threshold and the LabelSet permits
  abstention. Combines with ``ChunkedClassify`` and
  ``UnionAggregator`` to produce a fully no-LLM multi-label
  long-document classifier; see plan §11 endgame example.
- **Per-Program cost / latency bench** at `tests/bench_programs.py`.
  Distills the 6 ``docs/benchmarks/live-*.json`` snapshots captured
  by ``tests/scale/test_programs_live.py`` into a single
  ``docs/benchmarks/programs-cost-latency.json`` with per-Program
  ms/doc, $/doc, and tokens/sec. No new live LLM calls -- the
  bench reshapes existing snapshots so downstream consumers
  (kelvin-training, kaos-compliance) have a single source of
  truth for what each Program costs at the captured model.
- **CUAD span-verification harness** at
  `tests/quality/test_cuad_grounding.py` (audit task P1-4).
  Runs `CitedSummary` over the vendored 5-contract x 5-clause CUAD
  sample (CC-BY-4.0; sourced from
  `kaos-nlp-core/tests/fixtures/cuad-sample/`) and reports the
  per-cell verified-span count, the fraction of cells where any
  verified span contains the gold answer, and the overall
  verified-claim rate. Gated on ``KAOS_LLM_LIVE_PROVIDER``;
  emits ``docs/benchmarks/quality-cuad-grounding.json``.
- **BillSum ROUGE harness** at
  `tests/quality/test_billsum_rouge.py` (audit task P1-2). Runs
  ``AbstractiveSummary`` / ``HierarchicalSummary`` /
  ``MapReduceSummary`` over the BillSum test split (Apache-2.0;
  downloaded at runtime via HuggingFace ``datasets``) and reports
  ROUGE-1 / ROUGE-2 / ROUGE-L F-scores per program + aggregate
  mean. Skips cleanly when ``datasets`` / ``rouge-score`` aren't
  installed. Emits ``docs/benchmarks/quality-billsum-rouge.json``.
- **LEDGAR F1 harness** at `tests/quality/test_ledgar_f1.py`
  (audit task P1-3). Runs ``ZeroShotClassify`` over a stratified
  slice of the LEDGAR test split (CC-BY-NC-SA-4.0; **not**
  vendored, downloaded at runtime) with a 100-class LabelSet and
  reports F1-macro / F1-micro / accuracy + per-class precision /
  recall / support. Skips when ``datasets`` / ``scikit-learn``
  aren't installed. Emits ``docs/benchmarks/quality-ledgar-f1.json``.
  License note: the harness output is evaluation metadata only,
  not a redistribution of the underlying dataset.

### Documentation

- ty exclude list extended for the two harnesses that runtime-import
  optional libraries (``datasets``, ``rouge_score``, ``scikit-learn``)
  via ``pytest.importorskip``; ty resolves modules statically so
  these need the same exclude treatment the integration tests use.



## [0.1.0a9] — 2026-05-15

### Added — granular MCP-tool registration entry points (PRD PR 1)

- **`register_llm_core_program_tools(runtime)`** — registers the
  24 typed-program / optimizer / codec / batch / metric wrappers
  (Call, ChainOfThought, ReAct, Refine, Judge, Ensemble, Evaluate,
  Optimize, OptimizeCodec, OptimizeModel, Pareto, RecipeTune,
  CostReport, BestOfN, SaveLoad, Metric, AnalyzeTrial,
  ProgramExecute, ProgramOfThought, BatchCreate, BatchRun,
  BatchStatus, BatchResults, MiproV2).
- **`register_llm_core_alpha_tools(runtime)`** — registers only
  the 6 deterministic ``kaos-llm-core-alpha-*`` rule-based
  extractors (date, duration, entity, money, number, percent).
- **`register_llm_core_tools(runtime)`** is now a backward-compatible
  union of the two — every existing caller continues to see the
  same 30 tools with the same names and schemas.

These granular entry points let kaos-agents (PR 2) wire the
SessionToolSet ``programs`` group to either or both subsets
independently — a session can opt into the cheap rule-based
extractors without exposing the full optimizer / batch surface,
or vice versa. Motivated by
`kaos-modules/docs/internal/dynamic-tool-planning-prd.md` §4
("PR 1 — catalog expansion"). Purely additive: no tool name,
schema, or behavior changes.


## [0.1.0a8] — 2026-05-15

### Documentation

- **Use, data-handling, and AI-authorship disclosure** added to the
  README and to the ``programs/summarize/__init__.py`` /
  ``programs/classify/__init__.py`` module docstrings. The new
  language: (1) statistical-approximation warning + "not legal/
  financial/medical advice without expert review"; (2) explicit note
  that every Program transmits text to the configured LLM provider
  and that privileged / PHI / customer-confidential data needs an
  enterprise DPA; (3) AI-assisted authorship disclosure (Claude,
  Anthropic; human-reviewed).

### Fixed

- **Cost telemetry no longer silently reports `$0.00` for unknown models.**
  Previously ``apply_cost_estimates`` / ``estimate_cost`` looked up
  ``trace.model`` in ``PRICING`` with a plain ``dict.get`` and fell
  through to ``cost_usd=0`` when the model was missing — with no log,
  no warning, no test signal. Every live-scale benchmark JSON under
  ``docs/benchmarks/live-*.json`` reported ``$0.00`` despite real
  provider calls, because the installed ``kaos-llm-client`` lockfile
  lagged the model rate card and ``gpt-5.4-nano`` (and friends) were
  not in the pricing table. The fix is two parts:
  1. The lockfile is refreshed so the current rate card lands. After
     refresh, ``gpt-5.4-nano`` resolves correctly to ``$0.10/$0.40``
     per M tokens.
  2. ``apply_cost_estimates`` / ``estimate_cost`` now emit a
     one-shot warning naming the unknown model when a leaf trace
     has ``input_tokens > 0`` and no pricing entry — making the
     silent-zero failure visible. The warning is rate-limited via
     a module-level ``_warned_unknown_models`` set so long batches
     don't flood logs. The ``(program)`` placeholder and
     zero-token traces are explicitly excluded.
  - New regression test suite ``tests/unit/test_cost_unknown_model.py``
    asserts the warned-set side-effect (the canonical signal —
    pytest's ``caplog`` doesn't capture the ``kaos`` logger hierarchy
    by default). Includes a defensive "lockfile health check" that
    asserts at least one current-gen cheap model (``gpt-5.4-nano`` /
    ``gemini-2.5-flash`` / ``claude-haiku-4-5``) is in the installed
    pricing table.

### Security / Faithfulness

- **`CitedSummary` now verifies every claim against the source at
  runtime.** Previously the Program populated
  ``Summary.source_spans`` directly from the LLM's
  ``supporting_spans`` without checking whether the quoted substring
  actually appeared in the source — a hallucinated quote was
  indistinguishable from a real one. ``CitedSummary.forward`` now
  calls :meth:`Answer.verify` (already in
  ``kaos_llm_core.signatures.grounding``) and includes only spans
  from fully-verified claims in ``source_spans``. The full
  ``GroundedAnswer`` payload is preserved unmodified on
  ``Summary.payload`` so callers can inspect what failed.
  - New constructor parameters: ``verify_strategies`` (tuple of
    :class:`MatchStrategy`, default ``DEFAULT_MATCH_STRATEGIES`` =
    ``STRICT`` + ``SUBSTRING``), ``verify_threshold`` (float, default
    ``0.9``, for fuzzy strategies), ``refuse_below`` (float in
    ``[0.0, 1.0]``, default ``0.0``).
  - When the verified-claim fraction is strictly below
    ``refuse_below``, the summary text and ``source_spans`` are
    collapsed to empty and ``metadata["cited.refused"]`` is set to
    ``True``. Default ``0.0`` preserves prior behavior (never
    refuse) while making the gate explicit.
  - New metadata fields:
    ``cited.verified_claim_count``,
    ``cited.unverified_claim_count``,
    ``cited.verified_ratio``,
    ``cited.refused``,
    ``cited.verify_strategies``,
    ``cited.error_reasons``.

### Added

- **`kaos_llm_core.labels`** — new module defining the canonical label
  space type for classification programs. Phase 0 of the
  summarization/classification cross-module plan; concrete classifier
  programs follow in Phase 4.
  - `Label` — Pydantic model with ``name``, ``description``,
    ``examples``, optional ``parent`` for hierarchies, and a
    ``prompt_text`` convenience property.
  - `LabelSet` — Pydantic model wrapping ``labels`` plus the
    ``exclusive`` / ``allow_abstain`` / ``hierarchical`` policy flags.
    Container protocol (``__iter__`` / ``__len__`` / ``__contains__``),
    ``names`` / ``by_name`` / ``children`` / ``roots`` accessors, and
    ``validate_picks`` / ``assert_picks`` helpers. Construction-time
    validation rejects duplicate names, the reserved ``__abstain__``
    name, unknown hierarchical parents, and parent-graph cycles.
  - ``LabelSet.from_names(...)`` for the flag-free flat-set case.
  - `ABSTAIN_LABEL` — reserved sentinel name (``"__abstain__"``).
- **`kaos_llm_core.results`** — new module defining the canonical
  result containers.
  - `Summary[T]` — generic Pydantic model with ``text``, optional
    typed ``payload``, ``method`` tag (``abstractive`` / ``extractive``
    / ``hybrid``), ``depth``, ``chunks_used``, ``source_spans``, and
    free-form ``metadata``.
  - `Classification[L]` — generic Pydantic model parameterized over
    ``Label | str`` carrying ``labels``, optional ``scores``,
    ``abstained``, optional ``rationale``, plus the same provenance
    fields. ``names`` / ``top_label`` convenience accessors.
  - `SourceSpan` — lightweight positional ``(start, end, source_id)``
    reference. Complements the rich
    :class:`~kaos_llm_core.signatures.grounding.Span` used inside
    ``Cited[T]`` payloads.
  - `SummaryMethod` — exported ``Literal`` for the method tag.
- All seven new names (``ABSTAIN_LABEL``, ``Label``, ``LabelSet``,
  ``Summary``, ``SummaryMethod``, ``Classification``, ``SourceSpan``)
  are re-exported from the top-level ``kaos_llm_core`` package and
  listed in ``__all__``.

- **`kaos_llm_core.composition`** — new composition layer for
  summarization and classification programs. Phase 2 of the cross-module
  plan.
  - **Aggregator strategies** (single source of truth: pure
    aggregation functions in
    ``kaos_nlp_core.aggregation``):
    - `Aggregator` — runtime-checkable Protocol.
    - `VoteAggregator`, `MajorityAggregator(threshold=…)` — exclusive
      strategies.
    - `UnionAggregator`, `IntersectionAggregator` — multi-label
      strategies.
    - `WeightedAggregator(threshold, chunk_weight=…)` — single or
      multi-label, honors per-chunk weights/scores.
    - `MaxScoreAggregator(threshold=…)` — pools per-chunk score maps
      by max.
    - All strategies operate on
      :class:`~kaos_llm_core.results.Classification` instances plus
      a :class:`~kaos_llm_core.labels.LabelSet`, pool provenance
      (chunks_used/source_spans), and emit a label histogram in the
      returned metadata.
  - **Reducer strategies** (LLM-free orchestration; caller supplies
    an async ``merge_fn``):
    - `Reducer` — runtime-checkable Protocol with async
      ``reduce(leaves, merge_fn)``.
    - `MapReduce` — single merge call across all leaves.
    - `Refine` (in ``composition.reduce`` — not at the top level
      to avoid colliding with the existing :class:`Refine` Program).
    - `Tree(branching=4, max_depth=8)` — k-ary bottom-up merge with
      concurrent sibling merges and tail-singleton folding.
  - All composition classes are re-exported from
    ``kaos_llm_core.composition`` and (with the exception of
    ``Refine``) from the top-level ``kaos_llm_core`` package.

- **`kaos_llm_core.programs.summarize`** — new summarization Programs.
  Phase 3 of the cross-module plan. Each Program returns a
  :class:`Summary` (or :class:`Summary[T]` for the typed payload
  variant); routing, caching, batching, retries, observability, and
  cost are inherited from the base
  :class:`~kaos_llm_core.programs.call.Call` /
  :class:`~kaos_llm_core.programs.base.Program` infrastructure.
  - **Single-doc**:
    - `AbstractiveSummary` — single LLM call, plain ``str`` payload.
    - `StructuredSummary(schema=…)` — builds a Signature at runtime
      around the caller's Pydantic schema; the LLM produces both a
      plain-text summary and a typed structured payload.
    - `CitedSummary` — emits a
      :class:`~kaos_llm_core.signatures.grounding.GroundedAnswer`
      payload; pools supporting spans into
      :class:`Summary.source_spans` for downstream verification.
    - `AbstractiveSummarySignature` — public Signature class for the
      plain abstractive shape; reusable by callers wiring their own
      Call instances or optimizing prompts.
  - **Long-doc** (consume any
    :class:`kaos_nlp_core.chunking.Chunker`; default
    :class:`ParagraphChunker(max_tokens=1024)`):
    - `MapReduceSummary` — parallel per-chunk leaves +
      :class:`~kaos_llm_core.composition.MapReduce` reduce.
    - `RefineSummary` — sequential left-to-right
      :class:`~kaos_llm_core.composition.reduce.Refine` reduce.
    - `HierarchicalSummary(branching=4, max_depth=8)` — k-ary
      :class:`~kaos_llm_core.composition.Tree` reduce.
  - All seven names re-exported from the top-level ``kaos_llm_core``
    package and listed in ``__all__``.
  - Tests use the offline :class:`FunctionClient` stub provider to
    keep the unit gate deterministic; live integration coverage is
    expected via downstream callers.

- **`kaos_llm_core.programs.classify`** — new classification Programs.
  Phase 4 of the cross-module plan. Every Program returns a
  :class:`Classification[Label]` regardless of whether the underlying
  decision is from one LLM, a chunked vote, a hierarchical walk, or
  an ensemble.
  - `ZeroShotClassify(labels=…)` — single-label LLM classification.
    Constructs a Signature at Program-init with a Literal-typed
    ``label`` output field so providers that support constrained
    decoding push the constraint to the wire format.
  - `FewShotClassify(labels=…, examples=…)` — :class:`ZeroShotClassify`
    that requires a non-empty :class:`Example` pool.
  - `MultiLabelClassify(labels=…)` — emits a *subset* of label names
    plus per-label confidence. Requires
    ``LabelSet.exclusive=False``; preserves canonical label order in
    the result.
  - `HierarchicalClassify(labels=…)` — coarse-to-fine walk over a
    ``hierarchical=True``
    :class:`LabelSet`. Returns the leaf-most label plus the full
    path in ``metadata["hierarchical.path"]``.
  - `ChunkedClassify(labels=…, per_chunk=…, chunker=…, aggregator=…)`
    — long-document wrapper: chunks the input via any
    :class:`kaos_nlp_core.chunking.Chunker`, runs the per-chunk
    classifier in parallel, and combines results via any
    :class:`~kaos_llm_core.composition.Aggregator` (defaults:
    :class:`MajorityAggregator` for exclusive sets,
    :class:`UnionAggregator` for multi-label).
  - `EnsembleClassify(labels=…, members=…, aggregator=…)` — runs
    multiple member classifiers concurrently over the same input and
    combines them via an aggregator. Children are exposed via a
    ``named_calls`` override so the trace tree captures every
    member.
  - All six names re-exported from the top-level ``kaos_llm_core``
    package and listed in ``__all__``.

- **`kaos-llm-core-program-of-thought` MCP tool** (#91, Rec #3). Dedicated
  wrapper around `ProgramOfThought` that exposes code-as-reasoning over
  the MCP wire without forcing the agent to construct a full Program v3
  envelope. Writer LLM emits Python, a subprocess sandbox runs it
  (POSIX rlimits + tempdir cwd + wall-clock timeout), interpreter LLM
  parses captured stdout into a typed answer. Returns the answer plus
  the generated code and raw stdout/stderr for audit. Tool count
  bumps from 29 → 30.
- **Two-gate safety model.** Code execution is refused unless **both**
  the caller passes `allow_code_execution=true` per request **and**
  the server-side `KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP=1`
  environment variable is set. Either gate off and the tool errors
  before any subprocess spawns. Both default to off — `DANGEROUS BY
  DEFAULT`.
- `KaosLLMCoreSettings.allow_code_execution_via_mcp: bool = False`
  (env: `KAOS_LLM_CORE_ALLOW_CODE_EXECUTION_VIA_MCP`) — the runtime
  half of the gate.

### Mirrored from monorepo

This release mirrors the monorepo implementation of #91 (Rec #3)
landed on `273v/kaos-modules` and applies it onto the per-module
source-of-truth ahead of the next per-module release.

## [0.1.0a7] — 2026-05-11

### Changed

- **Cross-process env recorder (``KAOS_LLM_CORE_RECORDER_DIR``) now
  streams records to disk as JSONL with ``fsync()`` per line** instead
  of buffering and writing at exit. A header line is written at
  ``install_from_env()`` time so a downstream reader can identify the
  file before any ``Invocation`` runs. Adds SIGTERM / truncation
  tolerance — the recorder reader skips a corrupted final line
  without raising. Closes Sprint-3 #8 ("audit trail must survive
  ``SIGTERM``").
- **Header schema bumped from v3 → v4.** New header fields:
  - ``streaming: True`` — advertises the streaming policy
  - ``redaction_enabled: True`` — advertises the redaction policy
  - ``redaction_threshold_chars: 2048`` — body-length threshold above
    which a recorded ``output`` is redacted to a length-only summary

### Added

- **Transparency-lens redaction of long outputs** (KC16-4). Invocation
  records with ``len(output_text) > 2048`` now serialize as
  ``{"output": {"_redacted": True, "len_chars": <n>}}`` rather than
  the raw text. Threshold is overridable at recorder-install time via
  ``KAOS_LLM_CORE_RECORDER_REDACT_THRESHOLD_CHARS``; disable entirely
  with ``KAOS_LLM_CORE_RECORDER_REDACT_OUTPUTS=0``. Avoids audit JSONL
  ballooning to MB-scale per test and avoids leaking long model
  outputs into long-lived audit storage.
- **``tests/unit/test_env_recorder.py``** — header/schema-v4
  assertions (``streaming is True``, ``schema_version == 4``,
  ``redaction_enabled is True``, ``redaction_threshold_chars == 2048``)
  plus round-trip + redaction coverage.
- **``tests/unit/test_env_recorder_durability.py``** — SIGTERM /
  truncated-final-line / partial-write durability tests that lock in
  the streaming contract.

### Mirrored from monorepo

This release mirrors monorepo commits ``b8f5998`` (Sprint-3 #8 —
streaming recorder + per-line ``fsync``) and ``33a7c1a`` (KC16-4 —
redact long strings + schema-v4) from ``kaos-modules``. Per
``memory/feedback_per_module_split_mirror.md``, monorepo source edits
to published packages must be mirrored back into the per-module repo
before downstream siblings can depend on them. Required dependency
floor for kaos-agents v0.1.0a1 — its unit tests assert the v4 header
shape against this package.

## [0.1.0a6] — 2026-05-11

### Fixed

- **``observability.cost.PRICING`` now mirrors ``kaos-llm-client``'s
  authoritative ``MODEL_PRICING`` table.** Closes cross-sibling
  divergence flagged by kaos-agents PA15 (Gap #2 + Gap #5) / KC16-2
  (CRITICAL · FIX_BEFORE_TAG). The table was missing ten model rows
  that the client knew about, including ``gpt-5.5`` (``$5.00`` /
  ``$30.00`` per MTok) — the highest-cost OpenAI model currently
  shipped, which silently rolled up to ``$0.00`` on every
  ``ExecutionTrace`` that used it. ``o3`` rate corrected from the
  (10.0, 40.0) reasoning-premium tier to the (2.0, 8.0) standard
  tier the client dispatches against. Newly priced bare names (and
  their ``provider:model`` aliases): ``gpt-5.5``, ``gpt-4.1``,
  ``gpt-4.1-mini``, ``gpt-4.1-nano``, ``gpt-4o``, ``gpt-4o-mini``,
  ``claude-opus-4-7``, ``claude-sonnet-4-5``, ``grok-3``,
  ``grok-3-mini``.

### Added

- **Cross-sibling pricing-parity regression test**
  (``tests/unit/test_cost_pricing_parity.py``). Closes KC16-11
  ("Pricing-table parity untested across siblings"). Asserts every
  bare model in ``kaos-llm-client.cost.MODEL_PRICING`` exists in
  ``kaos-llm-core.observability.cost.PRICING`` with matching
  ``(input, output)`` rates. One-directional (client → core) — core
  may legitimately track preview-tier and future models the client
  doesn't yet price. Failure mode is loud and points the reader at
  the authoritative source.

## [0.1.0a5] — 2026-05-11

### Added

- **``HybridRubric`` + ``apply_rubric`` row-scoring stack.** New
  :class:`~kaos_llm_core.signatures.rubric.HybridRubric` value type
  carries typed ``Criterion`` rules (column / operator / value /
  weight, 11-operator enum) alongside a free-form
  ``qualitative_guidance`` channel. ``hard_weight`` + ``soft_weight``
  combine the two; constructor enforces sum-to-1.0 and rejects fully-
  empty rubrics.
  :func:`~kaos_llm_core.programs.scoring.apply_rubric` consumes the
  rubric + a ``TabularDocument`` and returns a new doc with ``_score``
  (FLOAT, [0, 1]) and ``_reasoning`` (TEXT) columns appended; hard
  channel is pure Python, soft channel is one ``Call`` per row
  concurrent via ``asyncio.gather``. Critically uses a dedicated
  ``RowJudgmentSignature`` rather than ``Judge.score_response`` —
  Judge is wired for producer/judge pairs and collapses row scores to
  ~0.10. 33 unit tests + smoke-verified live (Sonnet 4.6 produces
  ordered scores 0.94 / 0.87 / 0.36 on the synthetic 3-row table).
  Public API: ``Criterion``, ``CriterionOperator``, ``HybridRubric``
  on ``kaos_llm_core.signatures``; ``apply_rubric``,
  ``RowJudgmentSignature`` on ``kaos_llm_core.programs``. Files:
  ``kaos_llm_core/signatures/rubric.py``,
  ``kaos_llm_core/programs/scoring.py``,
  ``tests/unit/test_rubric_scoring.py``.

- **``SchemaDesigner`` + ``RubricDesigner`` + ``DealReviewProgram``.**
  Reference implementation of the "agent as runtime program
  synthesizer" architecture: five-phase pipeline ``design_schema`` →
  ``extract`` → ``design_rubric`` → ``apply_rubric`` → recommend.
  Each phase produces a typed artifact (``ExtractionSchema``,
  ``TabularDocument``, ``HybridRubric``, scored ``TabularDocument``).
  Designers are single ``Call`` invocations — sub-agent prototypes
  observed 9 of 10 columns identical across runs of the same
  question, so synthesis is stable enough without refinement
  wrappers.
  :class:`~kaos_llm_core.programs.deal_review.DealReviewProgram`
  orchestrates all five phases as a ``Program`` subclass (not an
  envelope) — the 2026-05-10 prototype comparison measured Program at
  2x smaller LOC, 54% cheaper per run, and able to call existing
  ``Extract`` / loop over corpora / pass dynamic schema between
  phases, none of which envelopes currently support. Live-validated
  on a 2-NDA mini-corpus at Sonnet 4.6 (7 LLM calls, ~$0.08): correctly
  endorses Acme (Michigan-governed, mutual NDA, no non-solicit) and
  declines EMNA (Delaware-governed). Public API:
  ``DealReviewProgram``, ``DealReviewResult``,
  ``SchemaDesignerSignature``, ``RubricDesignerSignature``,
  ``design_schema``, ``design_rubric``, ``sample_corpus_text`` on
  ``kaos_llm_core.programs``. Files:
  ``kaos_llm_core/programs/designers.py``,
  ``kaos_llm_core/programs/deal_review.py``.

- **``NeedsAggregation`` typed result + ``RefusalPolicy.allow_aggregation_route`` flag.**
  Distinguishes "the corpus genuinely lacks the facts"
  (``InsufficientEvidence``) from "the corpus has the facts but the
  answer must be derived" (``NeedsAggregation``). Empirically
  confirmed that every model tested — haiku-4-5, sonnet-4-6,
  gpt-5.4-mini, gpt-5.5 — collapses aggregation queries (longest /
  shortest / most / average / count) to ``InsufficientEvidence`` when
  there's no single span that IS the answer. The refusal route is
  architectural — model strength doesn't fix it. The new
  ``NeedsAggregation`` type carries ``operation`` (max / min / count /
  …), ``relevant_values`` (the retrieved values the agent identified
  as inputs to the aggregation), and ``suggested_tool`` (pointing at
  ``kaos-content-stats`` / ``kaos-retrieval-corpus-manifest`` /
  ``kaos-tabular-top-k`` / ``kaos-llm-core-compute``). Backward-
  compatible: ``RefusalPolicy.allow_aggregation_route`` defaults
  to ``False`` so existing programs collapse aggregation gaps to
  ``InsufficientEvidence`` as today. Recommended ``True`` for
  legal / financial / comparative review workflows. Public API
  additions are exported from ``kaos_llm_core.signatures``. Files:
  ``kaos_llm_core/signatures/grounding.py``,
  ``kaos_llm_core/signatures/__init__.py``.

### Fixed

- **``ExtractionResult.cost_usd`` + ``tokens_total`` populated from the
  inner Call.** Pre-fix, ``Extract`` declared both fields with default
  ``0.0``/``0`` but never assigned ``Call.invoke().usage.cost_usd`` /
  ``usage.total_tokens`` on success. Any downstream code doing budget
  enforcement on ``ExtractionResult`` silently under-counted extraction
  spend. Now populated on the non-grounded path; the grounded path
  remains zero (with an inline TODO) because ``GroundedResult`` doesn't
  expose ``usage`` yet — no regression vs prior behaviour. Smoke-
  verified: ``ExtractionResult.cost_usd = 0.000262``, ``tokens_total =
  255`` on a single-column extract at Haiku 4.5 (was 0/0 pre-fix).
  Files: ``kaos_llm_core/programs/extract.py``.

## [0.1.0a4] — 2026-05-11

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
### Security

- **bandit + vulture now run in both pre-commit and CI.** The
  ``.pre-commit-config.yaml`` gains two new hooks (bandit static
  security scan + vulture dead-code scan), mirrored by jobs in
  ``security.yml`` so the scan is publicly visible on every PR.
  Bandit skip list is justified inline per audit
  (``B101,B404,B603,B607``); vulture runs at ``--min-confidence
  100`` with a shared ``--ignore-names`` list for framework
  callbacks / signal handlers / OAuth field names that vulture
  can't infer from the import graph alone. Both hooks currently
  pass clean. Mirrors the rollout pattern from kaos-core.
### Changed

- **uv.lock is now tracked in git.** Previously gitignored at v0.1.0a1
  because the ``[mcp]`` optional extra (and the ``kaos-mcp`` dev
  dependency) referenced a sibling not yet on PyPI; ``uv lock``
  couldn't resolve them. ``kaos-mcp`` shipped (0.1.0a2), so the
  original gating reason no longer applies. Tracking the lockfile
  gives reproducible local dev environments, lets Dependabot surface
  sibling-version bumps as PRs, and makes the supply-chain pin set
  publicly auditable. Mirrors the org-wide convention being adopted
  across all 16 kaos-* repos.

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

[Unreleased]: https://github.com/273v/kaos-llm-core/compare/v0.1.0a5...HEAD
[0.1.0a5]: https://github.com/273v/kaos-llm-core/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/273v/kaos-llm-core/compare/v0.1.0a3...v0.1.0a4
[0.1.0a1]: https://github.com/273v/kaos-llm-core/releases/tag/v0.1.0a1
