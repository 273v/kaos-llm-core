# Summarization & Classification: Cross-Module Plan

**Status:** Phases 0–5 shipped to PyPI (with two named Phase-5 deliverables
still pending); Phases 6–8 + cross-cutting integration pending. Audit
follow-ups Q1–Q6 closed. See [§1.1 Status snapshot](#11-status-snapshot--2026-05-15) for the
truthful per-phase breakdown.
**Author:** design discussion 2026-05-14
**Status last updated:** 2026-05-15
**Scope:** `kaos-nlp-core`, `kaos-nlp-transformers`, `kaos-llm-core` —
library-level work in those three repositories. Downstream consumers
(kelvin-* products, kaos-compliance, etc.) are **out of scope**; this
plan only ships capabilities, not integrations.
**Goal:** a layered, composable stack that turns the summarization and
classification taxonomies into reusable Programs without reinventing
primitives in every caller.

---

## 0. Design principles

1. **Verbs at the bottom, nouns at the top.** Each layer exposes atomic
   operations (segment, pack, embed, score, call, ground, batch,
   aggregate, reduce). Higher layers are compositions, not new logic.
2. **Char offsets are sacred.** Every span produced by any layer carries
   `(start, end)` back to the original source. Provenance never gets
   regenerated, only forwarded.
3. **Deterministic before neural before generative.** If a step can be
   done with regex/segmentation/BM25, do it there. If it needs
   semantics but not generation, use embeddings. Reserve LLM calls for
   the irreducible part.
4. **One package owns each primitive.** No primitive is reimplemented
   in two places. Bridges (which depend on two layers) live in the
   higher of the two.
5. **Public types are frozen dataclasses or Pydantic.** No tuples, no
   dicts, no kwargs-by-convention.
6. **Caching, routing, batching, observability, grounding are
   horizontal services**, not features of individual Programs. A new
   Program inherits them by composing existing `Call`/`Program` nodes.

---

## 1. The pyramid

```
                                  ┌─────────────────────────────┐
            Layer 5: Surfaces     │ starter API · CLI · MCP     │
                                  └─────────────────────────────┘
                                  ┌─────────────────────────────┐
            Layer 4: Programs     │ Summarize · Classify        │
                                  │ (10 + 8 named compositions) │
                                  └─────────────────────────────┘
                                  ┌─────────────────────────────┐
            Layer 3: Composers    │ Chunker · Reducer · Aggreg. │
                                  │ Router · BatchRun · Cache   │
                                  └─────────────────────────────┘
                                  ┌─────────────────────────────┐
            Layer 2: Neural       │ Embed · Rerank · NLI*       │
                                  │ SemanticChunker · ExtRanker │
                                  └─────────────────────────────┘
                                  ┌─────────────────────────────┐
            Layer 1: Deterministic│ Segment · Tokenize · Search │
                                  │ Enum/Section · CTPH · Qual. │
                                  └─────────────────────────────┘
                                  ┌─────────────────────────────┐
            Layer 0: Types        │ Span · Segment · Chunk      │
                                  │ Label · LabelSet · Summary  │
                                  └─────────────────────────────┘
```

`*` = future; requires a registered NLI model in `kaos-nlp-transformers`.

---

## 1.1 Status snapshot — 2026-05-15

A truthful per-phase ledger. This section is **descriptive** — it
captures what exists today. The design sections that follow (§2–§7)
remain the canonical specification; where reality diverges from the
spec, the divergence is called out here.

### Shipped on PyPI

| Package | Version | What it carries from this plan |
|---|---|---|
| `kaos-nlp-core` | **0.1.0a6** | Layer-0 `Chunk`, Layer-1 deterministic chunkers (5), label aggregation primitives (6), dense-vector similarity kernels (NumKong-design SIMD ported into Rust at `rust/core/similarity/kernels.rs`; AVX-512F / AVX2+FMA / NEON / scalar with runtime ISA dispatch; pre-normalised fast-path variants for unit-norm inputs). |
| `kaos-nlp-transformers` | **0.2.0a6** | Layer-2 `SemanticChunker` + `ExtractiveRanker`, both wired to the pre-normalised SIMD fast paths. End-to-end throughput benches under `tests/bench_*.py`. |
| `kaos-llm-core` | **0.1.0a9** (Q3–Q6 dev tooling on `main` for next release) | Layer-0 `Label` / `LabelSet` / `Summary[T]` / `Classification[L]` / `SourceSpan`. Layer-3 `Reducer` protocol + 3 of 4 reducers + `Aggregator` protocol + 6 strategies. Layer-4 summarization Programs (6 of 10) + classification Programs (6 of 8). MCP registration split into `register_llm_core_program_tools` + `register_llm_core_alpha_tools` (a9; PRD §4 PR-1) — orthogonal to this plan but worth noting because §7.3 talks about MCP. |

### Per-phase ledger

| Phase | Plan section | Shipped | Pending |
|---|---|---|---|
| **0 — Foundation types** | §2 | ✅ all 5 types | — |
| **1 — Deterministic chunkers** | §3 | ✅ 5 chunkers + aggregators | — |
| **2 — Composers** | §5 | ✅ `Reducer` + `MapReduce` / `Refine` / `Tree`; `Aggregator` + 6 strategies | ❌ `Cluster` reducer (one of four named in §5.1) |
| **3 — Summarization Programs** | §6.1 | ✅ `AbstractiveSummary`, `StructuredSummary`, `CitedSummary`, `MapReduceSummary`, `RefineSummary`, `HierarchicalSummary` | ❌ `ExtractiveSummary` (Phase-5 deliverable — see below), `QueryFocusedSummary` / `ClusteredSummary` / `HybridSummary` (Phase 6) |
| **4 — Classification Programs** | §6.2 | ✅ `ZeroShotClassify`, `FewShotClassify`, `MultiLabelClassify`, `ChunkedClassify`, `EnsembleClassify`, `HierarchicalClassify` | ❌ `PrototypeClassify` (Phase-5 deliverable — see below), `RetrievalClassify` (Phase 6) |
| **5 — Neural primitives** | §4 | ✅ `SemanticChunker`, `ExtractiveRanker` | ❌ `ExtractiveSummary` + `PrototypeClassify` in `kaos-llm-core` (these are the LLM-free wrappers that delegate to nlp-transformers; the plan ships them under Phase 5, but they were missed during execution) |
| **6 — Retrieval-augmented variants** | §8 Phase 6 | — | ❌ All four: `QueryFocusedSummary`, `ClusteredSummary`, `HybridSummary`, `RetrievalClassify`. Existing `RAG` Program + `kaos-content` `SearchableDocument` / `SearchableCorpus` are the foundations. |
| **7 — Surfaces** | §7 | ⚠️ partial. `starter.summarize_sync` / `classify_sync` exist as thin sync wrappers but **not** the declarative façade in §7.1 (no `long_strategy="auto"` rules, no `query=` route, no `cited=`). 30 MCP program tools exist (`KaosLLMCoreCallTool`, `KaosLLMCoreJudgeTool`, etc.) including a generic `KaosLLMCoreProgramExecuteTool` that can call any Program, but the **specific declarative `summarize` and `classify` MCP tools** in §7.3 are not registered. CLI has `check` / `examples` / `analyze`; no `summarize` / `classify` subcommands yet. | ❌ Declarative starter façade upgrade, CLI subcommands, dedicated MCP tool pair. |
| **8 — Zero-shot NLI** | §8 Phase 8 | — | ❌ Deferred per the original plan. NLI model registration + `ZeroShotNLIClassifier` not built. |

### Cross-cutting (plan §5.3) — built-but-not-wired

These primitives **exist** in `kaos-llm-core` but are **not yet
threaded through `Program.invoke`**, so callers who construct a
Program don't get the benefit by default:

| Primitive | Lives at | Wired into `Program.invoke`? |
|---|---|---|
| `Budget` / `BudgetTracker` | `kaos_llm_core.optimization.budget` | ❌ no — `Program.invoke` in `programs/base.py` doesn't reference `budget` or `Budget` |
| `SemanticCache` | `kaos_llm_core.cache.semantic` | ❌ no — long-doc Programs (`HierarchicalSummary`, `MapReduceSummary`) don't accept or consult a cache |

The integration work is tracked as P1-7 (cache) and P1-8 (budget) in
the audit follow-up list. Without them, the §6.1 promise that "each
long-doc Program respects the passed `Budget`" is **aspirational, not
implemented**.

### Audit-driven follow-ups (Q1–Q6) — closed 2026-05-15

These weren't in the original plan but emerged from a multi-agent
review on 2026-05-15 ("alpha shipped, untested in anger"):

| ID | Deliverable | Where |
|---|---|---|
| Q1 | Refresh similarity benches post-numkong-pivot → triggered a full NumKong-design SIMD port | kaos-nlp-core 0.1.0a6 |
| Q1b | Wire `SemanticChunker` + `ExtractiveRanker` to the new `*_normalized` fast paths | kaos-nlp-transformers 0.2.0a6 |
| Q2 | End-to-end SemanticChunker + ExtractiveRanker throughput benches | `kaos-nlp-transformers/tests/bench_*.py` |
| Q3 | Per-Program cost / latency consolidator (no live LLM cost; reshapes existing `live-*.json` snapshots) | `kaos-llm-core/tests/bench_programs.py` |
| Q4 | BillSum ROUGE harness for `AbstractiveSummary` / `HierarchicalSummary` / `MapReduceSummary` | `kaos-llm-core/tests/quality/test_billsum_rouge.py` |
| Q5 | LEDGAR F1 harness for `ZeroShotClassify` (100-class) | `kaos-llm-core/tests/quality/test_ledgar_f1.py` |
| Q6 | CUAD span-verification harness for `CitedSummary` | `kaos-llm-core/tests/quality/test_cuad_grounding.py` |

The Q4 / Q5 / Q6 harnesses are `@pytest.mark.live`-gated dev tooling
on `main`; running them produces `docs/benchmarks/quality-*.json`
artifacts. They have **not been fired against a real provider yet**;
doing so is task A in §8.6.

---

## 2. Layer 0 — Foundation types

These types are the wire format between layers. Each lives in exactly
one package; others import.

| Type | Owner | Status | Purpose |
|---|---|---|---|
| `Span(start, end)` | `kaos-nlp-core` | exists | Half-open char range. |
| `Segment(text, start, end, confidence)` | `kaos-nlp-core` | exists | Sentence/paragraph/line output. |
| `TokenSpan(text, start, end)` | `kaos-nlp-core` | exists | Word-level token with offsets. |
| `Chunk` | `kaos-nlp-core` | **NEW** | Packed unit for downstream processing. |
| `Label` | `kaos-llm-core` | **NEW** | One classification target. |
| `LabelSet` | `kaos-llm-core` | **NEW** | Closed/open set of labels with policy flags. |
| `Summary[T]` | `kaos-llm-core` | **NEW** | Generic summary container; `T` is the schema. |
| `Classification[L]` | `kaos-llm-core` | **NEW** | Hard label, distribution, or multi-label result. |
| `Cited[T]` | `kaos-llm-core` | exists | Reused for cited summaries/classifications. |

### 2.1 `Chunk`

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    text: str                    # the chunk's text (verbatim slice of source)
    start: int                   # char offset in source document
    end: int                     # char offset (exclusive)
    parent_id: str | None        # source document id (or chunk id when recursive)
    chunk_id: str                # blake3 over (parent_id, start, end, text) — stable
    token_count: int             # approximate (tokenizer of choice)
    depth: int = 0               # 0 = leaf over source; n = produced by reducer at level n
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

`chunk_id` is deterministic — used directly as a cache key for any
Program that processes this chunk.

### 2.2 `Label` / `LabelSet`

```python
@dataclass(frozen=True, slots=True)
class Label:
    name: str
    description: str | None = None
    examples: tuple[str, ...] = ()
    parent: str | None = None        # for hierarchical taxonomies

class LabelSet:
    labels: tuple[Label, ...]
    exclusive: bool                  # single-label vs multi-label
    allow_abstain: bool              # "none of the above" is legal
    hierarchical: bool               # taxonomy walk vs flat
```

`LabelSet` is the single source of truth across `ZeroShotClassify`,
`PrototypeClassify`, `RetrievalClassify`, and downstream aggregators.

### 2.3 `Summary[T]`

```python
@dataclass(frozen=True, slots=True)
class Summary(Generic[T]):
    text: str                          # the natural-language summary
    schema: T | None                   # optional structured payload
    source_spans: tuple[Span, ...]     # citations back to the source
    chunks_used: tuple[str, ...]       # chunk_ids consumed
    method: Literal["abstractive", "extractive", "hybrid"]
    depth: int                         # reducer depth (0 = direct over source)
```

### 2.4 `Classification[L]`

```python
@dataclass(frozen=True, slots=True)
class Classification(Generic[L]):
    labels: tuple[L, ...]              # 0..1 if exclusive, 0..n if multi-label
    scores: Mapping[str, float] | None # confidence per candidate label
    abstained: bool
    rationale: str | None              # optional LLM rationale
    source_spans: tuple[Span, ...]     # provenance
```

---

## 3. Layer 1 — Deterministic NLP primitives (`kaos-nlp-core`)

### 3.1 What already exists

- `segment_sentences`, `segment_paragraphs`, `segment_lines` — char-offset safe.
- `tokenize`, `tokenize_words`, `tokenize_regex`.
- `parse_enumerator` — markdown / decimal / Roman / chapter / article.
- `search_sentences`, `search_paragraphs`, `Searcher` (BM25 / TF-IDF).
- `ctph_hash_str`, `MinHasher`, `find_duplicates`.
- `quality.score_text` and friends.
- `extract.alpha.*` — dates, money, percent, duration.

### 3.2 What to add

#### 3.2.1 Chunker protocol & implementations

New module `kaos_nlp_core.chunking`:

```python
class Chunker(Protocol):
    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]: ...

class FixedTokenChunker(Chunker):
    def __init__(self, max_tokens: int, overlap: int = 0, tokenizer="default"): ...

class SentenceChunker(Chunker):
    """Pack whole sentences up to a token budget; never split a sentence."""

class ParagraphChunker(Chunker):
    """Pack whole paragraphs; fall back to SentenceChunker for oversize paragraphs."""

class SectionChunker(Chunker):
    """Use parse_enumerator to detect headings; chunk per section, sub-chunk if needed."""

class HierarchicalChunker(Chunker):
    """Returns chunks with `depth` set, preserving doc → section → paragraph hierarchy."""
```

**Guarantees** (enforced by tests):
- Concatenation of `chunk.text` for adjacent non-overlapping chunks
  reconstructs the source minus only normalized whitespace gaps.
- Every chunk's `(start, end)` round-trips: `source[start:end] == chunk.text`.
- Token budgets are advisory ceilings; chunks may be smaller; chunks
  must never exceed the ceiling by more than the smallest indivisible
  unit (one sentence).
- Chunkers are deterministic: same input → identical `chunk_id`s.

#### 3.2.2 Deterministic label aggregators

New module `kaos_nlp_core.aggregation` (deterministic primitives only):

```python
def vote(labels: Iterable[Iterable[str]]) -> str: ...
def majority(labels: Iterable[Iterable[str]], threshold: float = 0.5) -> str | None: ...
def union(labels: Iterable[Iterable[str]]) -> frozenset[str]: ...
def intersection(labels: Iterable[Iterable[str]]) -> frozenset[str]: ...
def weighted(
    labels: Iterable[Iterable[str]],
    weights: Iterable[float],
    *,
    multi: bool,
) -> frozenset[str] | str: ...
```

These are pure functions, no LLM dependency. `kaos-llm-core` wraps them
into an `Aggregator` strategy class at Layer 3.

### 3.3 What stays out of nlp-core

- LLM calls.
- Embedding generation (lives in nlp-transformers).
- Reducer trees (cross-Layer; lives in llm-core).
- Anything that needs a model download.

---

## 4. Layer 2 — Neural primitives (`kaos-nlp-transformers`)

### 4.1 What already exists

- `EmbeddingModel.load(...).embed(texts)` — BGE / model2vec backends.
- `CrossEncoderReranker.load(...).rerank(query, results)` — bge-reranker-base.
- Device detection, registry with pinned SHAs, offline scope.

### 4.2 What to add

#### 4.2.1 `SemanticChunker`

New: `kaos_nlp_transformers.chunking.SemanticChunker(Chunker)`.

- Segment with `ParagraphChunker` as candidate boundaries.
- Embed adjacent paragraph windows; merge across boundaries where
  cosine similarity exceeds threshold; split where it drops.
- Cap by `max_tokens`. Determinism comes from the embedding model
  revision pin.

Implements the same `Chunker` protocol defined in `kaos-nlp-core`, so
it slots into the same Programs as the deterministic chunkers.

#### 4.2.2 `ExtractiveRanker`

New: `kaos_nlp_transformers.extraction.ExtractiveRanker`.

```python
class ExtractiveRanker:
    def rank(
        self,
        sentences: Sequence[Segment],
        *,
        query: str | None = None,        # query-focused if provided
        top_k: int | None = None,
        diversify: float = 0.0,           # MMR lambda
    ) -> list[ScoredSegment]: ...
```

- If `query` given: cross-encoder over (query, sentence) pairs.
- Else: centroid similarity (each sentence vs document centroid) for
  generic extractive summary.
- MMR for diversity when requested.

This is the no-LLM extractive summarizer. `ExtractiveSummary` (Layer 4)
just wraps it with `Summary[T]` packaging.

#### 4.2.3 Future: `ZeroShotNLI` (deferred)

Add an NLI head (e.g., `cross-encoder/nli-deberta-v3-base` or
equivalent) to the registry. Wrap as:

```python
class ZeroShotNLIClassifier:
    def classify(
        self,
        text: str,
        labels: LabelSet,
        *,
        hypothesis_template: str = "This text is about {}.",
    ) -> Classification[Label]: ...
```

Gates: license audit, revision pin, single-language note. Until this
ships, transformer-backed classification = `PrototypeClassify` only.

### 4.3 What stays out

- Generative models (no T5/BART/Pegasus). Abstractive summarization
  goes through `kaos-llm-client` providers.
- Fine-tuning. Inference-only.

---

## 5. Layer 3 — Composers (`kaos-llm-core`)

These are the building blocks that turn primitives into Programs. They
are not user-facing on their own; they're constructor arguments to
Layer 4 Programs.

### 5.1 `Reducer` protocol

New module `kaos_llm_core.composition.reduce`:

```python
class Reducer(Protocol):
    async def reduce(
        self,
        leaves: Sequence[Summary[T]],
        *,
        leaf_program: Program,
        node_program: Program,
        root_program: Program | None = None,
    ) -> Summary[T]: ...

class MapReduce(Reducer): ...           # one map pass + one reduce call
class Refine(Reducer): ...              # sequential running summary
class Tree(Reducer):                    # k-ary bottom-up
    def __init__(self, branching: int = 4, max_depth: int = 4): ...
class Cluster(Reducer):                 # embed leaves, k clusters, summarize each
    def __init__(self, k: int | Literal["auto"] = "auto"): ...
```

All Reducers:
- Use `batch_run` for parallelizable levels (Map, Tree internal).
- Emit nested `ExecutionTrace` matching the reducer tree.
- Carry source spans through every level.
- Respect a `Budget` if passed.

### 5.2 `Aggregator` strategy

New module `kaos_llm_core.composition.aggregate`:

```python
class Aggregator(Protocol):
    def combine(
        self,
        per_chunk: Sequence[Classification[L]],
        labels: LabelSet,
    ) -> Classification[L]: ...

class VoteAggregator(Aggregator): ...
class MajorityAggregator(Aggregator): ...
class UnionAggregator(Aggregator): ...           # multi-label only
class IntersectionAggregator(Aggregator): ...    # multi-label only
class WeightedAggregator(Aggregator): ...        # uses per-chunk scores
class MaxScoreAggregator(Aggregator): ...
```

Implementation delegates to `kaos_nlp_core.aggregation` for the pure
counting math.

### 5.3 Cross-cutting services (reused, not new)

- `Router` / `CascadeRouter` — model selection per level.
- `batch_run` — parallel leaves.
- `SemanticCache` keyed on `chunk_id` — chunk summaries reused across
  document edits (CTPH gives us "this is the same chunk" cheaply).
- `Budget` / `BudgetTracker` — cost ceilings end-to-end.
- `Cited[T]` / `Grounded` / `validate_cited_output` — faithfulness.
- `Judge` / `BestOfN` — quality gates on top of any output.
- `ExecutionTrace` — automatic, mirrors the program graph.

### 5.4 Acceptance criteria for Layer 3

- Reducers can be unit-tested with a mock `Program` that returns deterministic
  strings; no LLM required.
- Aggregators are pure functions of inputs; 100% branch coverage.
- A reducer over N leaves runs leaf work in parallel via `batch_run`
  with no manual asyncio gather.

---

## 6. Layer 4 — Programs (`kaos-llm-core`)

### 6.1 Summarization programs (`kaos_llm_core.programs.summarize`)

| Program | Method | Long-doc | Faithfulness | Status (2026-05-15) | Notes |
|---|---|---|---|---|---|
| `AbstractiveSummary` | abstractive | no | free | ✅ shipped | Single-shot, optional schema. |
| `ExtractiveSummary` | extractive | no | extractive-only | ❌ pending (Phase-5 leftover) | Wraps `ExtractiveRanker`; **no LLM call** in pure-extractive mode. |
| `CitedSummary` | abstractive | no | cited | ✅ shipped | Abstractive + `Grounded`. Verifies spans via `grounded.verify()`; unverified spans dropped; optional `refuse_below` threshold (audit P0-1). |
| `StructuredSummary[T]` | abstractive | no | free or cited | ✅ shipped | Pydantic output schema (`T`). |
| `MapReduceSummary` | abstractive | yes | free | ✅ shipped | Chunk → per-chunk summary → reduce. |
| `RefineSummary` | abstractive | yes | free | ✅ shipped | Sequential running summary. |
| `HierarchicalSummary` | abstractive | yes | free or cited | ✅ shipped | Tree reducer; the general workhorse. |
| `QueryFocusedSummary` | abstractive | yes | cited | ❌ pending (Phase 6) | BM25 + dense rerank → summarize top-k. Builds on existing `RAG` + `kaos-content.SearchableDocument`. |
| `ClusteredSummary` | abstractive | yes | free | ❌ pending (Phase 6) | Cluster reducer (good for multi-doc). Also gated on the missing `Cluster` reducer in `composition/reduce.py`. |
| `HybridSummary` | hybrid | yes | cited | ❌ pending (Phase 6) | Extractive top-k → abstractive over those. Depends on the missing `ExtractiveSummary`. |

Each takes:
- `chunker: Chunker | None` (long-doc programs only)
- `router: Router | None`
- `cache: Cache | None`
- `cited: bool = False` (where applicable)
- `output_schema: type[BaseModel] | None`
- `budget: Budget | None`

### 6.2 Classification programs (`kaos_llm_core.programs.classify`)

| Program | Supervision | Backend | Status (2026-05-15) | Notes |
|---|---|---|---|---|
| `ZeroShotClassify` | zero-shot | LLM | ✅ shipped | Schema-constrained decode against `LabelSet`. |
| `FewShotClassify` | few-shot | LLM | ✅ shipped | Adds `Example` pool to the prompt. Subclass of `ZeroShotClassify`. |
| `PrototypeClassify` | zero-shot | embedding | ❌ pending (Phase-5 leftover) | Cosine vs label-description embeddings. **No LLM.** Plan §4.2.2 ships this alongside `ExtractiveRanker`; the program wrapper in `kaos-llm-core` was missed. |
| `RetrievalClassify` | many-shot | embedding + LLM | ❌ pending (Phase 6) | kNN over labeled corpus + vote, optionally LLM tie-break. Builds on existing `RAG` + `SearchableCorpus`. |
| `HierarchicalClassify` | any | nested classifiers | ✅ shipped | Coarse-to-fine taxonomy walk. |
| `MultiLabelClassify` | zero/few-shot | LLM | ✅ shipped | Emits subset with per-label confidence. |
| `ChunkedClassify` | any | wrapper | ✅ shipped | Chunk → per-chunk classify → aggregator. |
| `EnsembleClassify` | any | wrapper | ✅ shipped | Multiple member classifiers + aggregator. |

`ChunkedClassify(per_chunk=PrototypeClassify(...), aggregator=UnionAggregator())`
gives a fully no-LLM multi-label long-doc classifier — fast, cheap,
deterministic.

### 6.3 Acceptance criteria for Layer 4

- Each Program has a `Signature` with explicit input/output fields.
- Each Program subclass implements `async forward(...)` and nothing else.
- Each Program returns `Invocation` with a populated trace.
- Each long-doc Program accepts any `Chunker` from Layer 1 or Layer 2.
- Each long-doc Program respects the passed `Budget`.

---

## 7. Layer 5 — Surfaces

### 7.1 Starter API (`kaos_llm_core.starter`)

Upgrade existing `summarize()` and `classify()` into declarative
façades that pick the right Layer 4 Program:

```python
def summarize(
    doc: str | Document,
    *,
    method: Literal["abstractive", "extractive", "hybrid"] = "abstractive",
    long_strategy: Literal["auto", "single", "map_reduce", "refine", "tree", "cluster", "retrieve"] = "auto",
    chunker: Chunker | Literal["sentence", "paragraph", "section", "token", "semantic"] = "paragraph",
    cited: bool = False,
    schema: type[BaseModel] | None = None,
    query: str | None = None,             # turns it into QueryFocusedSummary
    router: Router | str | None = None,
    budget: Budget | None = None,
) -> Summary: ...
```

```python
def classify(
    doc: str | Document,
    labels: LabelSet | Sequence[str],
    *,
    supervision: Literal["zero_shot", "few_shot", "retrieval", "prototype"] = "zero_shot",
    examples: Sequence[Example] | None = None,
    long_strategy: Literal["auto", "chunk", "summarize_first"] = "auto",
    chunker: Chunker | str = "paragraph",
    aggregator: Aggregator | str = "majority",
    cited: bool = False,
    router: Router | str | None = None,
    budget: Budget | None = None,
) -> Classification: ...
```

`long_strategy="auto"` rules:
- If estimated token count ≤ 70% of model context: `single`.
- Else, document has detected section structure: `tree` with `SectionChunker`.
- Else, `query` given: `retrieve`.
- Else: `tree` with `ParagraphChunker`.

These rules live in one function with tests; not scattered across
Programs.

### 7.2 CLI

`kaos-llm-core summarize <file>` and `kaos-llm-core classify <file>
--labels labels.json` expose the starter API. JSON output by default,
human-readable with `--pretty`. Cost report on `--cost`.

### 7.3 MCP tools

Register two new tools in `kaos_llm_core.integrations.mcp`:

- `summarize` — wraps the starter `summarize()`.
- `classify` — wraps the starter `classify()`.

Both emit structured `Cited[T]` JSON when `cited=true`.

---

## 8. Phasing

Each phase is independently shippable and adds visible value.

### Phase 0 — Foundation types (1 PR per package) — ✅ shipped

- `kaos-nlp-core`: `Chunk` dataclass + `Chunker` protocol stub.
- `kaos-llm-core`: `Label`, `LabelSet`, `Summary[T]`, `Classification[L]`,
  `SourceSpan`.
- Exported from `__all__` of each package.
- Tests cover round-trip frozen-dataclass behavior, hashability, JSON
  serialization.

**Delivered:** types exist; downstream Phases 1–4 consume them.

### Phase 1 — Deterministic chunkers (`kaos-nlp-core`) — ✅ shipped

- `FixedTokenChunker`, `SentenceChunker`, `ParagraphChunker`,
  `SectionChunker`, `HierarchicalChunker`.
- Property tests + scale tests over USC (68,759 docs) / EDGAR (200) /
  patents (200).
- Rust packer kernel (`pack_units`) shared across all five.
- Benchmarks in `docs/benchmarks/chunker-pack-rust-vs-python.json`
  and `chunker-scale-*.json`.

**Delivered:** chunking works end-to-end with no neural or LLM
dependency.

### Phase 2 — Composers (`kaos-llm-core`) — ⚠️ 3 of 4 reducers shipped

- ✅ `Reducer` protocol + `MapReduce`, `Refine`, `Tree`.
- ❌ `Cluster` reducer — see audit task P3-bundle-D.
- ✅ `Aggregator` protocol + 6 strategies (`Vote`, `Majority`, `Union`,
  `Intersection`, `Weighted`, `MaxScore`). Note: plan originally said
  "five strategies"; we shipped six.
- Backed by mock-Program tests; no live LLM in CI.

**Deliverable status:** primitives ready *except* `Cluster` (blocks
`ClusteredSummary` in Phase 6).

### Phase 3 — Summarization Programs (`kaos-llm-core`) — ✅ shipped

- `AbstractiveSummary`, `StructuredSummary`, `CitedSummary` (single-doc).
- `MapReduceSummary`, `RefineSummary`, `HierarchicalSummary` (long-doc).
- Tests use deterministic stub providers for unit; live integration
  tests behind `@pytest.mark.live` (results captured at
  `docs/benchmarks/live-*.json`).
- **Audit P0-1 fix shipped:** `CitedSummary` verifies spans via
  `grounded.verify()` before populating `Summary.source_spans`;
  optional `refuse_below` threshold.

**Delivered:** long-doc summarization end-to-end with cited variants.

### Phase 4 — Classification Programs (`kaos-llm-core`) — ✅ shipped

- `ZeroShotClassify`, `FewShotClassify`, `MultiLabelClassify`,
  `HierarchicalClassify`, `ChunkedClassify`, `EnsembleClassify`.
- **Audit P0-2 fix shipped:** aggregation set-iteration non-determinism
  removed (`dict.fromkeys` everywhere — no hash-randomization leak).
- **Audit P0-3 fix shipped:** cost telemetry refreshed `MODEL_PRICING`
  + one-shot warning on unknown models.

**Delivered:** long-doc classification end-to-end, LLM-only.

### Phase 5 — Neural primitives (`kaos-nlp-transformers`) — ⚠️ 2 of 4 deliverables shipped

- ✅ `SemanticChunker` (`kaos-nlp-transformers`) implementing the
  `Chunker` protocol. Boundary scan runs in Rust
  (`semantic_pack`); adjacent-pair cosine routes through
  `kaos_nlp_core.similarity.cosine_adjacent_normalized` (the
  pre-normalised SIMD fast path).
- ✅ `ExtractiveRanker` (`kaos-nlp-transformers`): centroid + query
  + MMR diversification, routed through the SIMD fast paths.
- ❌ **MISSED:** `ExtractiveSummary` Program in `kaos-llm-core`.
  The plan ships this as a thin wrapper around `ExtractiveRanker`
  with no LLM call. The program class was not built; the
  `programs/summarize/` module has `abstractive.py`, `long_doc.py`,
  `structured.py` only.
- ❌ **MISSED:** `PrototypeClassify` Program in `kaos-llm-core`.
  Same shape — wraps embedding cosine vs label-description
  embeddings with no LLM call. Not built.

**Delivered:** the neural Rust-backed work; the LLM-free
*program-level* wrappers (`ExtractiveSummary`, `PrototypeClassify`)
that the plan promised under this phase need a follow-up commit in
`kaos-llm-core`.

### Phase 6 — Retrieval-augmented variants — ❌ pending

- `QueryFocusedSummary`, `ClusteredSummary`, `HybridSummary`,
  `RetrievalClassify`. Uses the existing `RAG` Program in
  `kaos_llm_core.programs.rag` (shipped) and `kaos-content`'s
  `SearchableDocument` / `SearchableCorpus` (shipped on PyPI as
  `kaos-content`).

**Deliverable:** query-driven and retrieval-driven paths.

**Detailed scope** (deferred from this section to §8.6 below so the
forward plan lives in one place).

### Phase 7 — Surfaces — ⚠️ partial; not the declarative façade

- Current state of `kaos_llm_core.starter`: `text` /
  `extract` / `classify` / `summarize` (async) + `*_sync` wrappers.
  These exist but are thin sync wrappers around `Call` / single
  Programs; **not** the declarative façade in §7.1 with
  `long_strategy="auto"` rules, `query=` routing to
  `QueryFocusedSummary`, or `cited=True` routing to `CitedSummary`.
- Current MCP: `register_llm_core_program_tools` registers 24
  Program tools (Call / ChainOfThought / Judge / Ensemble /
  Evaluate / Optimize / CostReport / ReAct / Refine / BestOfN /
  SaveLoad / OptimizeCodec / OptimizeModel / Pareto / RecipeTune /
  Metric / AnalyzeTrial / ProgramExecute / ProgramOfThought / 4
  Batch tools / MiproV2) + 6 alpha tools. The §7.3 declarative
  `summarize` and `classify` MCP tools (wrapping the façade) are
  **not** among them — but `KaosLLMCoreProgramExecuteTool` does
  let an MCP caller invoke any registered Program by name.
- Current CLI (`kaos-llm-core`): `check`, `examples`, `analyze`
  subcommands. No `summarize` / `classify` subcommands.

**Deliverable status:** the *Program-level* MCP surface is shipped
(callers can already drive `AbstractiveSummary` /
`HierarchicalSummary` / etc. through `ProgramExecuteTool`). What's
*missing* is the **declarative one-liner**: the auto-strategy starter
façade, the CLI subcommands, and dedicated `summarize` /
`classify` MCP tools that wrap the façade.

### Phase 8 — Future: zero-shot NLI — ❌ deferred

- Register an NLI model in `kaos-nlp-transformers` (license audit +
  SHA pin).
- Implement `ZeroShotNLIClassifier`.
- Add `supervision="nli"` route to the starter classify façade.

**Deliverable:** GPU-free, no-LLM zero-shot classification path.

---

## 8.5 Cross-cutting integration (P1-7, P1-8) — gates Phase 6 / Phase 7

Two cross-cutting primitives are built but unwired (see §1.1
"Cross-cutting" sub-table). These should be threaded through
`Program.invoke` before Phase 6 / Phase 7 ships — otherwise the
declarative starter advertises `budget=` / `cache=` parameters that
silently do nothing.

### P1-7 — Wire `SemanticCache` into long-doc Programs

- **What:** `MapReduceSummary`, `HierarchicalSummary`, `RefineSummary`,
  `ChunkedClassify`, `EnsembleClassify` accept a `cache: SemanticCache |
  None = None` constructor argument. Inside `forward()`, per-chunk
  LLM calls are keyed by `(chunk.chunk_id, program_class.__name__,
  signature_hash, model)`; cache-hit short-circuits the call.
- **Why:** without this, re-summarising the same document re-burns
  the entire LLM cost. Real production cost driver.
- **Surface:** purely additive — existing constructors retain their
  signatures with `cache=None` as a no-op.
- **Tests:** new `tests/unit/test_cache_wiring.py` asserts cache-hit
  on a second invocation of an identical request, with a mock
  Program that counts call invocations.

### P1-8 — Wire `BudgetTracker` into `Program.invoke`

- **What:** `Program.invoke` accepts a `budget: Budget | None = None`
  kwarg; if non-None, opens a `BudgetTracker(budget=budget)` for the
  call, propagates it to `KaosContext`, and `Call._execute` checks
  it before each provider call. On exhaustion, raises
  `BudgetExceeded` with the partial trace.
- **Why:** without this, a runaway long-doc Program has no cost
  ceiling. The declarative starter advertises `budget=` already
  (see §7.1) — it must do something.
- **Surface:** new kwarg on `Program.invoke` and on the §7.1
  starter API. Backward-compatible (`None` = no enforcement).
- **Tests:** new `tests/unit/test_budget_wiring.py` asserts a
  Program with a 1-cent budget refuses to dispatch a 2-cent call,
  with a stub provider that pre-declares its cost.

These two are explicitly **library plumbing**, not consumer
integrations. Once wired, every existing Program automatically
respects the budget / cache without source-level changes.

---

## 8.6 Forward plan (2026-05-15)

This is the canonical forward roadmap. Items are ordered by
prerequisite, not by perceived priority — each is independently
shippable once its predecessors land.

### A — Run the Q4 / Q5 / Q6 quality harnesses against a live provider

- **What:** with `KAOS_LLM_LIVE_PROVIDER=anthropic` (or equivalent),
  fire `tests/quality/test_billsum_rouge.py`,
  `tests/quality/test_ledgar_f1.py`, `tests/quality/test_cuad_grounding.py`.
  Commit the resulting `docs/benchmarks/quality-*.json` artifacts.
- **Why:** the harnesses are dev tooling on `main` (PR #26) but
  have never been fired. Without the artifacts we have no
  public-benchmark evidence the Programs work; the audit-driven
  "alpha shipped, untested in anger" gap remains open.
- **Cost:** ~$2–5 in LLM spend.
- **Owner:** any contributor with provider credentials.
- **Output:** three new JSON artifacts under
  `kaos-llm-core/docs/benchmarks/`.

### B — Phase-5 leftovers (`ExtractiveSummary` + `PrototypeClassify`)

- **What:** add the two missing Programs in `kaos-llm-core`. Both
  are pure-Python wrappers; neither makes an LLM call. They
  delegate to `kaos_nlp_transformers.ExtractiveRanker` and the
  embedding model, respectively.
- **`ExtractiveSummary`:** segment → embed → rank via centroid (or
  optional query) → emit top-k `Segment`s as a `Summary[str]` whose
  `text` is the joined picks. Acceptance criteria same as Phase 3.
- **`PrototypeClassify`:** embed input + each label description →
  cosine via `cosine_one_to_many_normalized` → pick argmax (single)
  or threshold (multi). No LLM. Acceptance criteria same as Phase 4.
- **Why now:** plan §6.1 / §6.2 list these as Layer-4 Programs;
  they're the no-LLM paths the plan's "what success looks like"
  section explicitly relies on; also blocks `HybridSummary` in
  Phase 6.
- **Surface change:** additive; bump `kaos-llm-core` to
  `0.1.0a10` (next available alpha after the in-flight 0.1.0a9).

### C — P1-7 + P1-8 wiring (see §8.5)

- One PR per concern; or both in one if the diffs stay small.
- Bump `kaos-llm-core` once they land.

### D — Phase 6 retrieval-augmented variants

- **`QueryFocusedSummary`:** input = `(text, query)`. Build (or
  accept) a `SearchableDocument` from `kaos-content`; retrieve the
  top-k passages matching `query` (BM25 by default, optional dense
  rerank using `kaos_nlp_core.similarity.cosine_one_to_many_normalized`);
  summarise the joined passages via `CitedSummary` so spans tie
  back to the source. Acceptance criteria: ≥80% of returned spans
  must verify against the source via `grounded.verify()`.
- **`ClusteredSummary`:** requires `Cluster` reducer first (Phase-2
  leftover). Embed chunks, cluster by k-means (small numpy
  implementation per plan §10 decision), per-cluster summarise,
  emit `Summary[str]` per cluster + an over-all rollup.
- **`HybridSummary`:** extractive top-k via `ExtractiveSummary`
  (depends on B), then abstractive over the picks via
  `CitedSummary`. The "what" of `HybridSummary` is "rank first,
  then summarise the survivors so the cost is bounded by the
  pick-rate, not the doc length".
- **`RetrievalClassify`:** input = `(text, labeled_corpus)`. Embed
  text + every labeled example, retrieve k-nearest by cosine, take
  majority vote weighted by similarity. Optional LLM tie-break for
  ties or low-confidence. Builds on `RAG` + `SearchableCorpus`.

All four ship as **new Program classes in `kaos-llm-core`**;
nothing touches `kaos-nlp-core` or `kaos-nlp-transformers`. Live
tests gated on `@pytest.mark.live`; unit tests use stub retrievers
+ stub providers.

### E — Phase 7 surfaces

Three deliverables, all in `kaos-llm-core`:

1. **Declarative starter** (`kaos_llm_core.starter.summarize` /
   `classify`): upgrade the async functions (already exist) to the
   §7.1 signature with `long_strategy="auto"`, `query=`, `cited=`,
   `budget=`. Implement the auto-strategy rules in one
   `_resolve_long_strategy()` function with its own tests. `*_sync`
   wrappers update to delegate to the new façade.
2. **CLI subcommands** in `kaos_llm_core/cli.py`: add
   `kaos-llm-core summarize <file>` and `kaos-llm-core classify
   <file> --labels labels.json` per §7.2. JSON output by default,
   `--pretty` / `--cost` flags.
3. **Dedicated MCP tools** in
   `kaos_llm_core/integrations/mcp/`: register
   `KaosLLMCoreSummarizeTool` and `KaosLLMCoreClassifyTool` that
   wrap the starter façade. These complement (do not replace) the
   existing `KaosLLMCoreProgramExecuteTool` which can already drive
   any Program by name.

Live tests behind `@pytest.mark.live`; example scripts under
`examples/`.

### F — Phase 8 zero-shot NLI

- Register an NLI model in `kaos-nlp-transformers` (license-audit
  the chosen checkpoint, pin the SHA, document the pin in
  `models/` provenance docs).
- Build `ZeroShotNLIClassifier` in
  `kaos_llm_core.programs.classify` that calls the registered NLI
  model via a new tiny `kaos_nlp_transformers.nli` surface.
- Add `supervision="nli"` to the starter `classify` façade.

Independently shippable; deferred until a no-LLM-budget use case
asks for it.

---

## 9. Module-by-module change summary

### `kaos-nlp-core`

- **NEW** `chunking/` — `Chunk`, `Chunker` protocol, 5 implementations.
- **NEW** `aggregation/` — pure label aggregation functions.
- No breaking changes; alpha-version bump.

### `kaos-nlp-transformers`

- **NEW** `chunking.py` — `SemanticChunker`.
- **NEW** `extraction.py` — `ExtractiveRanker`.
- **FUTURE** registry entry for NLI model + `ZeroShotNLIClassifier`.
- No breaking changes; alpha-version bump.

### `kaos-llm-core`

- **NEW** `types_labels.py` — `Label`, `LabelSet`.
- **NEW** `types_results.py` — `Summary[T]`, `Classification[L]`.
- **NEW** `composition/` — `Reducer`, `Aggregator` strategies.
- **NEW** `programs/summarize/` — 10 programs above.
- **NEW** `programs/classify/` — 8 programs above.
- **UPDATED** `starter.py` — declarative `summarize()` / `classify()`.
- **NEW** CLI subcommands + MCP tools.
- Bump `__all__`; document in README.

---

## 10. Decisions and open questions

### Decisions

- **Chunk packing is deterministic and lives in `kaos-nlp-core`.** Even
  the semantic variant returns `Chunk` objects that conform to the
  same protocol.
- **Aggregators are pure functions, wrapped in strategy classes only
  at the LLM layer.** This keeps `kaos-nlp-core` LLM-free.
- **`Summary` and `Classification` are generic on payload type**, so
  schema-driven summarization and typed classification reuse the same
  containers.
- **`chunk_id` doubles as a cache key.** Don't introduce a separate
  cache-key abstraction.
- **No new abstract base class for "Skill" or "Pipeline."** Programs
  already are that. Resist the urge to add another layer.

### Open questions

1. **Tokenizer for chunking budgets** — use a single provider-agnostic
   tokenizer (tiktoken-compatible) or pick per-call based on the
   downstream model? Recommend single deterministic default with
   per-call override, since chunking happens before model selection.
2. **`ExtractiveSummary` as a Program or as a free function?** It does
   no LLM call. Argument for Program: uniform interface, observability,
   trace tree. Argument for function: simpler. Recommend Program; the
   trace is free and useful for hybrid programs that wrap it.
3. **Clustering library** — `Cluster` reducer needs k-means or similar.
   Add a tiny pure-numpy implementation to `kaos-nlp-transformers`
   (since clustering only matters when we already have embeddings) or
   take a `scikit-learn` extra? Recommend tiny numpy implementation;
   keeps the dependency surface flat.
4. **Hierarchical taxonomies** — should `LabelSet` itself enforce the
   tree, or should `HierarchicalClassify` reconstruct it from
   `Label.parent`? Recommend the latter; keeps `LabelSet` flat and
   makes flat-vs-hierarchical the same data with different consumers.
5. **NLI hypothesis templates** — domain-tunable string. Recommend
   exposing a registry of templates per domain (`general`, `legal`,
   `financial`) once Phase 8 ships.

---

## 11. What success looks like

A caller doing long-doc multi-label classification today writes:

```python
chunks = my_chunker(doc)
results = []
for chunk in chunks:
    r = await llm.complete(...)         # bespoke prompt
    parsed = parse_json(r)              # bespoke parser
    results.append(parsed)
final = my_aggregate(results)           # bespoke aggregator
```

After this plan ships, the same caller writes:

```python
result = await classify(
    doc,
    labels=contract_labels,
    long_strategy="chunk",
    aggregator="union",
    cited=True,
)
```

…and gets traces, cost reporting, caching, routing, and citations for
free. The same primitives compose into a no-LLM path
(`PrototypeClassify` + `UnionAggregator`) when the caller wants to
drop cost to near-zero.

The taxonomy from the design discussion becomes the parameter space of
exactly two functions.
