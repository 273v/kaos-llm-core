# Summarization & Classification: Cross-Module Plan

**Status:** draft
**Author:** design discussion 2026-05-14
**Scope:** `kaos-nlp-core`, `kaos-nlp-transformers`, `kaos-llm-core`
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

| Program | Method | Long-doc | Faithfulness | Notes |
|---|---|---|---|---|
| `AbstractiveSummary` | abstractive | no | free | Single-shot, optional schema. |
| `ExtractiveSummary` | extractive | no | extractive-only | Wraps `ExtractiveRanker`; **no LLM call** in pure-extractive mode. |
| `CitedSummary` | abstractive | no | cited | Abstractive + `Grounded`. |
| `StructuredSummary[T]` | abstractive | no | free or cited | Pydantic output schema (`T`). |
| `MapReduceSummary` | abstractive | yes | free | Chunk → per-chunk summary → reduce. |
| `RefineSummary` | abstractive | yes | free | Sequential running summary. |
| `HierarchicalSummary` | abstractive | yes | free or cited | Tree reducer; the general workhorse. |
| `QueryFocusedSummary` | abstractive | yes | cited | BM25 + dense rerank → summarize top-k. |
| `ClusteredSummary` | abstractive | yes | free | Cluster reducer (good for multi-doc). |
| `HybridSummary` | hybrid | yes | cited | Extractive top-k → abstractive over those. |

Each takes:
- `chunker: Chunker | None` (long-doc programs only)
- `router: Router | None`
- `cache: Cache | None`
- `cited: bool = False` (where applicable)
- `output_schema: type[BaseModel] | None`
- `budget: Budget | None`

### 6.2 Classification programs (`kaos_llm_core.programs.classify`)

| Program | Supervision | Backend | Notes |
|---|---|---|---|
| `ZeroShotClassify` | zero-shot | LLM | Schema-constrained decode against `LabelSet`. |
| `FewShotClassify` | few-shot | LLM | Adds `Example` pool to the prompt. |
| `PrototypeClassify` | zero-shot | embedding | Cosine vs label-description embeddings. **No LLM.** |
| `RetrievalClassify` | many-shot | embedding + LLM | kNN over labeled corpus + vote, optionally LLM tie-break. |
| `HierarchicalClassify` | any | nested classifiers | Coarse-to-fine taxonomy walk. |
| `MultiLabelClassify` | zero/few-shot | LLM | Emits subset with per-label confidence. |
| `ChunkedClassify` | any | wrapper | Chunk → per-chunk classify → aggregator. |
| `EnsembleClassify` | any | wrapper | Multiple member classifiers + aggregator. |

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

### Phase 0 — Foundation types (1 PR per package)

- `kaos-nlp-core`: introduce `Chunk` dataclass + `Chunker` protocol stub.
- `kaos-llm-core`: introduce `Label`, `LabelSet`, `Summary[T]`,
  `Classification[L]`.
- Export from `__all__` of each package.
- Tests: round-trip frozen-dataclass behavior, hashability, JSON
  serialization for `Summary` / `Classification`.

**Deliverable:** types exist, nothing uses them yet, public API
declared.

### Phase 1 — Deterministic chunkers (`kaos-nlp-core`)

- `FixedTokenChunker`, `SentenceChunker`, `ParagraphChunker`,
  `SectionChunker`, `HierarchicalChunker`.
- Property tests: round-trip offsets, budget compliance, determinism.
- Benchmarks on a fixed legal corpus.

**Deliverable:** chunking works end-to-end without any neural or LLM dependency.

### Phase 2 — Composers (`kaos-llm-core`)

- `Reducer` protocol + `MapReduce`, `Refine`, `Tree`, `Cluster`.
- `Aggregator` protocol + five strategies.
- Backed by mock-Program tests; no live LLM in CI.

**Deliverable:** primitives ready; no user-visible behavior yet.

### Phase 3 — Summarization Programs (`kaos-llm-core`)

- `AbstractiveSummary`, `StructuredSummary`, `CitedSummary` (single-doc).
- `MapReduceSummary`, `RefineSummary`, `HierarchicalSummary` (long-doc).
- Tests use deterministic stub provider for unit; live integration
  tests behind `@pytest.mark.live`.

**Deliverable:** long-doc summarization end-to-end with cited variants.

### Phase 4 — Classification Programs (`kaos-llm-core`)

- `ZeroShotClassify`, `FewShotClassify`, `MultiLabelClassify`,
  `HierarchicalClassify`, `ChunkedClassify`, `EnsembleClassify`.

**Deliverable:** long-doc classification end-to-end, LLM-only.

### Phase 5 — Neural primitives (`kaos-nlp-transformers`)

- `SemanticChunker` implementing the `Chunker` protocol.
- `ExtractiveRanker` (centroid + cross-encoder + MMR).
- `kaos-llm-core` gains `ExtractiveSummary` and `PrototypeClassify`
  (both delegate to nlp-transformers; no LLM call in the pure-extractive /
  pure-prototype paths).

**Deliverable:** no-LLM summarization and classification paths available.

### Phase 6 — Retrieval-augmented variants

- `QueryFocusedSummary`, `ClusteredSummary`, `HybridSummary`,
  `RetrievalClassify`. Uses existing `RAG` + `Searcher` infra.

**Deliverable:** query-driven and retrieval-driven paths.

### Phase 7 — Surfaces

- Upgrade `starter.summarize()` / `classify()` to the declarative
  façade with the auto-strategy rules.
- CLI commands + MCP tools.
- Example scripts under `examples/`.

**Deliverable:** one-liner end-user API.

### Phase 8 — Future: zero-shot NLI

- Register an NLI model in `kaos-nlp-transformers` (license audit + SHA
  pin).
- Implement `ZeroShotNLIClassifier`.
- Add `supervision="nli"` route to the starter classify façade.

**Deliverable:** GPU-free, no-LLM zero-shot classification path.

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
