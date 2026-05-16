"""Tests for the §7.1 declarative starter façade.

Covers :func:`kaos_llm_core.starter.summarize_doc` and
:func:`kaos_llm_core.starter.classify_doc`. The LLM calls are stubbed
through :class:`FunctionClient` patched onto the resolved settings, so
the tests stay offline and deterministic. Live coverage lives in
``tests/integration/test_phase8_live.py``.

What's asserted:

- ``long_strategy="auto"`` resolves to ``"single"`` for short inputs
  and ``"tree"`` / ``"chunk"`` for long inputs.
- The chosen Program is the one we expect (smoke-checked via the
  ``starter.long_strategy`` and ``program`` metadata fields).
- ``cited=True`` routes the single path through ``CitedSummary``.
- ``cache=`` and ``budget=`` are threaded into the long-doc Program
  on the chunked / tree branches.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from kaos_llm_client import ModelProfile, ProviderResponse, UsageInfo
from kaos_llm_client.providers.function import FunctionClient
from kaos_llm_client.types import ContentPart
from kaos_nlp_core.chunking import SentenceChunker

from kaos_llm_core.cache import InMemoryChunkCache
from kaos_llm_core.errors import CallError
from kaos_llm_core.labels import LabelSet
from kaos_llm_core.optimization.budget import Budget
from kaos_llm_core.programs.summarize import (
    AbstractiveSummary,
    CitedSummary,
    HierarchicalSummary,
)
from kaos_llm_core.results import Classification, Summary
from kaos_llm_core.starter import (
    _resolve_long_classify_strategy,
    _resolve_long_summary_strategy,
    classify_doc,
    summarize_doc,
)


def _resp(data: dict[str, Any]) -> ProviderResponse:
    return ProviderResponse.model_construct(
        provider="function",
        model="function-test",
        raw={},
        parts=[ContentPart.model_construct(type="text", text=json.dumps(data))],
        usage=UsageInfo.model_construct(input_tokens=100, output_tokens=50, total_tokens=150),
        stop_reason="end_turn",
        response_id="test-id",
        status_code=200,
        response_headers={},
        latency_ms=10.0,
    )


class _CountingFn:
    def __init__(self, factory: Callable[[int], dict[str, Any]]) -> None:
        self._factory = factory
        self.calls: int = 0

    def __call__(self, messages: list[dict[str, Any]], profile: ModelProfile) -> ProviderResponse:
        idx = self.calls
        self.calls += 1
        return _resp(self._factory(idx))


class _ConstructorIntercepted(Exception):
    """Marker raised by a patched ``__init__`` to assert routing only.

    Used in tests that verify ``summarize_doc`` / ``classify_doc``
    routes to the expected Program subclass without having to stub the
    full LLM response shape.
    """


@pytest.fixture(autouse=True)
def _patch_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``_resolve_default_model`` always return a known stub model.

    The façade reads ``settings.default_model`` to pick a model; tests
    pass ``model="function-test"`` explicitly, but a clean env var is
    cheap insurance.
    """
    monkeypatch.setenv("KAOS_LLM_CORE_DEFAULT_MODEL", "function-test")


# ---------------------------------------------------------------------------
# strategy resolution
# ---------------------------------------------------------------------------


class TestStrategyResolution:
    def test_summary_explicit_passes_through(self) -> None:
        assert _resolve_long_summary_strategy(100, "single") == "single"
        assert _resolve_long_summary_strategy(1_000_000, "single") == "single"
        assert _resolve_long_summary_strategy(100, "refine") == "refine"

    def test_summary_auto_short_returns_single(self) -> None:
        assert _resolve_long_summary_strategy(100, "auto") == "single"
        assert _resolve_long_summary_strategy(12_000, "auto") == "single"

    def test_summary_auto_long_returns_tree(self) -> None:
        assert _resolve_long_summary_strategy(12_001, "auto") == "tree"

    def test_classify_auto(self) -> None:
        assert _resolve_long_classify_strategy(100, "auto") == "single"
        assert _resolve_long_classify_strategy(12_001, "auto") == "chunk"
        # Explicit honors the request.
        assert _resolve_long_classify_strategy(100, "chunk") == "chunk"


# ---------------------------------------------------------------------------
# summarize_doc
# ---------------------------------------------------------------------------


class TestSummarizeDocSingle:
    @pytest.mark.asyncio
    async def test_short_input_uses_abstractive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patch the AbstractiveSummary __init__ to inject our FunctionClient.
        # The façade builds the Program internally; we monkey-patch the
        # class to intercept the construction.
        fn = _CountingFn(lambda i: {"summary": "short doc summary"})
        original_init = AbstractiveSummary.__init__

        def _patched_init(self: AbstractiveSummary, **kwargs: Any) -> None:
            kwargs.setdefault("client", FunctionClient(function=fn))
            original_init(self, **kwargs)

        monkeypatch.setattr(AbstractiveSummary, "__init__", _patched_init)

        result = await summarize_doc("This is a short document.")
        assert isinstance(result, Summary)
        assert result.text == "short doc summary"
        assert result.metadata["starter.long_strategy"] == "single"
        assert result.metadata["starter.facade"] == "summarize_doc"
        assert fn.calls == 1

    @pytest.mark.asyncio
    async def test_cited_routes_through_cited_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Verify the façade *constructs* a CitedSummary when ``cited=True``.
        # The CitedSummary GroundedAnswer schema is exhaustively covered in
        # ``test_summarize_programs.py``; here we only assert routing.
        recorded: list[type[Any]] = []

        def _patched_init(self: CitedSummary, **kwargs: Any) -> None:
            recorded.append(type(self))
            # Short-circuit construction: don't actually build a Call so
            # we don't need to stub a valid GroundedAnswer response.
            raise _ConstructorIntercepted

        monkeypatch.setattr(CitedSummary, "__init__", _patched_init)

        with pytest.raises(_ConstructorIntercepted):
            await summarize_doc("A short cited doc.", cited=True)
        assert recorded == [CitedSummary]


class TestSummarizeDocQueryRoute:
    """Plan §7.1: ``summarize_doc(query=…)`` overrides ``long_strategy``
    and routes through :class:`QueryFocusedSummary`."""

    @pytest.mark.asyncio
    async def test_query_routes_to_query_focused_summary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from kaos_llm_core.programs.summarize import QueryFocusedSummary

        recorded: list[type[Any]] = []

        def _patched_init(self: QueryFocusedSummary, **kwargs: Any) -> None:
            recorded.append(type(self))
            raise _ConstructorIntercepted

        monkeypatch.setattr(QueryFocusedSummary, "__init__", _patched_init)

        class _StubEmbedder:
            def embed(self, texts, *, batch_size=32):
                import numpy as np

                return np.zeros((len(list(texts)), 2), dtype=np.float32)

        with pytest.raises(_ConstructorIntercepted):
            await summarize_doc("Any doc.", query="who signed?", embedder=_StubEmbedder())
        assert recorded == [QueryFocusedSummary]

    @pytest.mark.asyncio
    async def test_query_requires_embedder(self) -> None:
        with pytest.raises(CallError, match="embedder"):
            await summarize_doc("Any doc.", query="who signed?")


class TestSummarizeDocLong:
    @pytest.mark.asyncio
    async def test_long_input_auto_resolves_to_tree(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fn = _CountingFn(lambda i: {"summary": f"chunk-{i}"})
        original_init = HierarchicalSummary.__init__

        def _patched_init(self: HierarchicalSummary, **kwargs: Any) -> None:
            # Inject FunctionClient + a one-sentence chunker so the
            # workload is deterministic at 3 leaves.
            kwargs.setdefault("client", FunctionClient(function=fn))
            kwargs.setdefault("chunker", SentenceChunker(max_tokens=1))
            original_init(self, **kwargs)

        monkeypatch.setattr(HierarchicalSummary, "__init__", _patched_init)

        long_doc = (
            "Paragraph one talks about lease terms. "
            "Paragraph two discusses indemnification. "
            "Paragraph three covers limitation of liability. " * 200
        )
        # The long_doc above is way over the 12k char threshold.
        assert len(long_doc) > 12_000
        result = await summarize_doc(long_doc)
        assert result.metadata["starter.long_strategy"] == "tree"
        assert result.metadata["starter.facade"] == "summarize_doc"
        # Some number of LLM calls happened on the leaves + merge tree.
        assert fn.calls > 0

    @pytest.mark.asyncio
    async def test_cache_and_budget_threaded_into_long_doc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fn = _CountingFn(lambda i: {"summary": f"chunk-{i}"})
        original_init = HierarchicalSummary.__init__
        seen_kwargs: dict[str, Any] = {}

        def _patched_init(self: HierarchicalSummary, **kwargs: Any) -> None:
            seen_kwargs.update(kwargs)
            kwargs.setdefault("client", FunctionClient(function=fn))
            kwargs.setdefault("chunker", SentenceChunker(max_tokens=1))
            original_init(self, **kwargs)

        monkeypatch.setattr(HierarchicalSummary, "__init__", _patched_init)

        cache = InMemoryChunkCache()
        budget = Budget(max_tokens=10_000)
        await summarize_doc(
            "Sentence one. Sentence two. Sentence three.",
            long_strategy="tree",  # force long path regardless of length
            cache=cache,
            budget=budget,
        )
        # The façade threaded our cache + budget into the program.
        assert seen_kwargs["cache"] is cache
        assert seen_kwargs["budget"] is budget


# ---------------------------------------------------------------------------
# classify_doc
# ---------------------------------------------------------------------------


_LABEL_NAMES = ("contract", "memo")


class TestClassifyDocSingle:
    @pytest.mark.asyncio
    async def test_short_input_uses_zero_shot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fn = _CountingFn(lambda i: {"label": "contract", "confidence": 0.9, "rationale": "stub"})
        from kaos_llm_core.programs.classify import ZeroShotClassify

        original_init = ZeroShotClassify.__init__

        def _patched_init(self: ZeroShotClassify, **kwargs: Any) -> None:
            kwargs.setdefault("client", FunctionClient(function=fn))
            original_init(self, **kwargs)

        monkeypatch.setattr(ZeroShotClassify, "__init__", _patched_init)

        result = await classify_doc("Short doc.", _LABEL_NAMES)
        assert isinstance(result, Classification)
        assert result.top_label == "contract"
        assert result.metadata["starter.long_strategy"] == "single"
        assert result.metadata["starter.facade"] == "classify_doc"
        assert result.metadata["starter.supervision"] == "zero_shot"

    @pytest.mark.asyncio
    async def test_accepts_existing_label_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fn = _CountingFn(lambda i: {"label": "memo", "confidence": 0.7, "rationale": "stub"})
        from kaos_llm_core.programs.classify import ZeroShotClassify

        original_init = ZeroShotClassify.__init__

        def _patched_init(self: ZeroShotClassify, **kwargs: Any) -> None:
            kwargs.setdefault("client", FunctionClient(function=fn))
            original_init(self, **kwargs)

        monkeypatch.setattr(ZeroShotClassify, "__init__", _patched_init)

        label_set = LabelSet.from_names(_LABEL_NAMES)
        result = await classify_doc("Short doc.", label_set)
        assert result.top_label == "memo"

    @pytest.mark.asyncio
    async def test_empty_labels_raises(self) -> None:
        with pytest.raises(CallError, match="non-empty"):
            await classify_doc("x", [])

    @pytest.mark.asyncio
    async def test_few_shot_requires_examples(self) -> None:
        with pytest.raises(CallError, match="few_shot"):
            await classify_doc(
                "x",
                _LABEL_NAMES,
                supervision="few_shot",
            )

    @pytest.mark.asyncio
    async def test_prototype_requires_embedder(self) -> None:
        with pytest.raises(CallError, match="embedder"):
            await classify_doc("x", _LABEL_NAMES, supervision="prototype")

    @pytest.mark.asyncio
    async def test_retrieval_requires_corpus(self) -> None:
        class _Emb:
            def embed(self, texts, *, batch_size=32):
                import numpy as np

                return np.zeros((len(list(texts)), 2), dtype=np.float32)

        with pytest.raises(CallError, match="corpus"):
            await classify_doc(
                "x",
                _LABEL_NAMES,
                supervision="retrieval",
                embedder=_Emb(),
            )

    @pytest.mark.asyncio
    async def test_nli_requires_scorer(self) -> None:
        with pytest.raises(CallError, match="nli_scorer"):
            await classify_doc("x", _LABEL_NAMES, supervision="nli")

    @pytest.mark.asyncio
    async def test_aggregator_string_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plan §11 endgame: ``classify_doc(..., aggregator="union", ...)``
        must resolve the short name into the matching Aggregator."""
        from kaos_llm_core.composition import UnionAggregator
        from kaos_llm_core.programs.classify import ChunkedClassify, ZeroShotClassify

        seen: dict[str, Any] = {}
        orig = ChunkedClassify.__init__

        def _patched(self: ChunkedClassify, **kwargs: Any) -> None:
            seen.update(kwargs)
            orig(self, **kwargs)

        monkeypatch.setattr(ChunkedClassify, "__init__", _patched)
        # Avoid actually running the inner LLM path; we only assert
        # the aggregator the façade hands to ChunkedClassify.
        orig_zero = ZeroShotClassify.__init__

        def _patched_zero(self: ZeroShotClassify, **kwargs: Any) -> None:
            from kaos_llm_client.providers.function import FunctionClient

            # The fixture _LABEL_NAMES is ("contract", "memo"); the
            # ZeroShotClassify literal-typed output validates against it.
            kwargs.setdefault(
                "client",
                FunctionClient(
                    function=lambda messages, profile: _resp(
                        {"label": "contract", "confidence": 0.9, "rationale": "x"}
                    )
                ),
            )
            orig_zero(self, **kwargs)

        monkeypatch.setattr(ZeroShotClassify, "__init__", _patched_zero)

        from kaos_nlp_core.chunking import SentenceChunker

        await classify_doc(
            "Sentence one. Sentence two. " * 5_000,  # force chunk path
            _LABEL_NAMES,
            aggregator="union",
            chunker=SentenceChunker(max_tokens=1),
        )
        assert isinstance(seen["aggregator"], UnionAggregator)

    @pytest.mark.asyncio
    async def test_unknown_aggregator_string_raises(self) -> None:
        with pytest.raises(CallError, match="unknown aggregator"):
            await classify_doc(
                "x",
                _LABEL_NAMES,
                aggregator="not_a_real_aggregator",
                long_strategy="chunk",
            )


class TestClassifyDocLong:
    @pytest.mark.asyncio
    async def test_auto_long_input_uses_chunked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from kaos_llm_core.programs.classify import ChunkedClassify, ZeroShotClassify

        fn = _CountingFn(lambda i: {"label": "contract", "confidence": 0.9, "rationale": "stub"})
        orig_zero = ZeroShotClassify.__init__

        def _patched_zero(self: ZeroShotClassify, **kwargs: Any) -> None:
            kwargs.setdefault("client", FunctionClient(function=fn))
            orig_zero(self, **kwargs)

        monkeypatch.setattr(ZeroShotClassify, "__init__", _patched_zero)

        orig_chunked = ChunkedClassify.__init__

        def _patched_chunked(self: ChunkedClassify, **kwargs: Any) -> None:
            kwargs.setdefault("chunker", SentenceChunker(max_tokens=1))
            orig_chunked(self, **kwargs)

        monkeypatch.setattr(ChunkedClassify, "__init__", _patched_chunked)

        long_doc = "Sentence one. Sentence two. Sentence three. " * 500
        assert len(long_doc) > 12_000
        result = await classify_doc(long_doc, _LABEL_NAMES)
        assert result.metadata["starter.long_strategy"] == "chunk"
        assert result.metadata["starter.facade"] == "classify_doc"
        assert fn.calls > 0
