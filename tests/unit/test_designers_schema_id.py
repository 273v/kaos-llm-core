"""Pin the schema_id resolution contract for ``design_schema``.

The legacy ``schema_id="synthesized"`` default caused every distinct
prompt to compile to the same ``Extract_synthesized_v1`` runtime
class, collapsing distinct prompts onto a single semantic-cache key.
This was a cache-collision bug + cross-session privacy hazard.

The new contract:

1. When the caller does NOT pass ``schema_id``, the function
   auto-derives ``sd_<sha256-prefix>`` from ``(question, corpus_sample,
   model)``. Distinct triples produce distinct ids; identical triples
   are idempotent.
2. When the caller passes the legacy ``"synthesized"`` string, the
   function also auto-derives (back-compat without preserving the
   broken behavior).
3. When the caller passes any other string, it is honored verbatim.

These properties are pure / synchronous — the unit tests below verify
them without exercising the LLM ``Call`` path (which lives in live
integration tests).
"""

from __future__ import annotations

from kaos_llm_core.programs.designers import _auto_schema_id


class TestAutoSchemaId:
    """``_auto_schema_id`` is the pure deterministic derivation; the
    higher-level ``design_schema`` resolution policy uses it.
    """

    def test_format_is_sd_prefix_plus_16_hex(self) -> None:
        sid = _auto_schema_id("question A", "sample A", "anthropic:claude-sonnet-4-6")
        assert sid.startswith("sd_")
        suffix = sid.removeprefix("sd_")
        assert len(suffix) == 16
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_distinct_questions_produce_distinct_ids(self) -> None:
        sid_a = _auto_schema_id("question A", "sample", "model")
        sid_b = _auto_schema_id("question B", "sample", "model")
        assert sid_a != sid_b, (
            "distinct questions must produce distinct ids — "
            "otherwise two prompts collide on the same Extract class"
        )

    def test_distinct_corpora_produce_distinct_ids(self) -> None:
        sid_a = _auto_schema_id("question", "sample A", "model")
        sid_b = _auto_schema_id("question", "sample B", "model")
        assert sid_a != sid_b

    def test_distinct_models_produce_distinct_ids(self) -> None:
        sid_a = _auto_schema_id("question", "sample", "anthropic:claude-sonnet-4-6")
        sid_b = _auto_schema_id("question", "sample", "openai:gpt-5.4-mini")
        assert sid_a != sid_b, (
            "model identity participates in the id — DSPy posture: "
            "compiled prompts are NOT model-agnostic"
        )

    def test_identical_inputs_are_idempotent(self) -> None:
        sid_a = _auto_schema_id("question", "sample", "model")
        sid_b = _auto_schema_id("question", "sample", "model")
        assert sid_a == sid_b, "same inputs must produce same id (cache key)"

    def test_unicode_inputs_do_not_crash(self) -> None:
        sid = _auto_schema_id(
            "Compare Vertrag — München & Köln 中文",
            "Sample with non-ASCII: éàü 中",
            "anthropic:claude-sonnet-4-6",
        )
        assert sid.startswith("sd_")
        assert len(sid.removeprefix("sd_")) == 16


class TestDesignSchemaIdResolution:
    """Pin the resolution policy of the public ``design_schema``
    function's ``schema_id`` parameter without exercising the live
    Call (which lives in live integration tests).
    """

    @staticmethod
    def _resolve(
        question: str,
        corpus_sample: str,
        model: str,
        schema_id: str | None,
    ) -> str:
        """Mirror the resolution block inside ``design_schema``.

        Keeping this as a tiny helper (rather than monkey-patching the
        live ``Call``) lets the test pin the policy without LLM cost.
        If the policy in ``design_schema`` changes, this test will
        diverge from the production behavior — that divergence is the
        signal to update the test.
        """
        if schema_id is None or schema_id == "synthesized":
            return _auto_schema_id(question, corpus_sample, model)
        return schema_id

    def test_none_default_auto_derives(self) -> None:
        sid = self._resolve("q", "s", "m", None)
        assert sid.startswith("sd_")
        assert sid == _auto_schema_id("q", "s", "m")

    def test_legacy_synthesized_is_rewritten(self) -> None:
        """The bare ``"synthesized"`` constant was the cache-collision
        footgun. Callers passing it (for back-compat) get the safe
        auto-derived id instead of the broken constant.
        """
        sid = self._resolve("q", "s", "m", "synthesized")
        assert sid != "synthesized"
        assert sid.startswith("sd_")

    def test_explicit_schema_id_is_honored(self) -> None:
        sid = self._resolve("q", "s", "m", "MyDealRoom-2026")
        assert sid == "MyDealRoom-2026"

    def test_explicit_id_is_not_hashed(self) -> None:
        sid = self._resolve("q", "s", "m", "stable-handle")
        assert sid == "stable-handle"
        # confirm it differs from the auto-derived id for the same
        # (q, s, m) — explicit pass-through actually pass through
        auto = _auto_schema_id("q", "s", "m")
        assert sid != auto

    def test_distinct_questions_with_default_id_get_distinct_ids(self) -> None:
        """The motivating bug: two distinct user prompts both used
        the ``"synthesized"`` default and compiled to the same Extract
        class. The fix MUST give them distinct ids.
        """
        sid_a = self._resolve("Question A", "sample", "m", None)
        sid_b = self._resolve("Question B", "sample", "m", None)
        assert sid_a != sid_b
