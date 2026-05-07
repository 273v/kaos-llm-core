"""Live integration tests for kaos-llm-core Phase 8 (starter API + multimodal).

These hit real LLM provider APIs to verify the Phase 8 surfaces end-to-end:

1. ``starter`` module (`text`, `extract`, `classify`, `summarize`) against
   a real Anthropic model — confirms the §11.D rule that ``_resolve_default_model``
   goes through ``KaosLLMCoreSettings``, not direct env reads.
2. Multimodal codec routing through every codec (JSON / Chat / XML) against
   a real vision-capable model. Caught a real bug during E2E: the empty-text
   edge case where Anthropic rejects zero-length text content blocks.
3. Configuration profile field round-trip (no API call needed for the
   profile resolution itself).

Run::

    uv run pytest tests/integration/test_phase8_live.py -v -m integration -s
"""

from __future__ import annotations

import os
import struct
import zlib

import pytest

from kaos_llm_core.codecs.chat_codec import ChatCodec
from kaos_llm_core.codecs.json_codec import JSONCodec
from kaos_llm_core.codecs.xml_codec import XMLCodec
from kaos_llm_core.programs.call import Call
from kaos_llm_core.settings import KaosLLMCoreSettings
from kaos_llm_core.signatures import InputField, OutputField, Signature
from kaos_llm_core.signatures.multimodal import Image
from kaos_llm_core.starter import classify, extract, summarize, text

pytestmark = pytest.mark.integration


def _has_key(*env_vars: str) -> bool:
    return any(os.getenv(v) for v in env_vars)


requires_anthropic = pytest.mark.skipif(
    not _has_key("KAOS_LLM_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    reason="No Anthropic API key",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Build a solid-color PNG inline. No external deps.

    Used by the live multimodal test so we can verify a real vision model
    correctly identifies the dominant color.
    """
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b""
    for _ in range(height):
        raw += b"\x00" + bytes(rgb) * width
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# ---------------------------------------------------------------------------
# Real-domain task: contract clause classification (small)
# ---------------------------------------------------------------------------


_TERMINATION_CLAUSE = (
    "Either party may terminate this Agreement upon thirty (30) days "
    "written notice to the other party."
)


# ---------------------------------------------------------------------------
# 1. starter API live tests
# ---------------------------------------------------------------------------


class TestStarterLive:
    """Phase 8.1 — the lowest-friction Python surface."""

    @requires_anthropic
    async def test_text_against_real_haiku(self) -> None:
        result = await text(
            "Reply with the single word 'yes'.",
            model="anthropic:claude-haiku-4-5",
        )
        assert isinstance(result, str)
        assert "yes" in result.lower()
        print(f"\n  [starter_text] result={result!r}")

    @requires_anthropic
    async def test_extract_against_real_haiku(self) -> None:
        result = await extract(
            "John Smith is 32 years old and lives in Boston.",
            {"name": str, "age": int, "city": str},
            model="anthropic:claude-haiku-4-5",
        )
        assert isinstance(result, dict)
        # The model should pick out at least name and age correctly.
        assert "Smith" in str(result.get("name", "")) or "John" in str(result.get("name", ""))
        assert result.get("age") == 32
        print(f"\n  [starter_extract] result={result}")

    @requires_anthropic
    async def test_classify_against_real_haiku(self) -> None:
        label = await classify(
            _TERMINATION_CLAUSE,
            labels=[
                "indemnification",
                "limitation_of_liability",
                "confidentiality",
                "termination",
                "payment_terms",
            ],
            model="anthropic:claude-haiku-4-5",
        )
        assert label == "termination", (
            f"Expected the standard termination clause to be classified as "
            f"'termination'; got {label!r}"
        )
        print(f"\n  [starter_classify] label={label!r}")

    @requires_anthropic
    async def test_summarize_against_real_haiku(self) -> None:
        long_text = (
            "The Securities and Exchange Commission filed a civil enforcement "
            "action against Acme Corp on March 14, 2026, alleging that Acme "
            "made materially misleading statements regarding its quarterly "
            "earnings forecasts during the period from Q3 2025 through Q1 "
            "2026. The complaint seeks disgorgement, civil penalties, and "
            "injunctive relief barring Acme from future violations of the "
            "federal securities laws. Acme denies the allegations and intends "
            "to defend against the action vigorously."
        )
        summary = await summarize(
            long_text,
            model="anthropic:claude-haiku-4-5",
            max_words=30,
            style="concise",
        )
        assert isinstance(summary, str)
        assert len(summary) > 0
        # The summary should mention either SEC, Acme, or the enforcement action.
        assert any(term in summary for term in ("SEC", "Acme", "enforcement", "Securities"))
        print(f"\n  [starter_summarize] {summary[:200]!r}")

    @requires_anthropic
    async def test_starter_resolves_via_settings_not_environ(self) -> None:
        """§11.D regression: starter must resolve model via KaosLLMCoreSettings,
        not via direct os.environ reads.
        """
        # Build a settings instance with an explicit default_model and verify
        # that starter.text() picks it up via the settings hierarchy.
        settings = KaosLLMCoreSettings(default_model="anthropic:claude-haiku-4-5")
        result = await text(
            "Reply with the single word 'ok'.",
            settings=settings,
        )
        assert "ok" in result.lower()


# ---------------------------------------------------------------------------
# 2. Multimodal codec round-trip against real vision model
# ---------------------------------------------------------------------------


class _ColorOnly(Signature):
    """Identify the dominant color of the supplied image. Reply with the color
    name only, lowercase, no punctuation.
    """

    image: Image = InputField(description="The image")
    color: str = OutputField(description="Dominant color, lowercase, single word")


class _ColorWithNote(Signature):
    """Identify the dominant color of the supplied image. The note field is
    additional context the user is providing alongside the image. Reply with
    the color name only, lowercase, no punctuation.
    """

    image: Image = InputField(description="The image")
    note: str = InputField(description="Optional context")
    color: str = OutputField(description="Dominant color, lowercase, single word")


class TestMultimodalLive:
    """Phase 8.2 — multimodal signature fields against a real vision model."""

    @requires_anthropic
    async def test_image_only_signature_against_haiku(self) -> None:
        """Phase 8.2 live regression: a Signature with ONLY a binary input.

        This is the empty-text edge case that Anthropic rejected with
        ``messages: text content blocks must be non-empty`` until the
        attach_binaries_to_user_message helper learned to omit zero-length
        text parts. The fix is verified at the unit level by
        test_codec_multimodal.py::TestEmptyTextEdgeCase, but we keep a live
        test here to confirm the behavior end-to-end against the actual
        Anthropic API.
        """
        png = _solid_png(8, 8, (255, 0, 0))  # Red
        img = Image.from_bytes(png, media_type="image/png")
        call = Call(_ColorOnly, model="anthropic:claude-haiku-4-5")
        invocation = await call.invoke(image=img)
        result = invocation.output
        assert "red" in result.color.lower(), (
            f"Expected the model to identify a red 8x8 PNG as red; got {result.color!r}"
        )
        # Cost / token plumbing must populate after the GAP-2 fix.
        trace = invocation.trace
        assert trace is not None
        assert trace.total_tokens > 0
        assert trace.cost_usd > 0
        print(f"\n  [multimodal_image_only] color={result.color!r} cost=${trace.cost_usd:.6f}")

    @requires_anthropic
    async def test_image_with_text_note_against_haiku(self) -> None:
        """The mixed case: a Signature with both a binary input AND a text input.

        Verifies the codec correctly emits a multipart user message with BOTH
        a non-empty text part (the note) and the image content part.
        """
        png = _solid_png(8, 8, (0, 255, 0))  # Green
        img = Image.from_bytes(png, media_type="image/png")
        call = Call(_ColorWithNote, model="anthropic:claude-haiku-4-5")
        result = await call(image=img, note="This is a small color swatch")
        assert "green" in result.color.lower(), f"Expected green; got {result.color!r}"
        print(f"\n  [multimodal_image_with_note] color={result.color!r}")

    def test_unit_round_trip_through_all_codecs_no_api(self) -> None:
        """Smoke: every codec produces a non-empty multipart message for an
        image-only signature. (No API call.)"""
        png = _solid_png(4, 4, (0, 0, 255))
        img = Image.from_bytes(png, media_type="image/png")
        for codec in (JSONCodec(), ChatCodec(), XMLCodec()):
            messages = codec.encode(_ColorOnly, {"image": img})
            final = messages[-1]
            assert final["role"] == "user"
            assert isinstance(final["content"], list)
            assert any(p.get("type") == "image_url" for p in final["content"])
            for part in final["content"]:
                if part.get("type") == "text":
                    assert part["text"], (
                        f"{type(codec).__name__} emitted an empty text part for an "
                        f"image-only signature; the empty-text fix has regressed."
                    )


# ---------------------------------------------------------------------------
# 3. Profile settings live (no API call needed)
# ---------------------------------------------------------------------------


class TestProfileLive:
    """Phase 8.3 — configuration profile resolution."""

    def test_global_profile_env_var_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KAOS_PROFILE is honored when KAOS_LLM_CORE_PROFILE is unset."""
        monkeypatch.delenv("KAOS_LLM_CORE_PROFILE", raising=False)
        monkeypatch.setenv("KAOS_PROFILE", "dev")
        settings = KaosLLMCoreSettings()
        assert settings.profile == "dev"

    def test_module_specific_overrides_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KAOS_PROFILE", "dev")
        monkeypatch.setenv("KAOS_LLM_CORE_PROFILE", "prod")
        settings = KaosLLMCoreSettings()
        assert settings.profile == "prod"
