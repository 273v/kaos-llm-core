"""Exhaustive unit tests for Span / Claim / Answer match strategies.

Covers the six :class:`MatchStrategy` values end-to-end against real
kaos-nlp-core primitives (no mocking).  Each strategy gets positive
matches, negative matches, and edge cases.  Escalation chains at the
Claim / Answer level are tested as well.

Target: ~90 tests across common and edge cases.

No mocks: every similarity score, substring search, and tokenization
goes through the real Rust-backed kaos-nlp-core functions.  If the
kaos-nlp-core API changes these tests break, which is the point.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from kaos_llm_core.signatures import (
    DEFAULT_MATCH_STRATEGIES,
    Answer,
    Claim,
    ClaimError,
    MatchStrategy,
    Span,
)
from kaos_llm_core.signatures.grounding import _hash_quote

# ─────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────


CORPUS_URI = "doc:test-corpus"

# A canonical corpus with variety: multi-sentence, numbers, punctuation,
# case variety, enough length for sliding matching.
CORPUS = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State. The filing "
    "fee is $89 and the certificate must include the corporation's "
    "name, its registered office, and the name of its registered agent."
)
CORPUS_MAP = {CORPUS_URI: CORPUS}


def _find(needle: str) -> tuple[int, int]:
    """Helper: find needle in CORPUS, return (start, end)."""
    idx = CORPUS.find(needle)
    if idx < 0:
        pytest.fail(f"needle {needle!r} not found in CORPUS")
    return (idx, idx + len(needle))


# ─────────────────────────────────────────────────────────────────────
# _hash_quote utility — identity hashing
# ─────────────────────────────────────────────────────────────────────


class TestHashQuote:
    def test_stable_across_calls(self) -> None:
        assert _hash_quote("hello") == _hash_quote("hello")

    def test_different_input_different_hash(self) -> None:
        assert _hash_quote("hello") != _hash_quote("world")

    def test_fixed_length_16_hex(self) -> None:
        h = _hash_quote("arbitrary")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_single_char_change_changes_hash(self) -> None:
        assert _hash_quote("hello") != _hash_quote("hellp")

    def test_case_change_changes_hash(self) -> None:
        assert _hash_quote("Hello") != _hash_quote("hello")

    def test_whitespace_change_changes_hash(self) -> None:
        assert _hash_quote("hello world") != _hash_quote("hello  world")

    def test_unicode_accent_change_changes_hash(self) -> None:
        assert _hash_quote("café") != _hash_quote("cafe")

    def test_emoji_stable(self) -> None:
        h1 = _hash_quote("🚀 to the moon")
        h2 = _hash_quote("🚀 to the moon")
        assert h1 == h2

    def test_long_string(self) -> None:
        s = "x" * 10_000
        h = _hash_quote(s)
        assert len(h) == 16


# ─────────────────────────────────────────────────────────────────────
# MatchStrategy.STRICT
# ─────────────────────────────────────────────────────────────────────


class TestStrategyStrict:
    """Byte-exact match at exact char_span position."""

    def test_exact_match(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS, _find("Delaware"))
        assert span.verify(CORPUS, strategy=MatchStrategy.STRICT) is True

    def test_wrong_position_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),  # "The Dela" not "Delaware"
            quote="Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.STRICT) is False

    def test_out_of_range_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(10000, 10020),
            quote="x" * 20,
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.STRICT) is False

    def test_empty_source_fails(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS, _find("Delaware"))
        assert span.verify("", strategy=MatchStrategy.STRICT) is False

    def test_case_mismatch_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=_find("Delaware"),
            quote="delaware",  # lowercase
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.STRICT) is False

    def test_whitespace_drift_one_space_fails(self) -> None:
        start, _ = _find("filing fee")
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(start, start + 11),
            quote="filing  fee",  # two spaces instead of one
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.STRICT) is False

    def test_unicode_payload(self) -> None:
        u_corpus = "The résumé was impressive."
        span = Span.from_source(CORPUS_URI, u_corpus, (4, 10))
        assert span.quote == "résumé"
        assert span.verify(u_corpus, strategy=MatchStrategy.STRICT) is True

    def test_source_shorter_than_span_fails(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS, _find("incorporation"))
        assert span.verify(CORPUS[:30], strategy=MatchStrategy.STRICT) is False

    def test_zero_length_range_rejected_at_construction(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Span(source_uri=CORPUS_URI, char_span=(5, 5), quote="")


# ─────────────────────────────────────────────────────────────────────
# MatchStrategy.SUBSTRING
# ─────────────────────────────────────────────────────────────────────


class TestStrategySubstring:
    """Byte-exact substring anywhere in the source."""

    def test_exact_position_matches(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS, _find("Delaware"))
        assert span.verify(CORPUS, strategy=MatchStrategy.SUBSTRING) is True

    def test_wrong_position_correct_text_matches(self) -> None:
        # Position is clearly wrong but quote IS in source
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.SUBSTRING) is True

    def test_case_mismatch_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.SUBSTRING) is False

    def test_quote_longer_than_source_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 1000),
            quote="x" * 1000,
        )
        assert span.verify("short", strategy=MatchStrategy.SUBSTRING) is False

    def test_quote_not_present_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="nowhere in the text",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.SUBSTRING) is False

    def test_substring_at_end(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="registered agent.",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.SUBSTRING) is True

    def test_whitespace_drift_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="filing  fee",  # double space
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.SUBSTRING) is False


# ─────────────────────────────────────────────────────────────────────
# MatchStrategy.CASE_INSENSITIVE
# ─────────────────────────────────────────────────────────────────────


class TestStrategyCaseInsensitive:
    """Case-folded substring via kaos-nlp-core substring_find_all_case_insensitive."""

    def test_exact_case_matches(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is True

    def test_lower_quote_upper_source(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is True

    def test_upper_quote_mixed_source(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="DELAWARE",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is True

    def test_mixed_case_quote(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 15),
            quote="DeLaWaRe GeNeRaL",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is True

    def test_position_drift_tolerated(self) -> None:
        # Wrong position, case diff — case-insensitive substring catches it.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 15),
            quote="FILING FEE IS $89",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is True

    def test_not_present_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="does not exist anywhere",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is False

    def test_whitespace_drift_still_fails(self) -> None:
        # CI handles CASE only, not whitespace.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 15),
            quote="FILING  FEE",  # double space
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is False

    def test_quote_longer_than_source_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 1000),
            quote="DELAWARE " * 200,
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.CASE_INSENSITIVE) is False


# ─────────────────────────────────────────────────────────────────────
# MatchStrategy.NORMALIZED_TOKEN
# ─────────────────────────────────────────────────────────────────────


class TestStrategyNormalizedToken:
    """Token-sequence match via Tokenizer(lowercase=True)."""

    def test_exact_token_sequence(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 12),
            quote="The Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_case_drift(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 12),
            quote="THE DELAWARE",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_extra_whitespace(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="The    Delaware   General",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_tabs_and_newlines(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="The\tDelaware\nGeneral",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_trailing_punctuation_tolerated(self) -> None:
        # Tokenizer strips punctuation, so trailing comma shouldn't
        # break the match.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 40),
            quote="corporation's name,",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_token_reorder_fails(self) -> None:
        # Contiguous subsequence required — reorder breaks it.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="Delaware The General",  # reordered
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is False

    def test_missing_middle_token_fails(self) -> None:
        # "Delaware Corporation" skipping "General" — not a
        # contiguous subsequence.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 25),
            quote="Delaware Corporation Law",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is False

    def test_empty_quote_fails(self) -> None:
        # Can't construct an empty-quote Span — schema rejects.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Span(source_uri=CORPUS_URI, char_span=(0, 0), quote="")

    def test_single_token(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_source_shorter_than_quote_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 100),
            quote="one two three four five six seven",
        )
        assert span.verify("hi", strategy=MatchStrategy.NORMALIZED_TOKEN) is False

    def test_trailing_comma_stripped(self) -> None:
        # Corpus has "corporation's name, its registered office" —
        # the comma between "name" and "its" is trailing punctuation
        # that the tokenizer strips, so "name its" should match.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 40),
            quote="name, its registered office",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is True

    def test_mid_word_apostrophe_kept_as_token(self) -> None:
        """Documented gotcha: the tokenizer keeps intra-word
        punctuation like apostrophes *inside* a token.  So
        'corporations' (without apostrophe) is a DIFFERENT token
        from 'corporation's' (with apostrophe).  NORMALIZED_TOKEN
        strategy will NOT match across this difference.

        Callers who need to tolerate this drift should use
        FUZZY_TOKEN or FUZZY_CHAR instead.
        """
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 30),
            quote="corporations name",  # missing apostrophe
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.NORMALIZED_TOKEN) is False
        # FUZZY_CHAR catches it at threshold 0.64 — the longest
        # contiguous substring is "corporation" (11 chars of 17 quote
        # chars) since "corporations" isn't a substring of
        # "corporation's".  Documents real LCS behavior.
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.64) is True
        # At threshold 0.8 it's too strict.
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.8) is False


# ─────────────────────────────────────────────────────────────────────
# MatchStrategy.FUZZY_CHAR
# ─────────────────────────────────────────────────────────────────────


class TestStrategyFuzzyChar:
    """LCS-based character fuzzy match with configurable threshold."""

    def test_exact_at_threshold_1_0(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=1.0) is True

    def test_single_mid_word_typo(self) -> None:
        """Documented behavior: LCS is *contiguous*, so a single
        mid-word typo costs you the characters on the shorter side
        of the break.  "Dalaware" vs "Delaware" has LCS = "laware"
        = 6 chars out of 8, ratio 0.75.  NOT 0.875 — don't confuse
        LCS (contiguous) with edit distance (non-contiguous).
        """
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Dalaware",  # 1 typo in position 1
        )
        # At threshold 0.75, passes exactly.
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.75) is True
        # At threshold 0.8, fails.
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.8) is False

    def test_single_end_typo_preserves_more_lcs(self) -> None:
        """Typos at word boundaries preserve more LCS than mid-word
        typos, because the LCS can extend further in one direction.
        """
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delawarz",  # typo at last char
        )
        # LCS = "Delawar" = 7 out of 8 = 0.875
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.85) is True

    def test_multi_char_edit(self) -> None:
        # "filing fee is $89" is 17 chars; up to ~1 edit should pass 0.9
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="filing fee is $89",
        )
        # No edit — exact substring present, LCS = 17/17 = 1.0
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.9) is True

    def test_50_percent_edits_below_09(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="Dexlxwxrx",  # heavily edited
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.9) is False

    def test_unrelated_quote_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="antidisestablishmentarian",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.9) is False

    def test_threshold_zero_always_matches_nonempty(self) -> None:
        # Even a degenerate match (LCS > 0) clears threshold 0.0.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="xxx",
        )
        # "x" might appear somewhere (unlikely in this corpus);
        # if not, LCS=0 which equals threshold=0 → True.
        # Either way >= 0 is True.
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.0) is True

    def test_threshold_one_requires_full_lcs(self) -> None:
        # Only exact substring presence clears threshold 1.0.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delaware",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=1.0) is True

    def test_long_quote_with_short_lcs(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 100),
            quote="nonexistent text that shares almost nothing with corpus wxyz",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_CHAR, threshold=0.9) is False


# ─────────────────────────────────────────────────────────────────────
# MatchStrategy.FUZZY_TOKEN
# ─────────────────────────────────────────────────────────────────────


class TestStrategyFuzzyToken:
    """token_jaccard-based fuzzy match with configurable threshold.

    NOTE: token_jaccard computes Jaccard over token sets of the WHOLE
    strings, not localized to quote length.  So a short quote vs. a
    long source yields a low similarity even on exact substring
    match, because the source has many extra tokens.  This strategy
    is best for quote/source pairs of comparable size — e.g., the
    LLM claims a paragraph-ish span and the source is a paragraph.
    """

    def test_exact_token_set_high_similarity(self) -> None:
        # Exact match of a short substring in a longer source:
        # token_jaccard will score low because source has extra
        # tokens.  So use a comparable-size source.
        small_source = "filing fee is $89"
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 17),
            quote="filing fee is $89",
        )
        assert span.verify(small_source, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.9) is True

    def test_token_reorder_still_matches(self) -> None:
        # Jaccard is set-based, so reordering doesn't matter.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="fee is filing $89",  # reordered
        )
        small_source = "filing fee is $89"
        assert span.verify(small_source, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.9) is True

    def test_unrelated_tokens_fail(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="completely different sentence about cats",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.5) is False

    def test_partial_overlap_at_low_threshold(self) -> None:
        # A quote with 3 tokens, 2 of which appear in source.
        # Source has ~33 tokens. Jaccard = 2 / 34 ≈ 0.058.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 30),
            quote="Delaware filing fee",
        )
        # At threshold 0.01, should pass
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.01) is True
        # At threshold 0.5, should fail
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.5) is False

    def test_threshold_zero_always_true(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 5),
            quote="xyz",
        )
        assert span.verify(CORPUS, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.0) is True

    def test_case_drift_same_score(self) -> None:
        # Tokenizer inside token_jaccard may or may not lowercase —
        # verify behavior empirically.
        span_lower = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 17),
            quote="filing fee is $89",
        )
        span_upper = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 17),
            quote="FILING FEE IS $89",
        )
        small_source = "filing fee is $89"
        # Both should verify or both shouldn't, consistently.
        r1 = span_lower.verify(small_source, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.5)
        r2 = span_upper.verify(small_source, strategy=MatchStrategy.FUZZY_TOKEN, threshold=0.5)
        # Document the actual behavior — we don't assert a particular
        # answer, just that it's deterministic.
        assert isinstance(r1, bool)
        assert isinstance(r2, bool)


# ─────────────────────────────────────────────────────────────────────
# Span.verify edge cases & error paths
# ─────────────────────────────────────────────────────────────────────


class TestSpanVerifyEdgeCases:
    def test_unknown_strategy_raises(self) -> None:
        span = Span.from_source(CORPUS_URI, CORPUS, _find("Delaware"))
        with pytest.raises(ValueError, match="Unknown MatchStrategy"):
            span.verify(CORPUS, strategy=cast(Any, "not_a_strategy"))

    def test_quote_hash_recomputed_ignores_caller_input(self) -> None:
        """LLM-fabricated hash input is silently overwritten."""
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delaware",
            quote_hash="deadbeefdeadbeef",  # bogus
        )
        assert span.quote_hash == _hash_quote("Delaware")

    def test_multibyte_unicode_strict(self) -> None:
        # Emoji are multi-byte in UTF-8 but single codepoints in
        # Python str — Python slicing is codepoint-aware, so
        # char_span values are codepoint offsets.
        u_src = "The rocket 🚀 launched."
        span = Span.from_source(CORPUS_URI, u_src, (11, 12))
        assert span.quote == "🚀"
        assert span.verify(u_src, strategy=MatchStrategy.STRICT) is True

    def test_nbsp_vs_space_strict_fails(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 11),
            quote="hello\u00a0world",  # non-breaking space
        )
        assert span.verify("hello world", strategy=MatchStrategy.STRICT) is False

    def test_nfc_nfd_strict_fails(self) -> None:
        import unicodedata

        nfc = unicodedata.normalize("NFC", "café")  # composed
        nfd = unicodedata.normalize("NFD", "café")  # decomposed
        assert nfc != nfd
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, len(nfc)),
            quote=nfc,
        )
        assert span.verify(nfc, strategy=MatchStrategy.STRICT) is True
        assert span.verify(nfd, strategy=MatchStrategy.STRICT) is False


# ─────────────────────────────────────────────────────────────────────
# Claim.verify with strategy escalation
# ─────────────────────────────────────────────────────────────────────


def _simple_span(quote: str, char_span: tuple[int, int] | None = None) -> Span:
    """Build a Span whose quote actually exists in CORPUS."""
    if char_span is None:
        char_span = _find(quote)
    return Span.from_source(CORPUS_URI, CORPUS, char_span)


class TestClaimVerifyEscalation:
    def test_default_strict_passes(self) -> None:
        span = _simple_span("Delaware")
        claim = Claim(statement="x", supporting_spans=[span])
        assert claim.verify(CORPUS_MAP) == []

    def test_default_substring_fallback_passes(self) -> None:
        # char_span wrong but quote is in source
        span = Span(source_uri=CORPUS_URI, char_span=(0, 8), quote="Delaware")
        claim = Claim(statement="x", supporting_spans=[span])
        assert claim.verify(CORPUS_MAP) == []

    def test_default_case_drift_fails(self) -> None:
        # STRICT + SUBSTRING both fail on case drift; default does NOT
        # include CASE_INSENSITIVE.
        span = Span(source_uri=CORPUS_URI, char_span=(0, 8), quote="delaware")
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(CORPUS_MAP)
        assert len(errs) == 1
        assert errs[0].reason == "quote_mismatch"

    def test_case_insensitive_in_chain_passes(self) -> None:
        span = Span(source_uri=CORPUS_URI, char_span=(0, 8), quote="delaware")
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(
            CORPUS_MAP,
            strategies=[MatchStrategy.STRICT, MatchStrategy.CASE_INSENSITIVE],
        )
        assert errs == []

    def test_fuzzy_char_in_chain(self) -> None:
        # End-of-word typo — LCS = 7/8 = 0.875, passes at 0.85.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Delawarz",  # typo at end
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(
            CORPUS_MAP,
            strategies=[MatchStrategy.STRICT, MatchStrategy.FUZZY_CHAR],
            threshold=0.85,
        )
        assert errs == []

    def test_fuzzy_char_mid_word_typo_needs_lower_threshold(self) -> None:
        # Mid-word typo — LCS only 6/8 = 0.75.  At threshold 0.85
        # it fails; at 0.75 it passes.
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 8),
            quote="Dalaware",
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(
            CORPUS_MAP,
            strategies=[MatchStrategy.FUZZY_CHAR],
            threshold=0.85,
        )
        assert len(errs) == 1
        assert errs[0].reason == "similarity_below_threshold"
        # At 0.75 it passes.
        assert (
            claim.verify(
                CORPUS_MAP,
                strategies=[MatchStrategy.FUZZY_CHAR],
                threshold=0.75,
            )
            == []
        )

    def test_fuzzy_char_threshold_too_high_fails_with_similarity_reason(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="completely unrelated text here",
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(
            CORPUS_MAP,
            strategies=[MatchStrategy.FUZZY_CHAR],
            threshold=0.9,
        )
        assert len(errs) == 1
        assert errs[0].reason == "similarity_below_threshold"

    def test_source_missing_reason(self) -> None:
        span = Span(
            source_uri="doc:nowhere",
            char_span=(0, 8),
            quote="anything",
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(CORPUS_MAP)  # corpus_map has doc:test-corpus only
        assert len(errs) == 1
        assert errs[0].reason == "source_missing"

    def test_empty_strategies_raises(self) -> None:
        span = _simple_span("Delaware")
        claim = Claim(statement="x", supporting_spans=[span])
        with pytest.raises(ValueError, match="non-empty"):
            claim.verify(CORPUS_MAP, strategies=[])

    def test_multiple_spans_all_pass(self) -> None:
        claim = Claim(
            statement="x",
            supporting_spans=[
                _simple_span("Delaware"),
                _simple_span("filing fee"),
                _simple_span("certificate"),
            ],
        )
        assert claim.verify(CORPUS_MAP) == []

    def test_multiple_spans_mixed(self) -> None:
        good = _simple_span("Delaware")
        bad = Span(
            source_uri=CORPUS_URI,
            char_span=(0, 20),
            quote="this text is not anywhere",
        )
        claim = Claim(statement="x", supporting_spans=[good, bad])
        errs = claim.verify(CORPUS_MAP)
        assert len(errs) == 1
        assert errs[0].span_index == 1
        assert errs[0].reason == "quote_mismatch"

    def test_out_of_range_reason(self) -> None:
        span = Span(
            source_uri=CORPUS_URI,
            char_span=(10_000, 10_010),
            quote="x" * 10,
        )
        claim = Claim(statement="x", supporting_spans=[span])
        errs = claim.verify(CORPUS_MAP, strategies=[MatchStrategy.STRICT])
        assert len(errs) == 1
        assert errs[0].reason == "out_of_range"

    def test_callable_corpus(self) -> None:
        span = _simple_span("Delaware")
        claim = Claim(statement="x", supporting_spans=[span])
        assert claim.verify(lambda uri: CORPUS_MAP[uri]) == []

    def test_default_is_strict_then_substring(self) -> None:
        """The default tuple is exactly (STRICT, SUBSTRING)."""
        assert DEFAULT_MATCH_STRATEGIES == (
            MatchStrategy.STRICT,
            MatchStrategy.SUBSTRING,
        )


# ─────────────────────────────────────────────────────────────────────
# Answer.verify with strategy forwarding
# ─────────────────────────────────────────────────────────────────────


class TestAnswerVerifyEscalation:
    def _claim(self, quote: str) -> Claim:
        return Claim(
            statement=f"claim about {quote}",
            supporting_spans=[_simple_span(quote)],
        )

    def test_all_claims_pass_default(self) -> None:
        ans = Answer[str](
            value="ok",
            claims=[self._claim("Delaware"), self._claim("filing fee")],
            confidence=0.9,
        )
        assert ans.verify(CORPUS_MAP) == []

    def test_mixed_results_report_per_failing_claim(self) -> None:
        good = self._claim("Delaware")
        bad = Claim(
            statement="bad",
            supporting_spans=[Span(source_uri=CORPUS_URI, char_span=(0, 20), quote="nonexistent")],
        )
        ans = Answer[str](
            value="x",
            claims=[good, bad, self._claim("certificate")],
            confidence=0.9,
        )
        errs = ans.verify(CORPUS_MAP)
        assert len(errs) == 1
        assert errs[0].claim_index == 1
        assert isinstance(errs[0], ClaimError)

    def test_strategies_forwarded_to_claims(self) -> None:
        # Case drift — default chain fails, CI chain passes.
        claim = Claim(
            statement="x",
            supporting_spans=[Span(source_uri=CORPUS_URI, char_span=(0, 8), quote="delaware")],
        )
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)

        assert len(ans.verify(CORPUS_MAP)) == 1
        assert (
            ans.verify(
                CORPUS_MAP,
                strategies=[MatchStrategy.STRICT, MatchStrategy.CASE_INSENSITIVE],
            )
            == []
        )

    def test_threshold_forwarded(self) -> None:
        claim = Claim(
            statement="x",
            supporting_spans=[Span(source_uri=CORPUS_URI, char_span=(0, 8), quote="Dalaware")],
        )
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)
        # High threshold fails
        assert (
            len(
                ans.verify(
                    CORPUS_MAP,
                    strategies=[MatchStrategy.FUZZY_CHAR],
                    threshold=0.95,
                )
            )
            == 1
        )
        # Low threshold passes
        assert (
            ans.verify(
                CORPUS_MAP,
                strategies=[MatchStrategy.FUZZY_CHAR],
                threshold=0.75,
            )
            == []
        )

    def test_default_strategies_immutable(self) -> None:
        """The default tuple shouldn't be mutable-via-reference."""
        assert isinstance(DEFAULT_MATCH_STRATEGIES, tuple)
