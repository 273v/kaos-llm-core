"""Adversarial and robustness tests for the grounding primitives.

These tests exist to prove the core safety guarantees:

1. A fabricated Span (LLM-invented quote that does not exist in the
   corpus) is caught by :meth:`Answer.verify` across every
   :class:`MatchStrategy` — no strategy, however lenient, lets a
   confabulation pass.
2. The polarity-aware helpers ``says_yes`` / ``says_no`` from the
   live integration runner correctly reject wrong-polarity
   responses (the motivating bug for the helpers: "Yes, Mexico is
   listed" must not pass a ``says_no`` check).
3. Verification performance on realistic corpus sizes (10s of KB)
   is acceptable.

If any of these regressions, it means the primitive has lost a
safety property and downstream workflows built on it are unsafe.
"""

from __future__ import annotations

import time

from kaos_llm_core.signatures import (
    Answer,
    Claim,
    MatchStrategy,
    Span,
)
from tests.integration.test_grounding_live import (
    contains_all,
    contains_any,
    says_no,
    says_yes,
)

# =====================================================================
# Polarity-aware value_check helpers
# =====================================================================


class TestSaysYes:
    """``says_yes`` matches LLM answers that begin with a yes-prefix.

    The helper is deliberately prefix-based rather than
    mid-sentence polarity: LLMs answering yes/no questions almost
    always start their answer with "Yes" / "Yes, " / "Yes." / etc.
    Mid-sentence polarity is unreliable in complex nested clauses
    (see the adversarial tests for fabricated-span detection —
    that's the airtight layer).
    """

    def test_plain_yes(self) -> None:
        assert says_yes("Yes") is True

    def test_yes_dot(self) -> None:
        assert says_yes("Yes.") is True

    def test_yes_comma(self) -> None:
        assert says_yes("Yes, the defense qualifies.") is True

    def test_yes_newline(self) -> None:
        assert says_yes("Yes\nThe defense qualifies") is True

    def test_affirmative_prefix(self) -> None:
        assert says_yes("Affirmative.") is True

    def test_correct_prefix(self) -> None:
        assert says_yes("Correct, the defense applies.") is True

    def test_indeed_prefix(self) -> None:
        assert says_yes("Indeed, all conditions are met.") is True

    def test_em_dash_boundary(self) -> None:
        """LLMs frequently use em-dash after a yes/no prefix:
        'Yes—based on the facts given, she qualifies...'"""
        assert says_yes("Yes—based on the facts given, she qualifies.") is True

    def test_hyphen_boundary(self) -> None:
        assert says_yes("Yes - the defense applies.") is True

    def test_leading_whitespace_stripped(self) -> None:
        assert says_yes("  Yes, the defense applies.") is True

    def test_case_insensitive(self) -> None:
        assert says_yes("YES") is True
        assert says_yes("yes") is True
        assert says_yes("Yes") is True

    def test_plain_no_rejected(self) -> None:
        assert says_yes("No") is False

    def test_does_not_apply_rejected(self) -> None:
        assert says_yes("The defense does not apply.") is False

    def test_narrative_without_prefix_rejected(self) -> None:
        """The helper only accepts explicit yes-prefixes.  A
        narrative answer like 'The executive qualifies' returns
        False — the caller should use contains_any for those."""
        assert says_yes("The executive qualifies for the defense.") is False

    def test_empty_rejected(self) -> None:
        assert says_yes("") is False

    def test_no_word_boundary_false_match(self) -> None:
        """'yesterday' and 'yeshiva' should not match 'yes'."""
        assert says_yes("Yesterday the filing was submitted.") is False
        assert says_yes("Yeshiva University announced.") is False

    def test_yes_and_no_ambiguous_but_prefix_wins(self) -> None:
        """'Yes and no' starts with 'yes' so it passes says_yes.
        This is a deliberate simplification — the value_check is a
        soft check; the airtight guarantee is Answer.verify()."""
        assert says_yes("Yes and no") is True


class TestSaysNo:
    """``says_no`` — symmetric prefix-based matcher."""

    def test_plain_no(self) -> None:
        assert says_no("No") is True

    def test_no_dot(self) -> None:
        assert says_no("No.") is True

    def test_no_comma(self) -> None:
        assert says_no("No, the defense does not apply.") is True

    def test_no_with_long_explanation(self) -> None:
        """The L13 motivating case: answer has positive-sounding
        sub-clauses ('satisfies all conditions') but starts with
        'No'.  Must still be recognized as a no answer."""
        assert (
            says_no(
                "No. If the sale was executed pursuant to a written plan "
                "that satisfies all conditions of Rule 10b5-1(c)(1), the "
                "purchase or sale is not on the basis of MNPI."
            )
            is True
        )

    def test_negative_prefix(self) -> None:
        assert says_no("Negative.") is True

    def test_leading_whitespace_stripped(self) -> None:
        assert says_no("  No, Mexico is not listed.") is True

    def test_case_insensitive(self) -> None:
        assert says_no("NO") is True
        assert says_no("No") is True
        assert says_no("no") is True

    def test_plain_yes_rejected(self) -> None:
        assert says_no("Yes") is False

    def test_qualifies_without_no_prefix_rejected(self) -> None:
        """'The defense qualifies' doesn't start with 'no' → not a
        clear no answer.  The caller is expected to use
        contains_any for narrative answers."""
        assert says_no("The defense qualifies.") is False

    def test_yes_mexico_is_listed_rejected(self) -> None:
        """The original motivating bug: 'Yes, Mexico is listed'
        should NOT pass a says_no check.  With prefix-based
        matching, we reject this trivially."""
        assert says_no("Yes, Mexico is listed in Item 1A.") is False

    def test_empty_rejected(self) -> None:
        assert says_no("") is False

    def test_word_boundary_nobody(self) -> None:
        """'Nobody' should not match 'no' prefix."""
        assert says_no("Nobody knows the answer.") is False

    def test_word_boundary_nor(self) -> None:
        """'Nor' and 'not' shouldn't false-trigger."""
        assert says_no("Nor does the defense apply.") is False
        assert says_no("Not applicable.") is False


class TestContainsHelpers:
    """``contains_all`` / ``contains_any`` factual checks."""

    def test_contains_all_requires_every_token(self) -> None:
        check = contains_all(["89", "filing"])
        assert check("The filing fee is $89.") is True
        assert check("The filing fee is $99.") is False  # missing 89
        assert check("$89 is the cost.") is False  # missing filing

    def test_contains_all_is_case_insensitive(self) -> None:
        check = contains_all(["neil", "armstrong"])
        assert check("NEIL ARMSTRONG walked the moon.") is True
        assert check("Buzz Aldrin walked the moon.") is False

    def test_contains_any_requires_one_token(self) -> None:
        check = contains_any(["02:56", "July 21"])
        assert check("Neil stepped out at 02:56 UTC.") is True
        assert check("On July 21, he stepped onto the surface.") is True
        assert check("He walked on the moon.") is False

    def test_contains_any_empty_list(self) -> None:
        check = contains_any([])
        # Empty required set: any() is False, so nothing passes.
        assert check("anything") is False


# =====================================================================
# Adversarial span tests — fabrication detection
# =====================================================================


_CORPUS_URI = "doc:adversarial"
_CORPUS_TEXT = (
    "The quick brown fox jumps over the lazy dog.  "
    "All work and no play makes Jack a dull boy.  "
    "The rain in Spain stays mainly in the plain."
)
_CORPUS_MAP = {_CORPUS_URI: _CORPUS_TEXT}


class TestFabricatedSpanDetection:
    """A Span whose quote does NOT appear in the corpus must fail
    verification under every MatchStrategy.  No strategy, however
    lenient, should accept a fully-fabricated citation.

    Threshold: we use 0.9 for fuzzy strategies.  Lower thresholds
    (< 0.5) intentionally accept anything with non-trivial overlap
    and are the caller's responsibility to set appropriately; the
    framework doesn't guarantee safety at arbitrarily-low thresholds.
    """

    def _fabricated_span(self, quote: str) -> Span:
        # Use a char_span that's valid but whose bytes do not match
        # the fabricated quote.  We're simulating the exact LLM
        # failure mode: "I cite position X, but the text I quote
        # isn't actually at X (or anywhere)."
        return Span(
            source_uri=_CORPUS_URI,
            char_span=(0, min(len(quote), len(_CORPUS_TEXT))),
            quote=quote,
        )

    def _assert_rejected(self, quote: str, strategy: MatchStrategy) -> None:
        span = self._fabricated_span(quote)
        claim = Claim(statement="fabricated", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)
        errors = ans.verify(_CORPUS_MAP, strategies=[strategy], threshold=0.9)
        assert len(errors) == 1, (
            f"Fabricated quote {quote!r} escaped strategy {strategy}: errors={errors}"
        )

    def test_completely_fabricated_rejected_strict(self) -> None:
        self._assert_rejected("The sky is green and cheese.", MatchStrategy.STRICT)

    def test_completely_fabricated_rejected_substring(self) -> None:
        self._assert_rejected("The sky is green and cheese.", MatchStrategy.SUBSTRING)

    def test_completely_fabricated_rejected_case_insensitive(self) -> None:
        self._assert_rejected("THE SKY IS GREEN AND CHEESE", MatchStrategy.CASE_INSENSITIVE)

    def test_completely_fabricated_rejected_normalized_token(self) -> None:
        self._assert_rejected("the sky is green and cheese", MatchStrategy.NORMALIZED_TOKEN)

    def test_completely_fabricated_rejected_fuzzy_char(self) -> None:
        self._assert_rejected(
            "completely fabricated content nowhere present",
            MatchStrategy.FUZZY_CHAR,
        )

    def test_completely_fabricated_rejected_fuzzy_token(self) -> None:
        # FUZZY_TOKEN at threshold 0.9 on fully-disjoint tokens.
        span = self._fabricated_span("antidisestablishmentarian constellations gyroscopes")
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)
        errors = ans.verify(_CORPUS_MAP, strategies=[MatchStrategy.FUZZY_TOKEN], threshold=0.9)
        assert len(errors) == 1


class TestPartialFabrication:
    """A Span whose quote is PART real, PART invented is caught as
    long as the strategy threshold is reasonable."""

    def test_real_prefix_fake_suffix_rejected_strict(self) -> None:
        span = Span(
            source_uri=_CORPUS_URI,
            char_span=(4, 30),
            quote="quick brown fox secretly ate the manuscript",  # real prefix + lie
        )
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)
        errors = ans.verify(_CORPUS_MAP)  # default STRICT+SUBSTRING
        assert len(errors) == 1, f"Partial fabrication escaped default chain: errors={errors}"

    def test_real_prefix_fake_suffix_rejected_case_insensitive(self) -> None:
        span = Span(
            source_uri=_CORPUS_URI,
            char_span=(4, 30),
            quote="QUICK BROWN FOX SECRETLY ATE THE MANUSCRIPT",
        )
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)
        errors = ans.verify(_CORPUS_MAP, strategies=[MatchStrategy.CASE_INSENSITIVE])
        assert len(errors) == 1


class TestWrongSourceUri:
    """Claims pointing to a source_uri not in the corpus dict are
    source_missing, regardless of quote content."""

    def test_unknown_source_fails(self) -> None:
        span = Span(
            source_uri="doc:nonexistent",
            char_span=(0, 10),
            quote="The quick ",  # this quote IS in the real corpus
        )
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)
        errors = ans.verify(_CORPUS_MAP)
        assert len(errors) == 1
        assert errors[0].reason == "source_missing"


# =====================================================================
# Performance on realistic corpus sizes
# =====================================================================


class TestVerificationPerformance:
    """Rough sanity check that verification on realistic corpora
    (10s of KB, roughly what one section of a 10-K might contain)
    completes in well under a second per claim.  Not a strict
    benchmark — just a regression guard so performance doesn't
    silently collapse if kaos-nlp-core's LCS or token_jaccard
    regresses."""

    def _big_corpus(self) -> str:
        # ~50KB of plausible text: a paragraph repeated with variation
        # so every strategy has to do real work.
        chunk = (
            "The Company depends on component and product manufacturing "
            "and logistical services provided by outsourcing partners, "
            "many of which are located outside of the U.S.  A significant "
            "majority of the Company's manufacturing is performed in "
            "whole or in part by outsourcing partners located primarily "
            "in China mainland, India, Japan, South Korea, Taiwan and "
            "Vietnam, in addition to sourcing from partners and "
            "facilities located in the U.S.  "
        )
        return chunk * 100  # ~50KB

    def test_strict_on_50kb_fast(self) -> None:
        corpus_text = self._big_corpus()
        corpus_uri = "doc:big"
        # Locate a known substring and build a real Span for it.
        needle = "single-source partners"
        # Adjust: the needle doesn't appear in our repeated corpus.
        # Use a substring that does appear.
        needle = "significant majority"
        idx = corpus_text.find(needle)
        assert idx > 0
        span = Span.from_source(corpus_uri, corpus_text, (idx, idx + len(needle)))
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)

        start = time.perf_counter()
        errors = ans.verify({corpus_uri: corpus_text})
        elapsed = time.perf_counter() - start

        assert errors == []
        assert elapsed < 0.5, f"STRICT verification took {elapsed:.3f}s (expected < 0.5s)"

    def test_fuzzy_char_on_50kb_fast(self) -> None:
        """FUZZY_CHAR uses longest_common_substring_length which is
        O(n*m) DP in Rust.  On 50KB of corpus with a short quote,
        this should still complete quickly."""
        corpus_text = self._big_corpus()
        corpus_uri = "doc:big"
        span = Span(
            source_uri=corpus_uri,
            char_span=(0, 40),
            quote="A significant majority of manufacturing",  # close but not exact
        )
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)

        start = time.perf_counter()
        errors = ans.verify(
            {corpus_uri: corpus_text},
            strategies=[MatchStrategy.FUZZY_CHAR],
            threshold=0.85,
        )
        elapsed = time.perf_counter() - start
        # Accept either match or no-match; we only care about speed.
        # 1.0s is generous — real corpus would be ~200KB for a full
        # 10-K section; our 50KB should complete in ~10x less time.
        assert elapsed < 1.0, f"FUZZY_CHAR on 50KB took {elapsed:.3f}s (expected < 1.0s)"
        # Report for diagnostic purposes.
        print(f"FUZZY_CHAR 50KB: {elapsed * 1000:.1f}ms, errors={len(errors)}")

    def test_normalized_token_on_50kb_fast(self) -> None:
        corpus_text = self._big_corpus()
        corpus_uri = "doc:big"
        span = Span(
            source_uri=corpus_uri,
            char_span=(0, 40),
            quote="significant MAJORITY of the Company's",
        )
        claim = Claim(statement="x", supporting_spans=[span])
        ans = Answer[str](value="x", claims=[claim], confidence=0.9)

        start = time.perf_counter()
        errors = ans.verify(
            {corpus_uri: corpus_text},
            strategies=[MatchStrategy.NORMALIZED_TOKEN],
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"NORMALIZED_TOKEN on 50KB took {elapsed:.3f}s (expected < 1.0s)"
        print(f"NORMALIZED_TOKEN 50KB: {elapsed * 1000:.1f}ms, errors={len(errors)}")
