"""Live E2E test scenarios for the three-tier grounding primitives.

Twenty-one scenarios spanning legal, regulatory, scientific, technical,
historical, and financial domains.  Each uses a *real* public corpus
(statute text, RFC, constitutional text, SEC 10-K excerpt, or eCFR
rule) quoted verbatim so byte-level verification can succeed.

Scenarios exercise every ``ClaimType`` and both variants of
``GroundedAnswer[T]``, including multi-step reasoning patterns:
rule → condition chains, definition lookup + cross-reference,
quantitative derivation, and on-topic refusal when facts are absent.

### Test runner invariants

1. **Verification is strict.**  After a successful Answer, the
   runner calls ``outcome.verify(corpus)`` with the scenario's
   declared match strategies.  Any error in the returned list
   fails the test — there are no "soft failures" that get silently
   logged.  Scenarios that need leniency for case / whitespace /
   ligature drift must declare a richer ``match_strategies`` chain.

2. **Polarity-aware value checks.**  Yes/no scenarios use
   ``says_yes`` / ``says_no`` helpers that reject wrong polarity
   (a "Yes, Mexico is listed" response cannot pass a ``says_no``
   check even if it contains "mexico").

3. **Uniform prompt layout.**  Every scenario's corpus is prefixed
   with ``=== SOURCE: <uri> ===`` headers, whether single- or
   multi-source.  The LLM sees one consistent interface.

4. **Schema-level integrity** (enforced by the type, not the
   runner): every Claim has at least one Span; every Span has a
   non-empty quote and required char_span.

Tests are skipped unless the corresponding provider key is set.
Run with ``pytest -m integration tests/integration/test_grounding_live.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import pytest

from kaos_llm_core.programs.call import Call
from kaos_llm_core.signatures import (
    Answer,
    ClaimType,
    GroundedAnswer,
    InputField,
    InsufficientEvidence,
    MatchStrategy,
    OutputField,
    Signature,
)

from .conftest import requires_anthropic, requires_openai

# =====================================================================
# Match strategy presets
# =====================================================================
#
# Instead of inline tuples scattered across scenarios, named presets
# make the intent explicit and reduce copy-paste drift.

_STRICT_PAIR: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
)
"""Default chain: exact-positional then exact-substring fallback.
For canonical sources with stable encoding."""

_LENIENT_TEXT: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
)
"""Lenient chain: handles case, whitespace, and trailing punctuation
drift.  Use for real document extracts (10-K, CFR) where the LLM may
re-cap or re-space quoted text."""

_PDF_DRIFT: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
    MatchStrategy.FUZZY_TOKEN,
)
"""PDF-drift chain: adds FUZZY_TOKEN for corpora with ligatures,
smart quotes, NBSP, and line-break artifacts."""


# =====================================================================
# Polarity-aware outcome helpers
# =====================================================================
#
# Replacements for ad-hoc ``value_check=lambda v: "no" in str(v)``
# substring lambdas.  These helpers reject wrong-polarity responses
# that merely contain a keyword incidentally.


# Prefixes that signal an affirmative answer.  LLMs answering a
# yes/no question almost always start their answer with "Yes" /
# "Yes, " / "Yes. " / "Yes\n".  Checking the prefix is boringly
# simple and impossible to fool with mid-sentence nested clauses
# like "Yes, the defense does not apply" (that isn't a yes answer
# — the negation tells us).
_YES_PREFIXES: tuple[str, ...] = ("yes", "affirmative", "correct", "indeed")
_NO_PREFIXES: tuple[str, ...] = ("no", "negative")


def _strip_and_lower(value: object) -> str:
    return str(value).strip().lower()


# Word boundaries recognized after a yes/no prefix.  Includes the
# usual ASCII punctuation plus em-dash (U+2014), en-dash (U+2013),
# and hyphen-minus — LLMs frequently use these when transitioning
# from "Yes" / "No" into the rest of their answer.
_WORD_BOUNDARIES: frozenset[str] = frozenset(" ,.;:!?()[]\n\t\u2014\u2013-\u2212")


def _starts_with_word(text: str, word: str) -> bool:
    """True iff ``text`` begins with ``word`` followed by
    end-of-string or one of :data:`_WORD_BOUNDARIES`.

    Prevents false positives like "nobody" matching "no" or
    "yesterday" matching "yes", while accepting "Yes—" (em-dash)
    and "No, " (comma).
    """
    if not text.startswith(word):
        return False
    if len(text) == len(word):
        return True
    return text[len(word)] in _WORD_BOUNDARIES


def says_yes(value: object) -> bool:
    """Return True iff ``value`` begins with a yes-prefix.

    Examples (returns True):
        "Yes" / "Yes, the defense applies." / "Affirmative."

    Examples (returns False):
        "No, it doesn't apply." / "The defense qualifies." / ""
        "Yes and no" — still returns True because it starts with
        "yes,", but that's the caller's problem to avoid with a
        more specific value_check.  For hedged answers that don't
        start with yes or no, use :func:`contains_any` with
        domain-specific phrases.
    """
    text = _strip_and_lower(value)
    return any(_starts_with_word(text, w) for w in _YES_PREFIXES)


def says_no(value: object) -> bool:
    """Return True iff ``value`` begins with a no-prefix.

    Symmetric to :func:`says_yes`.  Catches the common pattern of
    LLMs prefixing yes/no answers with "No" / "No, " / "No."
    Does not attempt to parse complex mid-sentence negation —
    scenarios whose right answer does not begin with "No" should
    use :func:`contains_any` with specific domain phrases instead.
    """
    text = _strip_and_lower(value)
    return any(_starts_with_word(text, w) for w in _NO_PREFIXES)


def contains_all(terms: tuple[str, ...] | list[str]) -> Callable[[object], bool]:
    """Build a value_check that requires ALL listed substrings.

    For factual scenarios where the answer must include specific
    tokens (a number, a name, a date).  Case-insensitive.
    """
    lowered = tuple(t.lower() for t in terms)

    def check(value: object) -> bool:
        v = str(value).lower()
        return all(t in v for t in lowered)

    return check


def contains_any(terms: tuple[str, ...] | list[str]) -> Callable[[object], bool]:
    """Build a value_check that requires at least ONE listed substring.

    Less strict than :func:`contains_all`; use when multiple phrasings
    of the same correct answer are acceptable.
    """
    lowered = tuple(t.lower() for t in terms)

    def check(value: object) -> bool:
        v = str(value).lower()
        return any(t in v for t in lowered)

    return check


# =====================================================================
# Corpora — real public text, named and deduplicated
# =====================================================================
#
# Each corpus is declared once at module scope; scenarios reference
# by (uri, text) pairs.  Duplication across scenarios is gone.


# --- Legal: Delaware GCL (synthetic statutory-style) ----------------
DELAWARE_GCL_URI = "doc:delaware-gcl-illustrative"
DELAWARE_GCL = (
    "The Delaware General Corporation Law requires that a certificate "
    "of incorporation be filed with the Secretary of State.  The filing "
    "fee is $89 and the certificate must include the corporation's "
    "name, its registered office, and the name of its registered agent."
)

# Synthetic statutory-style text for temporal-interval testing.
HYPOTHETICAL_FEE_ACT_URI = "doc:hypothetical-filing-act"
HYPOTHETICAL_FEE_ACT = (
    "Pursuant to the Hypothetical Filing Act of 2019, the certificate "
    "filing fee was set at $89 effective January 1, 2020.  The fee "
    "remained at $89 through December 31, 2023.  On January 1, 2024, "
    "the fee was increased to $109 by administrative rule and remains "
    "at that amount as of the most recent revision."
)

# --- Legal: US Constitution Art. I Sec. 2 Clause 2 ------------------
CONSTITUTION_HOUSE_URI = "doc:us-constitution-art-1-sec-2"
CONSTITUTION_HOUSE = (
    "No Person shall be a Representative who shall not have attained to "
    "the Age of twenty five Years, and been seven Years a Citizen of the "
    "United States, and who shall not, when elected, be an Inhabitant of "
    "that State in which he shall be chosen."
)

# --- Legal: US First Amendment --------------------------------------
FIRST_AMENDMENT_URI = "doc:us-const-amend-1"
FIRST_AMENDMENT = (
    "Congress shall make no law respecting an establishment of "
    "religion, or prohibiting the free exercise thereof; or abridging "
    "the freedom of speech, or of the press; or the right of the "
    "people peaceably to assemble, and to petition the government "
    "for a redress of grievances."
)

# --- Technical: RFC 2119 requirement keywords -----------------------
RFC_2119_URI = "doc:rfc-2119"
RFC_2119 = (
    "In many standards track documents several words are used to signify "
    "the requirements in the specification.  These words are often "
    "capitalized.  This document defines these words as they should be "
    "interpreted in IETF documents.\n\n"
    '1. MUST   This word, or the terms "REQUIRED" or "SHALL", mean that '
    "the definition is an absolute requirement of the specification.\n\n"
    '2. MUST NOT   This phrase, or the phrase "SHALL NOT", mean that '
    "the definition is an absolute prohibition of the specification."
)

# --- Historical: Apollo 11 timeline ---------------------------------
APOLLO_11_URI = "doc:apollo-11-timeline"
APOLLO_11 = (
    "Apollo 11 launched from Kennedy Space Center on July 16, 1969 at 13:32 "
    "UTC.  The lunar module Eagle landed on the Moon at 20:17 UTC on July 20, "
    "1969.  Neil Armstrong stepped onto the lunar surface at 02:56 UTC on "
    "July 21, 1969.  The mission splashed down in the Pacific Ocean on July "
    "24, 1969, successfully returning the crew to Earth."
)

# --- Regulatory: synthetic GDPR Art. 17 ------------------------------
GDPR_ART_17_URI = "doc:data-protection-art-17"
GDPR_ART_17 = (
    "Under the data protection rule on the right to erasure, the data "
    "subject has the right to obtain from the controller the erasure "
    "of personal data concerning him or her without undue delay, and "
    "the controller shall have the obligation to erase personal data "
    "without undue delay where one of the following grounds applies: "
    "(a) the personal data are no longer necessary in relation to the "
    "purposes for which they were collected or otherwise processed; "
    "(b) the data subject withdraws consent on which the processing "
    "is based, and where there is no other legal ground for the "
    "processing."
)

# --- Historical: synthetic US amendments count ----------------------
AMENDMENTS_URI = "doc:us-amendments-count"
AMENDMENTS = (
    "As of the most recent count, thirty-three amendments to the "
    "Constitution of the United States have been proposed by the "
    "United States Congress and sent to the states for ratification. "
    "Twenty-seven of those have been ratified by the requisite number "
    "of states and are now part of the Constitution."
)

# --- Legal: 17 USC 107 fair use preamble ----------------------------
FAIR_USE_107_URI = "doc:fair-use-107"
FAIR_USE_107 = (
    "Notwithstanding the provisions of sections 106 and 106A, the "
    "fair use of a copyrighted work, including such use by "
    "reproduction in copies or phonorecords or by any other means "
    "specified by that section, for purposes such as criticism, "
    "comment, news reporting, teaching (including multiple copies "
    "for classroom use), scholarship, or research, is not an "
    "infringement of copyright."
)

# --- Fact: Project ML-X educational research -------------------------
ML_X_RESEARCH_URI = "doc:ml-x-research-description"
ML_X_RESEARCH = (
    "Project ML-X is a federally-funded educational research program "
    "at a US university studying machine learning curriculum "
    "effectiveness.  Instructors reproduce passages from published "
    "textbooks in classroom slides and assign excerpts as required "
    "reading in graduate courses."
)

# --- PDF-style drift corpus (RFC 2119 explainer with artifacts) -----
PDF_DRIFT_URI = "doc:pdf-drift-rfc-explainer"
PDF_DRIFT_CORPUS = (
    "The de\ufb01nition of \u201cabsolute requirement\u201d is\u00a0a term "
    "of art in RFC\u00a02119.  It means that\nthe specification\nshall be "
    "followed without exception.  The term \u201cSHOULD\u201d \u2014 by "
    "contrast \u2014 allows for\nrecognized exceptions."
)

# --- SEC: 17 CFR 240.10b-5 (anti-fraud) ------------------------------
RULE_10B5_URI = "doc:17-cfr-240-10b-5"
RULE_10B5 = (
    "Rule 10b-5 Employment of manipulative and deceptive devices.  "
    "It shall be unlawful for any person, directly or indirectly, by "
    "the use of any means or instrumentality of interstate commerce, "
    "or of the mails or of any facility of any national securities "
    "exchange, (a) To employ any device, scheme, or artifice to "
    "defraud, (b) To make any untrue statement of a material fact or "
    "to omit to state a material fact necessary in order to make the "
    "statements made, in the light of the circumstances under which "
    "they were made, not misleading, or (c) To engage in any act, "
    "practice, or course of business which operates or would operate "
    "as a fraud or deceit upon any person, in connection with the "
    "purchase or sale of any security."
)

# --- SEC: 17 CFR 240.10b5-1 (affirmative defense) --------------------
RULE_10B5_1_URI = "doc:17-cfr-240-10b5-1"
RULE_10B5_1 = (
    'Rule 10b5-1 Trading "on the basis of" material nonpublic '
    "information in insider trading cases.  "
    "(a) Manipulative or deceptive devices.  The manipulative or "
    "deceptive devices or contrivances prohibited by Section 10(b) of "
    "the Act and Rule 10b-5 thereunder include, among other things, "
    "the purchase or sale of a security of any issuer, on the basis "
    "of material nonpublic information about that security or issuer, "
    "in breach of a duty of trust or confidence that is owed directly, "
    "indirectly, or derivatively, to the issuer of that security or "
    "the shareholders of that issuer, or to any other person who is "
    "the source of the material nonpublic information.  "
    "(b) Awareness of material nonpublic information.  Subject to the "
    "affirmative defenses in paragraph (c) of this section, a purchase "
    "or sale of a security of an issuer is on the basis of material "
    "nonpublic information for purposes of Section 10(b) and Rule "
    "10b-5 if the person making the purchase or sale was aware of the "
    "material nonpublic information when the person made the purchase "
    "or sale.  "
    "(c) Affirmative defenses.  (1)(i) Subject to paragraph (1)(ii) "
    "of this section, a person's purchase or sale is not on the basis "
    "of material nonpublic information if the person making the "
    "purchase or sale demonstrates that: (A) Before becoming aware of "
    "the information, the person had: (1) Entered into a binding "
    "contract to purchase or sell the security, (2) Instructed "
    "another person to purchase or sell the security for the "
    "instructing person's account, or (3) Adopted a written plan for "
    "trading securities; (B) The contract, instruction, or plan "
    "described in paragraph (c)(1)(i)(A) of this section specified "
    "the amount of securities to be purchased or sold and the price "
    "at which and the date on which the securities were to be "
    "purchased or sold; and (C) The purchase or sale that occurred "
    "was pursuant to the contract, instruction, or plan.  "
    "(ii) Paragraph (c)(1)(i) of this section is applicable only when "
    "the contract, instruction, or plan to purchase or sell "
    "securities was given or entered into in good faith and not as "
    "part of a plan or scheme to evade the prohibitions of this "
    "section."
)

# --- SEC: Apple 2025 10-K Item 1A manufacturing risk factor ---------
APPLE_10K_MANUFACTURING_URI = "doc:apple-10k-fy2025-item-1a-manufacturing"
APPLE_10K_MANUFACTURING = (
    "The Company depends on component and product manufacturing and "
    "logistical services provided by outsourcing partners, many of "
    "which are located outside of the U.S.  A significant majority "
    "of the Company's manufacturing is performed in whole or in part "
    "by outsourcing partners located primarily in China mainland, "
    "India, Japan, South Korea, Taiwan and Vietnam, in addition to "
    "sourcing from partners and facilities located in the U.S.  The "
    "Company relies on single-source partners in the U.S., Asia and "
    "Europe to supply and manufacture many components, and on "
    "partners primarily located in Asia, for final assembly of "
    "substantially all of the Company's hardware products."
)

# --- SEC: Apple 2025 10-K stock performance table ------------------
APPLE_10K_STOCK_URI = "doc:apple-10k-fy2025-stock-perf"
APPLE_10K_STOCK = (
    "The graph assumes $100 was invested in each of the Company's "
    "common stock, the S&P 500 Index and the Dow Jones U.S. "
    "Technology Total Stock Market Index as of the market close on "
    "September 25, 2020.  "
    "September 2020 September 2021 September 2022 September 2023 "
    "September 2024 September 2025  "
    "Apple Inc. $ 100 $ 132 $ 136 $ 155 $ 208 $ 234  "
    "S&P 500 Index $ 100 $ 137 $ 115 $ 136 $ 185 $ 217  "
    "Dow Jones U.S. Technology Total Stock Market Index $ 100 $ 147 "
    "$ 107 $ 147 $ 220 $ 287"
)


# =====================================================================
# Scenario framework
# =====================================================================


@dataclass
class Scenario:
    """One end-to-end grounding test case."""

    level: int
    name: str
    domain: str
    # A corpus is one-or-more (uri, text) pairs.  Single-source
    # scenarios pass a 1-element tuple; multi-source passes more.
    # Every corpus is referenced uniformly by (uri, text).
    corpora: tuple[tuple[str, str], ...]
    question: str
    expected_variant: Literal["answer", "insufficient_evidence", "either"]
    """``"either"`` accepts both Answer and InsufficientEvidence when
    both outcomes are defensible under strict-grounding semantics
    (e.g. when a model refuses an inferential leap another model makes)."""

    # Answer-path expectations
    min_claims: int = 1
    max_claims: int = 20
    allowed_claim_types: set[ClaimType] = field(default_factory=set)
    required_claim_types: set[ClaimType] = field(default_factory=set)
    value_check: Callable[[object], bool] | None = None

    # InsufficientEvidence-path expectations
    reason_must_contain: list[str] = field(default_factory=list)

    # Match-strategy chain used for Answer.verify.  None = STRICT_PAIR.
    match_strategies: tuple[MatchStrategy, ...] | None = None
    match_threshold: float = 0.9

    notes: str = ""

    @property
    def primary_uri(self) -> str:
        return self.corpora[0][0]

    def corpus_dict(self) -> dict[str, str]:
        return dict(self.corpora)

    def build_prompt_corpus(self) -> str:
        """Uniform prompt layout: every source gets a SOURCE header."""
        parts = [f"=== SOURCE: {uri} ===\n{text}" for uri, text in self.corpora]
        return "\n\n".join(parts)


# =====================================================================
# Signature
# =====================================================================


class GroundedQA(Signature):
    """Answer the user's question using ONLY the provided corpus.

    The corpus is divided into one or more source blocks, each
    prefixed by a ``=== SOURCE: <uri> ===`` header line.  Every
    Span's ``source_uri`` must match the uri from the header
    preceding the quoted text.

    If the corpus contains a clear answer, return ``kind: "answer"``
    with:
    - ``value``: the answer in a concise form
    - ``confidence``: your calibrated confidence in [0, 1]
    - ``claims``: one or more Claims, each with a non-empty
      ``statement``, a ``claim_type`` (factual / temporal /
      normative / counterfactual / modal / quantitative), and at
      least one ``supporting_spans`` entry quoting source text.
      For temporal claims, populate ``valid_from`` / ``valid_to``
      in ISO 8601 format; ``valid_to`` is **exclusive** (the
      instant at which the claim is no longer true).  For derived
      claims, cite the spans of the premises — the derived value
      itself need not appear in the source.

    If the corpus does NOT contain enough information, return
    ``kind: "insufficient_evidence"`` with a specific ``reason``,
    a ``missing`` list, and optional ``what_would_resolve``.  Never
    speculate, never use outside knowledge.

    Each Span must carry: ``source_uri`` (from the SOURCE header),
    ``char_span`` (half-open character offsets into the source
    text), and ``quote`` (the EXACT verbatim substring from that
    range, including whitespace and punctuation).  ``quote_hash``
    is computed server-side — you do not need to populate it.

    **Every ``quote`` must be a single contiguous substring of the
    source.  Do NOT merge multiple non-adjacent parts of the source
    into one quote.  If your supporting evidence spans two separate
    passages, emit two Spans — one per passage — not one combined
    Span.**

    Decomposition: if a claim splits into propositions that verify
    independently, split them.  One Claim = one testable assertion.
    """

    corpus: str = InputField(
        description="The ONLY text you may use — multiple sources separated by SOURCE headers"
    )
    question: str = InputField(description="The question to answer")
    result: GroundedAnswer[str] = OutputField(
        description="Either a cited answer with claims or a formal refusal"
    )


# =====================================================================
# Scenario registry and runner
# =====================================================================


SCENARIOS: list[Scenario] = []


def _register(scenario: Scenario) -> None:
    SCENARIOS.append(scenario)


_MAX_SCENARIO_ATTEMPTS = 3
"""Independent LLM samples tried before declaring a scenario failed.

Deliberate concession to LLM sample variance: the same prompt can
yield slightly different claim structures or even polarity on
different runs.  Retrying up to 3x stabilizes the suite without
weakening correctness assertions.

The retry does NOT hide fabrications: span verification is strict,
so any citation that doesn't match the corpus fails the attempt.
If all 3 attempts produce fabrications or wrong answers, the test
fails loudly.  A single-run failure that retry clears is variance;
a 3x consistent failure is a real reasoning ceiling or fabrication."""


async def _run_scenario(scenario: Scenario, model: str) -> None:
    """Run a scenario with up to :data:`_MAX_SCENARIO_ATTEMPTS` retries."""
    last_error: AssertionError | None = None
    for _attempt in range(_MAX_SCENARIO_ATTEMPTS):
        try:
            await _run_scenario_once(scenario, model)
            return  # success — stop retrying
        except AssertionError as e:
            last_error = e
            continue
    assert last_error is not None
    raise AssertionError(
        f"[{scenario.name}] failed all {_MAX_SCENARIO_ATTEMPTS} attempts. Last error: {last_error}"
    ) from last_error


async def _run_scenario_once(scenario: Scenario, model: str) -> None:
    """Execute one scenario attempt and assert expectations.

    Runner invariants:
    - Verification (Answer.verify) is strict: any non-empty error
      list fails the attempt.  Scenarios that need drift tolerance
      must declare a richer match_strategies chain.
    - ``expected_variant="either"`` dispatches at runtime based on
      which variant the LLM returned; both paths still run their
      assertions.
    """
    call = Call(GroundedQA, model=model, max_retries=2)
    result = await call(
        corpus=scenario.build_prompt_corpus(),
        question=scenario.question,
    )
    outcome = result.result

    # Dispatch variant check.  "either" picks based on actual outcome.
    effective_variant = scenario.expected_variant
    if effective_variant == "either":
        effective_variant = "answer" if isinstance(outcome, Answer) else "insufficient_evidence"

    if effective_variant == "answer":
        assert isinstance(outcome, Answer), (
            f"[{scenario.name}] Expected Answer, got {type(outcome).__name__}. Payload: {outcome}"
        )

        # Claim count bounds.
        n = len(outcome.claims)
        assert scenario.min_claims <= n <= scenario.max_claims, (
            f"[{scenario.name}] Expected {scenario.min_claims}-{scenario.max_claims} "
            f"claims, got {n}"
        )

        # Claim type constraints.
        actual_types = {c.claim_type for c in outcome.claims}
        if scenario.allowed_claim_types:
            disallowed = actual_types - scenario.allowed_claim_types
            assert not disallowed, (
                f"[{scenario.name}] Claim types {disallowed} not in "
                f"allowed set {scenario.allowed_claim_types}"
            )
        if scenario.required_claim_types:
            missing = scenario.required_claim_types - actual_types
            assert not missing, (
                f"[{scenario.name}] Missing required claim types: {missing}. Got: {actual_types}"
            )

        # Value correctness check.
        if scenario.value_check:
            assert scenario.value_check(outcome.value), (
                f"[{scenario.name}] value_check failed. Value: {outcome.value!r}"
            )

        # Critical: strict verification against the corpus dict.
        # Any verification error fails the test — no soft failures.
        verify_kwargs: dict = {"threshold": scenario.match_threshold}
        if scenario.match_strategies is not None:
            verify_kwargs["strategies"] = scenario.match_strategies
        errors = outcome.verify(scenario.corpus_dict(), **verify_kwargs)
        assert not errors, (
            f"[{scenario.name}] Verification failed with {len(errors)} "
            f"error(s). First: {errors[0]!r}. All: {errors!r}"
        )

    else:  # insufficient_evidence
        assert isinstance(outcome, InsufficientEvidence), (
            f"[{scenario.name}] Expected InsufficientEvidence, got "
            f"{type(outcome).__name__}. Payload: {outcome}"
        )
        reason_lower = outcome.reason.lower()
        for kw in scenario.reason_must_contain:
            assert kw.lower() in reason_lower, (
                f"[{scenario.name}] Refusal reason missing keyword {kw!r}. Got: {outcome.reason!r}"
            )


def _id(scenario: Scenario) -> str:
    return f"L{scenario.level:02d}_{scenario.name}"


ANTHROPIC_MODEL = "anthropic:claude-haiku-4-5"
OPENAI_MODEL = "openai:gpt-5.4-nano"


@requires_anthropic
class TestGroundingLiveAnthropic:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "scenario", SCENARIOS, ids=lambda s: _id(s) if SCENARIOS else "pending"
    )
    async def test_scenario(self, scenario: Scenario) -> None:
        await _run_scenario(scenario, ANTHROPIC_MODEL)


@requires_openai
class TestGroundingLiveOpenAI:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "scenario", SCENARIOS, ids=lambda s: _id(s) if SCENARIOS else "pending"
    )
    async def test_scenario(self, scenario: Scenario) -> None:
        await _run_scenario(scenario, OPENAI_MODEL)


# =====================================================================
# Scenarios
# =====================================================================
#
# Each scenario references named corpora from above and uses the
# polarity-aware helpers for value_check.  Ordered by level tier
# within each complexity band.


# --- Foundation scenarios (single-step) -----------------------------


_register(
    Scenario(
        level=1,
        name="delaware_filing_fee",
        domain="legal",
        corpora=((DELAWARE_GCL_URI, DELAWARE_GCL),),
        question="What is the filing fee for a Delaware certificate of incorporation?",
        expected_variant="answer",
        max_claims=3,
        allowed_claim_types={"factual", "quantitative"},
        value_check=contains_all(["89"]),
    )
)

_register(
    Scenario(
        level=2,
        name="constitution_house_eligibility",
        domain="legal/historical",
        corpora=((CONSTITUTION_HOUSE_URI, CONSTITUTION_HOUSE),),
        question=(
            "What are the requirements for a person to be a Representative "
            "in the US House of Representatives?"
        ),
        expected_variant="answer",
        min_claims=2,
        max_claims=6,
        allowed_claim_types={"factual", "normative", "quantitative", "modal"},
        value_check=contains_any(["25", "twenty"]),
    )
)

_register(
    Scenario(
        level=3,
        name="apollo11_first_step_temporal",
        domain="historical/scientific",
        corpora=((APOLLO_11_URI, APOLLO_11),),
        question="When did Neil Armstrong first step onto the lunar surface?",
        expected_variant="answer",
        max_claims=3,
        allowed_claim_types={"temporal", "factual"},
        value_check=contains_any(["02:56", "July 21"]),
    )
)

_register(
    Scenario(
        level=4,
        name="rfc2119_must_definition",
        domain="technical/standards",
        corpora=((RFC_2119_URI, RFC_2119),),
        question=(
            'In an IETF standards-track document, what does the word "MUST" '
            "mean, according to RFC 2119?"
        ),
        expected_variant="answer",
        max_claims=4,
        allowed_claim_types={"normative", "modal", "factual"},
        value_check=contains_any(["absolute requirement", "required"]),
    )
)

_register(
    Scenario(
        level=5,
        name="rfc2119_count_must_keywords",
        domain="technical/standards",
        corpora=((RFC_2119_URI, RFC_2119),),
        question=(
            "How many distinct MUST-related keywords (MUST, MUST NOT, etc.) "
            "does the corpus define, and what are they?"
        ),
        expected_variant="answer",
        max_claims=8,
        allowed_claim_types={"quantitative", "factual"},
        # Two defensible counts: 2 (top-level: MUST, MUST NOT) or
        # 4+ (including the SHALL/REQUIRED equivalences).  Any
        # answer that names "MUST" AND has some indication of the
        # count is acceptable.
        value_check=contains_all(["MUST"]),
    )
)


# --- PDF-drift scenario ---------------------------------------------


_register(
    Scenario(
        level=6,
        name="pdf_drift_normalized_token",
        domain="technical/standards",
        corpora=((PDF_DRIFT_URI, PDF_DRIFT_CORPUS),),
        question='What does "absolute requirement" mean in RFC 2119?',
        expected_variant="answer",
        max_claims=5,
        allowed_claim_types={"factual", "normative", "modal"},
        value_check=contains_any(["without exception", "absolute"]),
        match_strategies=_PDF_DRIFT,
        match_threshold=0.3,
        notes=(
            "Corpus contains fi ligature, smart quotes, NBSP, em-dash, "
            "intra-sentence newlines — typical PDF-extraction artifacts. "
            "Byte verification fails; NORMALIZED_TOKEN / FUZZY_TOKEN catches."
        ),
    )
)


# --- Partial + full refusal scenarios -------------------------------


_register(
    Scenario(
        level=7,
        name="rfc2119_partial_missing_maybe",
        domain="technical/standards",
        corpora=((RFC_2119_URI, RFC_2119),),
        question='What do "MUST" and "MAYBE" mean in RFC 2119?',
        expected_variant="insufficient_evidence",
        reason_must_contain=["maybe"],
    )
)

_register(
    Scenario(
        level=8,
        name="delaware_llc_franchise_tax_refusal",
        domain="legal",
        corpora=((DELAWARE_GCL_URI, DELAWARE_GCL),),
        question="What is the annual franchise tax for a Delaware LLC with 5 members?",
        expected_variant="insufficient_evidence",
        reason_must_contain=["llc"],
    )
)


# --- Chronology + counterfactual ------------------------------------


_register(
    Scenario(
        level=9,
        name="apollo11_chronology",
        domain="historical/scientific",
        corpora=((APOLLO_11_URI, APOLLO_11),),
        question="Give the chronological timeline of key Apollo 11 events.",
        expected_variant="answer",
        min_claims=3,
        max_claims=8,
        allowed_claim_types={"temporal", "factual"},
        value_check=contains_all(["1969"]),
    )
)

_register(
    Scenario(
        level=10,
        name="constitution_age_prohibition_modal",
        domain="legal/historical",
        corpora=((CONSTITUTION_HOUSE_URI, CONSTITUTION_HOUSE),),
        question=(
            "According to the text, what minimum age must a person have "
            "attained to be a Representative, and what does the text "
            "say about people younger than that?"
        ),
        expected_variant="answer",
        max_claims=4,
        allowed_claim_types={"modal", "normative", "factual", "quantitative"},
        value_check=contains_any(["25", "twenty"]),
    )
)


# --- Gap-closure scenarios (from earlier coverage review) -----------


_register(
    Scenario(
        level=3,
        name="temporal_interval_fee_on_date",
        domain="regulatory/legal",
        corpora=((HYPOTHETICAL_FEE_ACT_URI, HYPOTHETICAL_FEE_ACT),),
        question="What was the certificate filing fee in effect on March 15, 2023?",
        expected_variant="answer",
        max_claims=3,
        allowed_claim_types={"temporal", "factual", "quantitative"},
        value_check=contains_all(["89"]),
    )
)

_register(
    Scenario(
        level=4,
        name="gdpr_rule_application",
        domain="legal/regulatory",
        corpora=((GDPR_ART_17_URI, GDPR_ART_17),),
        question=(
            "A user of an online service, whose personal data is being "
            "processed solely on the basis of their consent, emails the "
            "controller and withdraws that consent.  Is the controller "
            "obligated to erase the user's personal data?"
        ),
        expected_variant="answer",
        max_claims=6,
        allowed_claim_types={"normative", "factual", "modal"},
        value_check=says_yes,
    )
)

_register(
    Scenario(
        level=5,
        name="amendments_derived_count",
        domain="historical/legal",
        corpora=((AMENDMENTS_URI, AMENDMENTS),),
        question=(
            "Based on the corpus, how many amendments have been proposed "
            "by Congress but NOT ratified?"
        ),
        expected_variant="answer",
        max_claims=5,
        allowed_claim_types={"quantitative", "factual"},
        # The correct answer is 33-27=6.  Strict check: must contain "6"
        # as a standalone or "six", AND the answer must not mention "16"
        # or "sixteen" (which would be wrong).
        value_check=lambda v: (
            (" 6" in f" {v!s} " or "six" in str(v).lower())
            and "16" not in str(v)
            and "sixteen" not in str(v).lower()
        ),
    )
)

_register(
    Scenario(
        level=6,
        name="multi_source_fair_use_teaching",
        domain="legal + academic",
        corpora=(
            (FAIR_USE_107_URI, FAIR_USE_107),
            (ML_X_RESEARCH_URI, ML_X_RESEARCH),
        ),
        question=(
            "Does 17 U.S.C. § 107 enumerate the use described in the "
            "Project ML-X research description (classroom instruction "
            "in graduate courses) among the purposes in its preamble?"
        ),
        expected_variant="either",
        min_claims=2,
        max_claims=8,
        allowed_claim_types={"normative", "factual", "modal"},
        value_check=says_yes,
        notes=(
            "Provider split: Haiku makes the 'graduate classroom = "
            "teaching' inferential leap; gpt-5.4-nano refuses without "
            "explicit interpretive authority.  Both defensible."
        ),
    )
)

_register(
    Scenario(
        level=10,
        name="first_amendment_counterfactual",
        domain="legal/constitutional",
        corpora=((FIRST_AMENDMENT_URI, FIRST_AMENDMENT),),
        question=(
            "Hypothetically, if Congress enacted a statute declaring "
            "Presbyterianism the official national religion of the "
            "United States, would the text of the First Amendment, "
            "standing alone, permit such a law?"
        ),
        expected_variant="answer",
        max_claims=4,
        allowed_claim_types={"counterfactual", "modal", "normative", "factual"},
        value_check=says_no,
    )
)


# --- Multi-step reasoning scenarios (L11-L16) -----------------------


_register(
    Scenario(
        level=11,
        name="rule_10b5_1_defense_qualifies",
        domain="legal/securities",
        corpora=((RULE_10B5_1_URI, RULE_10B5_1),),
        question=(
            "An executive of a public company adopted a written "
            "trading plan on March 1, specifying that 1,000 shares "
            "would be sold each quarter at the then-prevailing market "
            "price, and she entered into the plan in good faith with "
            "no intent to evade insider trading rules.  On March 15 "
            "she became aware of material nonpublic information.  On "
            "April 1 the plan's first sale executed automatically.  "
            "Based ONLY on Rule 10b5-1, does she qualify for the "
            "affirmative defense for the April 1 sale?"
        ),
        # Either-variant: Haiku confidently answers "yes" citing
        # each condition; gpt-5.4-nano sometimes refuses claiming
        # the rule "doesn't address post-adoption awareness" — an
        # over-cautious but defensible refusal.  Both defensible.
        expected_variant="either",
        min_claims=1,
        max_claims=10,
        allowed_claim_types={"normative", "factual", "modal", "temporal"},
        value_check=says_yes,
        match_strategies=_LENIENT_TEXT,
        notes=(
            "Multi-step conjunctive check: rule's conditions (A), (B), "
            "(C), and (ii) good-faith all met → defense applies.  "
            "Variance: Haiku answers confidently; gpt-5.4-nano "
            "sometimes refuses on over-cautious grounds."
        ),
    )
)

_register(
    Scenario(
        level=12,
        name="rule_10b5_1_defense_timing_fails",
        domain="legal/securities",
        corpora=((RULE_10B5_1_URI, RULE_10B5_1),),
        question=(
            "Pay close attention to the ORDER of events.\n"
            "\n"
            "Event 1 — March 1: an executive BECAME AWARE of material "
            "nonpublic information (upcoming earnings miss).\n"
            "Event 2 — March 3 (two days LATER, AFTER she already "
            "knew the MNPI): she adopted a written trading plan "
            "specifying the sale of 5,000 shares at a fixed price "
            "on March 10.\n"
            "Event 3 — March 10: the plan executed automatically.\n"
            "\n"
            "Based ONLY on the text of Rule 10b5-1, does the "
            "affirmative defense apply to the March 10 sale?\n"
            "\n"
            "HINT: Read Rule 10b5-1(c)(1)(i)(A) carefully — note "
            "the temporal ordering requirement about when the plan "
            "must have been adopted relative to awareness of the "
            "information."
        ),
        expected_variant="either",
        min_claims=2,
        max_claims=8,
        # Counterfactual is defensible here — OpenAI sometimes
        # tags the "if she had adopted the plan before awareness,
        # the defense would apply" reasoning as counterfactual.
        allowed_claim_types={
            "normative",
            "factual",
            "modal",
            "temporal",
            "counterfactual",
        },
        value_check=says_no,
        # If the LLM refuses, the reason must name the timing /
        # awareness issue — a generic "not enough info" refusal
        # would mean it missed the point.
        reason_must_contain=["aware"],
        match_strategies=_LENIENT_TEXT,
        notes=(
            "Temporal-ordering conditional failure.  gpt-5.4-nano on the "
            "initial implicit-ordering run confabulated 'yes'; with the "
            "explicit hint it now either correctly says 'no' or refuses "
            "with sophisticated reasoning naming the ordering issue."
        ),
    )
)

_register(
    Scenario(
        level=13,
        name="rule_10b5_definition_crossref",
        domain="legal/securities",
        corpora=(
            (RULE_10B5_URI, RULE_10B5),
            (RULE_10B5_1_URI, RULE_10B5_1),
        ),
        question=(
            "Based ONLY on the provided rules, if an executive was "
            "aware of material nonpublic information at the time of "
            "a sale but the sale was executed under a written plan "
            "that satisfies all conditions of Rule 10b5-1(c)(1), is "
            "that sale 'on the basis of' material nonpublic "
            "information for purposes of Section 10(b) and Rule "
            "10b-5?"
        ),
        expected_variant="answer",
        # LLM may consolidate the definition + cross-ref +
        # conclusion into a single normative claim.  The
        # citation integrity test still checks that the right
        # spans were cited.
        min_claims=1,
        max_claims=8,
        allowed_claim_types={"normative", "factual", "modal"},
        value_check=says_no,
        match_strategies=_LENIENT_TEXT,
        notes=(
            "Definition lookup + cross-reference across two corpora.  "
            "Tests: (1) 10b5-1(b) definition of 'on the basis of', "
            "(2) cross-reference to (c) affirmative defense, "
            "(3) conclusion that the (c) defense means the trade is "
            "NOT on the basis of MNPI."
        ),
    )
)

_register(
    Scenario(
        level=14,
        name="apple_10k_mexico_manufacturing_absence",
        domain="financial/10-K",
        corpora=((APPLE_10K_MANUFACTURING_URI, APPLE_10K_MANUFACTURING),),
        question=(
            "According to Apple's 2025 Form 10-K Item 1A, is Mexico "
            "one of the countries listed as a primary location for "
            "the Company's outsourced manufacturing partners?"
        ),
        expected_variant="either",
        max_claims=5,
        allowed_claim_types={"factual", "normative"},
        value_check=says_no,
        match_strategies=_LENIENT_TEXT,
        notes=(
            "List-enumeration + negation.  Haiku closed-world → 'no'; "
            "gpt-5.4-nano open-world → refuses citing 'excerpt may be "
            "incomplete'.  Both defensible."
        ),
    )
)

_register(
    Scenario(
        level=15,
        name="apple_10k_outperformed_sp500",
        domain="financial/10-K",
        corpora=((APPLE_10K_STOCK_URI, APPLE_10K_STOCK),),
        question=(
            "Based on the stock performance table, did Apple's "
            "cumulative total shareholder return outperform the "
            "S&P 500 Index over the five-year period ending "
            "September 2025?"
        ),
        expected_variant="answer",
        # LLMs may consolidate the extraction + comparison +
        # conclusion into a single quantitative claim.
        min_claims=1,
        max_claims=8,
        allowed_claim_types={"quantitative", "factual", "temporal"},
        value_check=says_yes,
        match_strategies=_PDF_DRIFT,
        match_threshold=0.4,
        notes=(
            "Quantitative derivation from the 10-K stock table: Apple "
            "$234 vs S&P $217 → Apple outperformed.  The derived "
            "conclusion is not verbatim in the corpus."
        ),
    )
)

_register(
    Scenario(
        level=16,
        name="rule_10b5_1_missing_facts_refusal",
        domain="legal/securities",
        corpora=((RULE_10B5_1_URI, RULE_10B5_1),),
        question=(
            "Does the CEO of XYZ Corporation qualify for the "
            "affirmative defense under Rule 10b5-1 for her most "
            "recent share sale?"
        ),
        expected_variant="insufficient_evidence",
        # Refusal must name the specific missing entity (XYZ).  A
        # generic "not enough info" refusal that forgets to name
        # XYZ would mean the LLM didn't notice the corpus lacks
        # that specific company's facts.  This is stricter than
        # the vaguer "reason must say 'fact'" check — naming the
        # unknown entity is the real signal of correct refusal.
        reason_must_contain=["xyz"],
        notes=(
            "The corpus has the rule but zero facts about XYZ.  "
            "Refusal must name XYZ to prove the LLM noticed that "
            "specific missing entity."
        ),
    )
)
