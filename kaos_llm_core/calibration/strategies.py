"""Shared match-strategy cascade used by calibration scripts."""

from __future__ import annotations

from kaos_llm_core.signatures import MatchStrategy

# Fuzzy strategies excluded on purpose — only NORMALIZED_TOKEN applies
# Unicode normalization, so with curly-quote / em-dash drift the fuzzy
# strategies would give spurious failures against ASCII corpora. The WS-1
# corpus is ASCII-only; the WS-3.7 corpus contains synthetic HTML/Markdown
# that is also ASCII-leaning but not strictly enforced — this cascade
# tolerates the realistic spread.
DEFAULT_LENIENT_STRATEGIES: tuple[MatchStrategy, ...] = (
    MatchStrategy.STRICT,
    MatchStrategy.SUBSTRING,
    MatchStrategy.CASE_INSENSITIVE,
    MatchStrategy.NORMALIZED_TOKEN,
)


__all__ = ["DEFAULT_LENIENT_STRATEGIES"]
