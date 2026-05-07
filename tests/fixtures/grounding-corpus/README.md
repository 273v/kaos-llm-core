# Grounding Corpus Fixture

Public-domain or synthetic-illustrative short documents used by `test_grounding_e2e.py`
and `scripts/grounding-calibration.py` to measure grounding quality of LLM outputs.

All files are **ASCII-only** on purpose: the grounding verifier's
`_normalize_unicode` path only runs for `NORMALIZED_TOKEN` strategy, so curly
quotes / em-dashes would create spurious FUZZY_* failures unrelated to model
quality. See `docs/design/grounding-actual-state.md §4`.

## Documents

| File | URI | Source |
|------|-----|--------|
| `delaware-gcl.txt` | `doc:grounding/delaware-gcl` | Illustrative Delaware GCL paraphrase |
| `first-amendment.txt` | `doc:grounding/first-amendment` | US Constitution, First Amendment (public domain) |
| `rfc-2119.txt` | `doc:grounding/rfc-2119` | RFC 2119 keyword definitions (IETF, public) |
| `fair-use-107.txt` | `doc:grounding/fair-use-107` | 17 USC 107 preamble (US Code, public domain) |
| `rule-10b-5.txt` | `doc:grounding/rule-10b-5` | 17 CFR 240.10b-5 (federal regulation, public) |
| `apollo-11.txt` | `doc:grounding/apollo-11` | Apollo 11 mission timeline (NASA fact sheet paraphrase) |
| `gdpr-art-17.txt` | `doc:grounding/gdpr-art-17` | GDPR Article 17 (EU public domain legislative text) |
| `voyager-fact-sheet.txt` | `doc:grounding/voyager` | Voyager mission facts (NASA public domain) |
| `miranda-holding.txt` | `doc:grounding/miranda` | Miranda v. Arizona, 384 U.S. 436 (1966) holding excerpt |
| `nist-password-guidance.txt` | `doc:grounding/nist-password` | NIST SP 800-63B memorized-secret requirements |

## Golden Q/A set

`grounding-questions.jsonl` — 20 questions:

- **10 answerable** — each has exactly one supporting document. A correct model
  returns `Answer[str]` with at least one `Claim` whose `supporting_spans` all
  verify against that document's text.
- **10 unanswerable** — the corpus does not contain the answer. A correct
  model returns `InsufficientEvidence` (or a verifiable refusal phrase).

Schema per line:

```json
{
  "id": "q01",
  "answerable": true,
  "question": "What is the filing fee ...?",
  "expected_doc_id": "doc:grounding/delaware-gcl",
  "expected_answer_hint": "89 dollars / $89",
  "diagnostic_char_span": [100, 120]
}
```

- `expected_doc_id` is only set when `answerable=true`.
- `expected_answer_hint` is prose; it is NOT asserted — only used in the
  calibration report to help a human eyeball miscalibration.
- `diagnostic_char_span` is a hint for the calibration report; the test does
  not compare LLM-returned spans to this interval, since equally correct spans
  can point at different supporting text.
- For `answerable=false`, `reason` describes why the corpus cannot answer.
