# Multi-Format Corpus Fixture (WS-3.7)

10 documents across 5 formats. Used by the WS-3.7 benchmark to prove the
full extractor → `Corpus` → `CorpusIndex` → `RAG` → `Answer` pipeline
works on realistic mixed-format inputs.

All content is US-government origin (public domain) or synthetic text
summarizing public-domain source material. No third-party copyrighted
content is included.

## Inventory

| File | Format | Source | Licensing |
|------|--------|--------|-----------|
| `fda_guidance.pdf` | PDF | Copy of `kaos-pdf/tests/fixtures/kl3m_fda_guidance.pdf` — FDA guidance document | US govt public domain |
| `court_order.pdf` | PDF | Copy of `kaos-pdf/tests/fixtures/casd_court_order.pdf` — SDCA court order | US govt public domain |
| `bcfp_consumer_rights.docx` | DOCX | Copy of `kaos-office/tests/fixtures/docx/bcfp_consumer-rights-summary_2018-09.docx` — CFPB/BCFP consumer rights summary | US govt public domain |
| `uspto_filing.docx` | DOCX | Copy of `kaos-office/tests/fixtures/docx/p2021-203386.docx` — USPTO filing | US govt public domain |
| `ecfr_17_cfr_240_10b5.html` | HTML | Synthetic HTML summarizing 17 CFR 240.10b-5 (Rule 10b-5) | Synthetic; based on US federal regulation (public domain) |
| `scotus_miranda_summary.html` | HTML | Synthetic HTML summarizing Miranda v. Arizona, 384 U.S. 436 (1966) | Synthetic; based on US Supreme Court opinion (public domain) |
| `federal_register_pfas_summary.md` | Markdown | Synthetic markdown summarizing an EPA PFAS drinking-water rule | Synthetic; based on US federal regulation (public domain) |
| `nist_ai_rmf_overview.md` | Markdown | Synthetic markdown summarizing the NIST AI RMF 1.0 | Synthetic; based on NIST publication (US govt public domain) |
| `rfc_2119.txt` | Plain text | Copy of `kaos-llm-core/tests/fixtures/grounding-corpus/rfc-2119.txt` | Original RFC 2119 (public per IETF trust policy) |
| `nist_passwords.txt` | Plain text | Copy of `kaos-llm-core/tests/fixtures/grounding-corpus/nist-password-guidance.txt` — NIST SP 800-63B excerpts | US govt public domain |

## Thematic scope

The corpus covers "US federal regulation and jurisprudence" broadly —
FDA, CFPB/BCFP, SEC (Rule 10b-5), SCOTUS (Miranda), EPA (PFAS), NIST (AI
RMF, passwords), IETF (RFC 2119), USPTO. This gives multiple plausible
"refusal targets" for unanswerable questions (EU regulations, state law,
copyright law, tax law — none covered) while keeping questions about the
included documents well-grounded.

## Expected URIs

Each document's `doc_uri` at corpus-build time is `file:///absolute/path/to/<filename>`
from `path.as_uri()` in the WS-3.7 dispatch helper. The benchmark
harness maps these to stable short slugs via its `_URI_TO_SLUG` table;
the golden question JSONL references those slugs.

## Extending this fixture

WS-3.7 Phase 2 will expand this to ~30 documents by fetching fresh content
from eCFR, Federal Register, NIST, NASA, and whitehouse GitHub repositories.
Each fetched file must be annotated here with URL + retrieval date +
license note.
