# kaos-llm-core test fixtures

This directory holds every test fixture consumed by `kaos-llm-core`'s
unit, integration, and benchmark suites. Per
[`docs/oss/50-data-and-fixtures/provenance-policy.md`](../../../docs/oss/50-data-and-fixtures/provenance-policy.md),
every fixture directory has a `README.md` with per-file source,
license, retrieved date, and SHA-256 columns.

## Index

| Directory | Purpose | README |
|-----------|---------|--------|
| `cross-doc-corpus/` | Synthetic employment + real-estate agreements for cross-document `extract_corpus` fan-out tests | [README](cross-doc-corpus/README.md) |
| `cuad-sample/` | 5-contract × 5-clause subset of CUAD v1 (CC-BY-4.0) for the WS-TR.PR-5 CUAD extraction benchmark | [README](cuad-sample/README.md) |
| `expanded-corpus/` | EDGAR / GAO / amicus / EPA excerpts for the FUND-6 30-doc corpus QA expansion | [README](expanded-corpus/README.md) |
| `grounding-corpus/` | 10 ASCII-only public-domain / synthetic short documents for `test_grounding_e2e.py` and `scripts/grounding-calibration.py` | [README](grounding-corpus/README.md) |
| `long-doc-sample/` | A single ~63 KB PPD employment agreement (SEC EX-10) for WS-TR.PR-6b chunk-retry validation | [README](long-doc-sample/README.md) |
| `multiformat-corpus/` | 10 US-government / synthetic documents across PDF / DOCX / HTML / Markdown / TXT for the WS-3.7 benchmark | [README](multiformat-corpus/README.md) |
| `privilege-sample/` | Two synthetic email-style fixtures (privileged vs non-privileged) for the privilege-classification e2e | [README](privilege-sample/README.md) |

## Top-level files

| File | Source | License | Retrieved | SHA-256 | Notes |
|------|--------|---------|-----------|---------|-------|
| `trec6_mini.jsonl` | Hand-curated 35-row subset of the TREC-6 question-classification corpus (20 train + 15 val, balanced across the 6 ABBR/DESC/ENTY/HUM/LOC/NUM categories). Authored in-house from the question prompts paraphrased in style of the original TREC-6 distribution; no rows were copied verbatim from the upstream TREC release. Used to drive the live MIPROv2 integration test (`tests/integration/test_mipro_v2_live.py`) without requiring a network fetch. | Apache-2.0 (this repo) | 2026-04-09 | `01df4db7134a0889c5af52996fca465b1c04c36591befab02511e03c77bdf816` | Phase 17.1 C5 — live MIPROv2 vs CoOptimizer joint-search benchmark |

## Regenerating hashes

If a top-level fixture is refreshed::

    cd kaos-llm-core/tests/fixtures
    sha256sum *.jsonl

and update the `Retrieved` column via
`git log --diff-filter=A --follow --format=%cI -- <path> | tail -1 | cut -c1-10`.

## TODO — items needing human verification

- `trec6_mini.jsonl`: confirm whether any of the 35 questions are
  verbatim from the upstream TREC-6 question-classification corpus.
  TREC-6 is distributed by NIST under research-use terms; if any
  rows are copied verbatim rather than paraphrased, the License
  column needs to be amended from "Apache-2.0 (this repo)" to the
  appropriate NIST TREC research-use citation. Resolution: spot-check
  the 35 questions against the canonical TREC-6 question set.
