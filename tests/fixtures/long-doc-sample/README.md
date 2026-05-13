# Long-Doc Sample Fixture (WS-TR.PR-6b)

A single multi-section employment agreement used by the WS-TR.PR-6b
chunk-retry validation: long enough (~63 KB, ~7–8 chunks at the default
`initial_chunk_chars=8000`) to exercise the per-cell chunk-retry path,
but short enough that a long-context model can also see the whole
document in one call for the baseline. See `MANIFEST.json` for the
target columns and `golden.jsonl` for the per-column ground truth.

## Provenance

| File | Source | License | Retrieved | SHA-256 | Notes |
|------|--------|---------|-----------|---------|-------|
| `ppd-employment-agreement.txt` | Pharmaceutical Product Development, Inc. CEO employment agreement, filed as a SEC 10-K EX-10 exhibit. Vendored via `kelvin-nlp` test resources (originally `employment_02.txt`); see `MANIFEST.json` → `fixture_lineage`. The upstream SEC URL was not captured at vendoring time. | Public domain — 17 USC §105 (SEC filing as a US government work) | 2026-04-15 | `7657d1dc9a784996f3d141decb8008b891f4a398ada434eacc7e637315240fad` | Real corporate exhibit; parties (PPD, Raymond H. Hill) are named in the public filing record |
| `golden.jsonl` | Hand-authored ground truth (273V) for the WS-TR.PR-6b chunk-retry benchmark — one row per target column with the verbatim CUAD-style span and offsets | Apache-2.0 (this repo) | 2026-04-15 | `305d9c4ab10d77576054ee98d4dc8fe9b41edb07b35a3b9793e9aa413204be94` | Companion to `MANIFEST.json` |
| `MANIFEST.json` | Hand-authored manifest pinning the document, schema, and target-column rationale | Apache-2.0 (this repo) | 2026-04-15 | `2a3b2a0a485c046b2247af9ef4170b910e61fcb0ff2317a309d93c78b69e3c6f` | — |

## Used by

- `kaos_llm_core.programs.chunk_retry` integration tests
- WS-TR.PR-6b benchmark scripts (long-document column-extraction recovery)

## Regenerating hashes

If the fixture is refreshed, regenerate the SHA-256 column with::

    cd kaos-llm-core/tests/fixtures/long-doc-sample
    sha256sum *.txt *.jsonl *.json

and update the `Retrieved` column to the new vendoring date via
`git log --diff-filter=A --follow --format=%cI -- <path> | tail -1 | cut -c1-10`.

## TODO — items needing human verification

- `ppd-employment-agreement.txt`: the original SEC EDGAR filing URL
  (accession number + filing date + filer CIK) is not captured. The
  document is filed as an EX-10 exhibit somewhere in PPD's 10-K
  filings; until the precise EDGAR URL is pinned, the per-row `Source`
  column documents the lineage chain (kelvin-nlp `employment_02.txt`
  → upstream SEC EX-10) but does not cite a stable upstream URL.
  Resolution: re-locate the filing via EDGAR's full-text search and
  add the canonical `https://www.sec.gov/Archives/...` URL.
