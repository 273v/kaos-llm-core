# CUAD Sample Fixture (WS-TR.PR-5)

A **5-contract × 5-clause** subset of The Atticus Project's
[CUAD v1](https://github.com/TheAtticusProject/cuad) dataset, used by
the `kaos-llm-core` CUAD extraction calibration benchmark
(`scripts/cuad_extraction_benchmark.py`). Each `*.txt` is the verbatim
`context` field from CUAD v1 for the named contract; no text
modifications. The five target clauses (Parties / Agreement Date /
Governing Law / Termination For Convenience / Cap On Liability) and
their CUAD verbatim gold spans live in `cuad-extraction-golden.jsonl`.

Full license, attribution, citation, modification log, and the upstream
`CUADv1.json` SHA-256 are in `DATA_LICENSES.md`. The per-contract
mapping back to CUAD indices is in `MANIFEST.json`.

## Provenance

| File | Source | License | Retrieved | SHA-256 | Notes |
|------|--------|---------|-----------|---------|-------|
| `ticketscominc-06-22-1999-ex-10-22-sponsorship-agreement.txt` | CUAD v1 → CUAD index 346, "TICKETSCOMINC_06_22_1999-EX-10.22-SPONSORSHIP AGREEMENT" (SEC EX-10 exhibit, originally a public filing) | CC-BY-4.0 (CUAD) | 2026-04-14 | `4928e56782c0c6c8d589f9ceb019dc418e38b7b324be28f49a20a81b03d1eaa7` | Sponsorship agreement |
| `mphasetechnologiesinc-20030911-10-k-ex-10-15-1560667-ex-10-1.txt` | CUAD v1 → CUAD index 186, "MphaseTechnologiesInc_20030911_10-K_EX-10.15_1560667_EX-10.15_Co-Branding Agreement" | CC-BY-4.0 (CUAD) | 2026-04-14 | `c00e73fa23804a8ea70f011dfaca0debf75bccdeb375ca864cbf2a36a0ff4ce8` | Co-branding / reseller |
| `dragonsystemsinc-01-08-1999-ex-10-17-outsourcing-agreement.txt` | CUAD v1 → CUAD index 62, "DRAGONSYSTEMSINC_01_08_1999-EX-10.17-OUTSOURCING AGREEMENT" | CC-BY-4.0 (CUAD) | 2026-04-14 | `8d8020e5c0bcaafac609fe53a2e7edbe2ecbf15541dc846ed25ac63ec822b37e` | Outsourcing agreement |
| `centrackinternationalinc-10-29-1999-ex-10-3-web-site-hosting.txt` | CUAD v1 → CUAD index 3, "CENTRACKINTERNATIONALINC_10_29_1999-EX-10.3-WEB SITE HOSTING AGREEMENT" | CC-BY-4.0 (CUAD) | 2026-04-14 | `8532356d811b76bd7ae536cf22910742ba61815dc39741fd9b295d7bc80c99f1` | Web hosting agreement |
| `lucidinc-04-15-2011-ex-10-9-distributor-agreement.txt` | CUAD v1 → CUAD index 164, "LUCIDINC_04_15_2011-EX-10.9-DISTRIBUTOR AGREEMENT" | CC-BY-4.0 (CUAD) | 2026-04-14 | `85676ae23975c26b41b9bc868f278c17addb353cf956628ad3c6c6d6108ce716` | Distributor agreement |
| `cuad-extraction-golden.jsonl` | Hand-extracted from CUAD v1's SQuAD-format annotations: one JSON object per contract, grouped by clause type, restricted to the 5 target clauses | CC-BY-4.0 (CUAD-derived) | 2026-04-14 | `af1bed6bbfdbc1f8aad2a30daf7cea576605bf556fbc17df6245a020949b8342` | See `DATA_LICENSES.md` §Modifications |
| `MANIFEST.json` | Hand-authored manifest pinning each contract back to its CUAD index + upstream `CUADv1.json` SHA-256 | CC-BY-4.0 (CUAD-derived) | 2026-04-14 | `59565272682bc2da088b15dcc34433df4c39420127eb3f266368a98d8b99cf50` | — |
| `DATA_LICENSES.md` | Hand-authored attribution + citation + modification log per CUAD CC-BY-4.0 redistribution requirements | CC-BY-4.0 (CUAD-derived) | 2026-04-14 | `3a1c9fdeb7606accc62c8b398f25e4214310902f064d99854fe60e3537c1ef9c` | Read this for the full license terms |

## Source

- Upstream URL: <https://github.com/TheAtticusProject/cuad>
- Upstream artifact: `CUADv1.json` inside `data.zip`
- Upstream SHA-256: `ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae`
- DOI: <https://doi.org/10.5281/zenodo.4595826>
- Citation: Hendrycks, D., Burns, C., Chen, A., & Ball, S. (2021).
  *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*.
  NeurIPS 2021 Datasets & Benchmarks Track. arXiv:2103.06268.

## Used by

- `kaos_llm_core/scripts/cuad_extraction_benchmark.py`
- WS-TR.PR-5 benchmark suite (CUAD-anchored extraction calibration —
  VLAIR-comparable)
- Integration tests under `tests/integration/` that exercise CUAD-shaped
  contract extraction

## Regenerating hashes

If the fixture is refreshed (e.g. CUAD v2 ships), regenerate the
SHA-256 column with::

    cd kaos-llm-core/tests/fixtures/cuad-sample
    sha256sum *.txt *.jsonl *.json *.md

and update the `Retrieved` column to the new vendoring date via
`git log --diff-filter=A --follow --format=%cI -- <path> | tail -1 | cut -c1-10`.

Also update the upstream `source_sha256` in `MANIFEST.json` and
`DATA_LICENSES.md` against the new `CUADv1.json` (or `CUADv2.json`)
checksum.
