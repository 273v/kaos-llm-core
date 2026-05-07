# Cross-Document Corpus Fixture

Three short employment / real-estate agreements used by the cross-document
extraction tests to exercise multi-document fan-out under `extract_corpus`.

All three text files are **synthetic, illustrative documents** — none are
real contracts, and any resemblance to actual companies, persons, parties,
or addresses is coincidental. They are deliberately written in the style
of EDGAR EX-10 employment exhibits so the extractors face realistic
clause shapes (parties, effective date, governing law, severance,
non-compete) without exposing any real-person PII.

## Documents

| File | URI | Type | Source |
|------|-----|------|--------|
| `galera_employment.txt` | `doc:cross-doc/galera_employment` | employment agreement | synthetic — fictional employer / executive |
| `tenon_employment.txt` | `doc:cross-doc/tenon_employment` | employment agreement | synthetic — fictional employer / executive |
| `amerimetro_real_estate.txt` | `doc:cross-doc/amerimetro_real_estate` | real-estate sale agreement | synthetic — fictional buyer / seller |

## License

Synthetic content. Apache-2.0 with the rest of the kaos-llm-core test fixtures.

## Used by

- `tests/integration/test_dedup_extract_composition.py`
- Anything else that loads files matching `tests/fixtures/cross-doc-corpus/*.txt`
