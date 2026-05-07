# FUND-6 Expanded Corpus Fixtures

Additional documents for the 30-doc corpus QA expansion. Each file
is a real publicly available excerpt selected to cover document types
missing from the existing grounding-corpus (legal fundamentals),
multiformat-corpus (format diversity), and cuad-sample (contracts).

| File | Type | Source | License |
|------|------|--------|---------|
| `sec_10k_msft_excerpt.txt` | 10-K (EDGAR) | Microsoft Corp 10-K (SEC EDGAR) | Public domain (gov) |
| `gao_report_excerpt.txt` | GAO report | GAO-24-106300 (Cybersecurity) | Public domain (gov) |
| `court_brief_excerpt.txt` | Amicus brief | Supreme Court amicus brief (public filing) | Public domain (court) |
| `epa_rulemaking_excerpt.txt` | EPA rulemaking | EPA CWA 40 CFR 122 (NPDES) | Public domain (gov) |

All excerpts are trimmed to 1-3 KB for fixture portability. Each has
hand-authored Q/A triples in `expanded-qa-golden.jsonl`.
