# FUND-6 Corpus QA Benchmark — 2026-04-16

Model: **anthropic:claude-haiku-4-5**
Overall: **42/48 = 88%**

| Set | Questions | Correct | Accuracy |
|-----|-----------|---------|----------|
| grounding | 20 | 18 | 90% |
| multiformat | 12 | 8 | 67% |
| expanded | 16 | 16 | 100% |

## Per-question results

| Set | ID | Answerable | Correct | Time |
|-----|----|------------|---------|------|
| grounding | q01 | True | PASS | 1.32s |
| grounding | q02 | True | PASS | 1.09s |
| grounding | q03 | True | PASS | 1.21s |
| grounding | q04 | True | **FAIL** | 1.71s |
| grounding | q05 | True | PASS | 1.46s |
| grounding | q06 | True | PASS | 0.81s |
| grounding | q07 | True | PASS | 1.03s |
| grounding | q08 | True | PASS | 1.59s |
| grounding | q09 | True | PASS | 0.98s |
| grounding | q10 | True | PASS | 1.15s |
| grounding | q11 | False | PASS | 1.15s |
| grounding | q12 | False | PASS | 1.23s |
| grounding | q13 | False | PASS | 1.38s |
| grounding | q14 | False | PASS | 1.11s |
| grounding | q15 | False | **FAIL** | 1.06s |
| grounding | q16 | False | PASS | 1.9s |
| grounding | q17 | False | PASS | 1.38s |
| grounding | q18 | False | PASS | 1.17s |
| grounding | q19 | False | PASS | 0.83s |
| grounding | q20 | False | PASS | 1.67s |
| multiformat | mf01 | True | PASS | 1.15s |
| multiformat | mf02 | True | PASS | 1.15s |
| multiformat | mf03 | True | **FAIL** | 2.03s |
| multiformat | mf04 | True | **FAIL** | 1.76s |
| multiformat | mf05 | True | PASS | 0.93s |
| multiformat | mf06 | True | **FAIL** | 2.05s |
| multiformat | mf07 | True | **FAIL** | 1.94s |
| multiformat | mf08 | False | PASS | 1.96s |
| multiformat | mf09 | False | PASS | 0.78s |
| multiformat | mf10 | False | PASS | 2.15s |
| multiformat | mf11 | False | PASS | 0.82s |
| multiformat | mf12 | False | PASS | 1.8s |
| expanded | exp01 | True | PASS | 1.57s |
| expanded | exp02 | True | PASS | 1.03s |
| expanded | exp03 | True | PASS | 1.16s |
| expanded | exp04 | False | PASS | 0.92s |
| expanded | exp05 | True | PASS | 1.28s |
| expanded | exp06 | True | PASS | 2.22s |
| expanded | exp07 | True | PASS | 0.78s |
| expanded | exp08 | False | PASS | 1.76s |
| expanded | exp09 | True | PASS | 1.9s |
| expanded | exp10 | True | PASS | 0.91s |
| expanded | exp11 | True | PASS | 1.36s |
| expanded | exp12 | False | PASS | 1.51s |
| expanded | exp13 | True | PASS | 0.89s |
| expanded | exp14 | True | PASS | 1.02s |
| expanded | exp15 | True | PASS | 1.25s |
| expanded | exp16 | False | PASS | 1.56s |