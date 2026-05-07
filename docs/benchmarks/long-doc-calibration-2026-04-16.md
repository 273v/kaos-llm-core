# FUND-7 Long-Document Calibration — 2026-04-16

Model: **anthropic:claude-haiku-4-5**
Document: `ppd-employment-agreement.txt` (62,974 chars)

**Key finding.** Both baseline and chunk-retry achieve 100% on this
fixture because 63K chars fits well within Haiku's 200K context
window — the model sees the entire document in a single call. This is
the correct result: it confirms (1) the extraction pipeline works
correctly on a real 63KB legal document, (2) chunk-retry doesn't
degrade accuracy, and (3) all 4 target columns extract with correct
values against the hand-authored golden JSONL.

**To demonstrate chunk-retry recovery**, we'd need a document larger
than the model's effective attention span — likely 200KB+ for Haiku
or a model with a shorter window. FUND-7.1 will add a 200KB+
Microsoft 10-K fixture to surface that boundary. The current fixture
validates the pipeline works; the next fixture validates it recovers.

| Mode | Correct | Total | Accuracy | Time | Cost |
|------|---------|-------|----------|------|------|
| Baseline | 4 | 4 | 100% | 2.09s | $0.0000 |
| ChunkRetry | 4 | 4 | 100% | 1.94s | $0.0000 |

## Per-cell recovery

| Column | Baseline | ChunkRetry | Recovered? |
|--------|----------|------------|------------|
| parties | PASS | PASS |  |
| effective_date | PASS | PASS |  |
| governing_law | PASS | PASS |  |
| termination_for_convenience | PASS | PASS |  |