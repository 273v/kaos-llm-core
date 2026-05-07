# Privilege Classification Sample Fixture

Two short email-style documents used by the privilege-classification end-to-end
test to verify that a Signature can correctly distinguish privileged from
non-privileged communications.

Both files are **fully synthetic, illustrative documents** — the firm names,
domain names, sender / recipient addresses, matter references, and legal
positions are all fictional. Any resemblance to real law firms, attorneys,
or matters is coincidental.

## Documents

| File | Class | Style |
|------|-------|-------|
| `privileged-email.txt` | privileged | attorney-client legal advice memo (synthetic) |
| `non-privileged-email.txt` | non-privileged | routine business correspondence (synthetic) |

## License

Synthetic content. Apache-2.0 with the rest of the kaos-llm-core test fixtures.

## Used by

- `scripts/privilege_classification_e2e.py`
- Tests that load fixtures from `tests/fixtures/privilege-sample/`
