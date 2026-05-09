# Tests, Fixtures, Fuzzing, And CI Standards

This document defines test tiers, fixture rules, fuzzing expectations,
and GitHub Actions standards for `kaos-llm-core`.

## Test Tiers

Use explicit markers for non-unit tests. Markers currently declared in
`pyproject.toml`: `integration`, `live`, `unit`.

| Tier | Marker | Network | Credentials | Purpose |
|---|---|---|---|---|
| Unit | none, or `unit` if declared | No | No | Fast deterministic behavior. |
| Integration | `integration` | Usually no | No | Multiple local components together. |
| Network | `network` | Yes | No secrets | Public HTTP or unauthenticated services. |
| Live | `live` | Yes | Yes | Real provider APIs and credentials. |
| Slow | `slow` | Maybe | Maybe | Long-running checks, benchmarks, corpora. |
| Security | `security` if declared, otherwise focused unit tests | No by default | No | Abuse cases, limits, traversal, injection. |
| Fuzz | `fuzz` if declared | No | No | Property/fuzz targets and minimized cases. |

Unit-tier CI must not require network, credentials, local services, or
large downloads.

## Test Requirements

- New behavior needs tests.
- Bug fixes need regression tests.
- Security fixes need abuse-case tests where safe.
- README quick starts and CLI examples need smoke coverage or manual
  verification before release.
- Public providers and advertised extras need at least one test at the
  appropriate tier.
- Tests should assert semantics, not just non-empty output.
- Tests should avoid wall-clock sleeps unless testing timeouts.

## Marker Discipline

- Live-provider tests must be marked `live`.
- Network tests must be marked `network`.
- Slow tests must be marked `slow`.
- Integration modules should use module-level markers when every test in
  the file belongs to that tier.
- CI unit selection should be able to run:

```bash
uv run pytest -m "not live and not network and not slow" --no-cov
```

The command above must not collect tests that need credentials or
external services.

## Fixtures

Detected fixture directories: `tests/fixtures`.

Fixtures must be:

- Small enough for normal repository use.
- Redistributable under compatible terms.
- Free of customer data, privileged content, secrets, and PII.
- Documented with source, license, and purpose.
- Stable enough to support deterministic tests.

Do not commit:

- Customer documents.
- Real credentials.
- Unknown-license data.
- Non-commercial or no-derivatives data for redistributed fixtures.
- Large binary corpora that should be downloaded and hash-verified.

## Fixture Provenance

Every fixture directory should include a README or manifest that records:

- File name.
- Source URL or generation method.
- License or public-domain status.
- Retrieval date when relevant.
- SHA256 for externally sourced files.
- Reason the fixture exists.
- Any transformations applied.

Generated fixtures should include the generator script or enough
information to recreate them.

## Golden Files

Golden files are allowed when output stability matters.

Rules:

- Keep golden files small and reviewable.
- Include a command for regenerating them.
- Review diffs semantically.
- Do not bless broad golden changes without explaining the behavior
  change.
- Store comments in a companion README when the file format cannot
  carry comments.

## Fuzzing

Use fuzzing for parsers, decoders, extractors, graph loaders, archive
handling, URL/file handling, and Rust/PyO3 boundaries.

Python fuzz/property testing:

- Prefer Hypothesis for structured inputs.
- Keep failing examples as regression tests.
- Bound generated sizes so local runs stay practical.

Fuzz targets should check:

- No crashes or panics.
- No infinite loops.
- No unbounded memory growth.
- Valid errors for invalid inputs.
- Round-trip or invariant properties where available.

## Coverage

- Coverage is a signal, not the goal.
- New important branches should be covered.
- Public API, error paths, security limits, and serialization deserve
  explicit tests.
- Do not add trivial tests only to move a percentage.

## CI Workflows

Required PR checks:

- Formatting.
- Linting.
- Type checking.
- Unit tests.
- Build check.
- Dependency/security audit where configured.

Recommended scheduled or manual checks:

- Network tests.
- Live-provider tests.
- Full security scan.
- Dependency audit.
- Fuzz corpus run.
- Benchmark regression check.

Release workflow checks:

- Clean checkout.
- Build wheel and sdist.
- Strict metadata check.
- Fresh install smoke test.
- Publish through OIDC.
- Verify published install after release when practical.

## GitHub Actions Standards

- Use least-privilege `permissions`.
- Do not expose secrets to forked PRs.
- Pin third-party actions to trusted versions.
- Prefer OIDC over static credentials.
- Separate build, test, security, and publish jobs.
- Cache dependencies carefully; never cache secrets.
- Keep workflow logs free of credentials and private paths.
- Use environment protection for publishing.

## Local Verification Commands

Base development setup:

```bash
uv sync --group dev
```

Fast local quality gate:

```bash
uv run ruff format --check kaos_llm_core tests
uv run ruff check kaos_llm_core tests
uv run ty check kaos_llm_core tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

Packaging gate when packaging, metadata, README rendering, or release
behavior changes:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

## Release Gate

Before release:

- Unit CI is green.
- Required integration/network/live tiers are green or explicitly
  deferred.
- Security checks are green.
- Fixtures have provenance.
- Fuzz/security regressions are included for parser or input-safety
  fixes.
- Build artifacts pass metadata checks.
- Fresh install smoke test passes.
