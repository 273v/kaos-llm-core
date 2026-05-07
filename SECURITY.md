# Security policy

## Reporting a vulnerability

We take security seriously. If you believe you have found a security
vulnerability in `kaos-llm-core`, please report it privately so we can address it
before public disclosure.

**Please do not file a public GitHub issue for security reports.**

### How to report

Use [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-llm-core/security/advisories/new)
to send a report. Alternatively, email **security@273ventures.com**.

Include as much of the following as you can:

- A description of the vulnerability and its impact
- Steps to reproduce, including affected versions
- Any proof-of-concept code, if available
- Suggested mitigations, if you have any

### What to expect

- **Acknowledgement** — within 3 business days of your report.
- **Initial triage** — within 7 business days, including a severity assessment.
- **Fix and disclosure** — coordinated with you. Our target window is 90 days
  from acknowledgement to public disclosure, faster for high-severity issues.
- **Credit** — we credit reporters in the release notes and security advisory
  unless you prefer to remain anonymous.

## Supported versions

`kaos-llm-core` follows Semantic Versioning. While the project is pre-1.0, only
the latest minor release receives security fixes. After 1.0, the latest two
minor releases will be supported.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Scope

In-scope:

- The `kaos-llm-core` Python package as published on PyPI
- The `273v/kaos-llm-core` GitHub repository (CI, release, supply chain)
- Signature input validation and codec encode/decode boundaries
- Program execution: ReAct, Refine, BestOfN, ChainOfThought, RAG, Grounded,
  ProgramOfThought (the subprocess sandbox + `allow_code_execution=True`
  opt-in gate)
- Program v3 envelope JSON parsing and execution
- Batch runner (`batch_run` library + MCP tools): JSONL log integrity,
  workspace SQLite (WAL, multi-process safety), cost-cap enforcement
- Semantic cache disk persistence (replay-on-init invariants)
- MCP server (`kaos-llm-core-serve`) — request validation, tool annotations,
  response size caps

Out of scope:

- Third-party dependencies (report to the upstream project — `pydantic`,
  `kaos-core`, `kaos-content`, `kaos-llm-client`, `kaos-nlp-core`)
- Provider-side issues at OpenAI / Anthropic / Google / xAI / Groq /
  Mistral / OpenRouter (report to the upstream provider — these are
  surfaced through `kaos-llm-client`'s transport)
- Issues caused by user-supplied configuration that explicitly disables
  safety features (e.g., constructing `ProgramOfThought(allow_code_execution=True)`
  and then passing untrusted input)
