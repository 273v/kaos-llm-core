"""kaos-llm-core integrations layer.

Phase 14: integration adapters and the shared helpers they depend on.

- ``integrations.common`` — runtime-agnostic helpers used by both the
  starter API, the ``@llm_call`` decorator, and the MCP tool wrappers.
  Single source of truth for cross-cutting concerns like dynamic
  Signature construction.
- ``integrations.mcp`` — MCP tool wrappers for the kaos-mcp bridge.
  One module per tool keeps the surface inspectable; ``serve.py``
  imports the registration function from ``integrations.mcp``.

This package replaces the monolithic ``kaos_llm_core.tools`` module
that previously held all 17 MCP tool classes plus their helpers in
one 2865-line file. The split happens incrementally — Phase 14A
lands the common helpers, Phase 14B does the per-tool split, Phase
14C wires starter + decorator onto the common helpers.
"""

__all__: list[str] = []
