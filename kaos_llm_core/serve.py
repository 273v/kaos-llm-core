"""Run the KAOS MCP server with kaos-llm-core tools.

Usage:
    # stdio (for Claude Code / Claude Desktop)
    kaos-llm-core-serve

    # HTTP
    kaos-llm-core-serve --http --port 8000
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="KAOS MCP Server with kaos-llm-core tools")
    parser.add_argument("--http", action="store_true", help="Use streamable HTTP transport")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    try:
        from kaos_core import KaosRuntime

        # kaos-mcp is the optional [mcp] extra; not declared in 0.1.0a1
        # because it isn't on PyPI yet (returns in 0.1.0a2).
        from kaos_mcp import KaosMCPServer, KaosMCPSettings  # ty: ignore[unresolved-import]
    except ImportError:
        print(
            "Error: MCP server requires kaos-mcp.\nInstall with: pip install kaos-mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    from kaos_llm_core.integrations.mcp.registration import register_llm_core_tools

    runtime = KaosRuntime()
    n_tools = register_llm_core_tools(runtime)
    print(f"Registered {n_tools} kaos-llm-core tools", file=sys.stderr)

    instructions = (
        "kaos-llm-core provides LLM programming tools for typed, structured LLM calls. "
        "Use kaos-llm-core-call for structured extraction/classification with validated output. "
        "Use kaos-llm-core-reason for chain-of-thought reasoning with step-by-step analysis."
    )

    settings = KaosMCPSettings(
        name="kaos-llm-core-server",
        instructions=instructions,
        transport="streamable-http" if args.http else "stdio",
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server = KaosMCPServer(runtime=runtime, settings=settings)

    if args.http:
        print(f"Starting HTTP server on {args.host}:{args.port}/mcp", file=sys.stderr)
        server.run_streamable_http()
    else:
        print("Starting stdio server", file=sys.stderr)
        server.run_stdio()


if __name__ == "__main__":
    main()
