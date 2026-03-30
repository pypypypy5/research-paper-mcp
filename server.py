#!/usr/bin/env python3
"""
Research Paper Ingestion MCP Server
====================================

Autonomous knowledge acquisition from academic research papers.
"""

import asyncio
import logging

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

from env_loader import load_env_file
from paper_tools import (
    TOOL_DEFINITIONS,
    analyze_citations,
    call_tool,
    download_paper,
    extract_insights,
    get_arxiv_sections,
    search_arxiv,
    search_semantic_scholar,
    store_paper_knowledge,
)


load_env_file()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("research-paper-mcp")

server = Server("research-paper-mcp")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available research paper ingestion tools."""
    return TOOL_DEFINITIONS


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests."""
    return await call_tool(name, arguments or {})


async def main() -> None:
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Research Paper MCP Server starting...")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="research-paper-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
