"""Notebook-friendly MCP client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MCPToolInfo:
    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPToolCallResult:
    tool_name: str
    is_error: bool
    content_texts: list[str]
    raw_content: Any


@dataclass(frozen=True)
class MCPInspectAndCallResult:
    server_name: str
    server_version: str | None
    tools: list[MCPToolInfo]
    call: MCPToolCallResult


async def inspect_and_call_stdio_tool(
    *,
    command: str,
    args: list[str],
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    read_timeout_seconds: float | None = 30,
) -> MCPInspectAndCallResult:
    """Connect to an MCP stdio server, inspect tools, then call one tool.

    This mirrors the LS11/LS12 learning path: connect, inspect advertised
    capabilities, send one request, then validate the returned content.
    """

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise ImportError("Install 'mcp[cli]' to use MCP client helpers.") from exc

    server_params = StdioServerParameters(
        command=command,
        args=args,
        cwd=cwd,
        env=env,
    )
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=read_timeout_seconds,
        ) as session:
            initialized = await session.initialize()
            tools_result = await session.list_tools()
            call_result = await session.call_tool(
                tool_name,
                arguments or {},
                read_timeout_seconds=read_timeout_seconds,
            )

    server_info = initialized.server_info
    return MCPInspectAndCallResult(
        server_name=server_info.name,
        server_version=getattr(server_info, "version", None),
        tools=[
            MCPToolInfo(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in tools_result.tools
        ],
        call=MCPToolCallResult(
            tool_name=tool_name,
            is_error=bool(call_result.is_error),
            content_texts=_content_texts(call_result.content),
            raw_content=call_result.content,
        ),
    )


def _content_texts(content: Any) -> list[str]:
    texts: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(str(text))
        elif hasattr(item, "model_dump_json"):
            texts.append(item.model_dump_json())
        else:
            texts.append(str(item))
    return texts
