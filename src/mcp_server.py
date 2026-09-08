"""MCP server exposing a small subset of helper-core capabilities."""

from __future__ import annotations

from typing import Any

from documents import Document
from keyword_search import BM25Retriever
from models import all_allowed_model_ids


def course_core_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "allowed_models": list(all_allowed_model_ids()),
    }


def keyword_search_documents(
    *,
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    docs = [
        Document(id=str(index), text=document, metadata={"index": index})
        for index, document in enumerate(documents)
        if document.strip()
    ]
    results = BM25Retriever.from_documents(docs).search(query, top_k=top_k)
    return [
        {
            "index": int(result.document.id),
            "text": result.document.text,
            "score": result.keyword_score,
        }
        for result in results
    ]


def build_mcp_server(name: str = "ms-ai-ml-helper-core"):
    """Build an MCP server using the official SDK, with v1 fallback support."""

    server_cls = _load_mcp_server_class()
    server = server_cls(name)

    @server.tool()
    def health() -> dict[str, Any]:
        """Return server health and the course model allowlist."""

        return course_core_health()

    @server.tool()
    def keyword_search(query: str, documents: list[str], top_k: int = 3) -> list[dict[str, Any]]:
        """Rank plain text documents with BM25 keyword search."""

        return keyword_search_documents(query=query, documents=documents, top_k=top_k)

    return server


def _load_mcp_server_class():
    try:
        from mcp.server import MCPServer

        return MCPServer
    except Exception:
        try:
            from mcp.server.fastmcp import FastMCP

            return FastMCP
        except Exception as exc:
            raise ImportError("Install 'mcp[cli]' to build or run the MCP server.") from exc


mcp = build_mcp_server()


if __name__ == "__main__":
    if hasattr(mcp, "run"):
        mcp.run()
    else:
        raise SystemExit("Run this server with: mcp run src/mcp_server.py")
