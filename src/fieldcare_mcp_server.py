"""MCP server for FieldCare service-document search."""

from __future__ import annotations

from fieldcare import search_fieldcare_service_docs
from mcp_server import _load_mcp_server_class


def build_fieldcare_mcp_server(name: str = "fieldcare-service-docs"):
    server = _load_mcp_server_class()(name)

    @server.tool()
    def search_service_docs(
        query: str, top_k: int = 5, retrieval_config: dict | None = None
    ) -> list[dict]:
        """Search current FieldCare troubleshooting, safety, and policy documents."""

        return search_fieldcare_service_docs(
            query, top_k=top_k, retrieval_config=retrieval_config
        )

    return server


mcp = build_fieldcare_mcp_server()


if __name__ == "__main__":
    if hasattr(mcp, "run"):
        mcp.run()
    else:
        raise SystemExit("Run this server with: mcp run src/fieldcare_mcp_server.py")
