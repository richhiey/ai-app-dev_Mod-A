from mcp_server import build_mcp_server, keyword_search_documents


def test_keyword_search_documents_is_usable_without_running_server() -> None:
    results = keyword_search_documents(
        query="tool schema",
        documents=[
            "RAG retrieval uses embeddings.",
            "Tool schemas describe function inputs and outputs.",
        ],
        top_k=1,
    )

    assert results[0]["index"] == 1
    assert results[0]["score"] > 0


def test_build_mcp_server_returns_server_instance() -> None:
    server = build_mcp_server()

    assert server is not None
