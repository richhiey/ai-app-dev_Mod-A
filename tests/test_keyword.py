from documents import Document
from keyword_search import BM25Retriever


def test_bm25_ranks_keyword_matches() -> None:
    docs = [
        Document(id="rag", text="RAG retrieval uses chunks, embeddings, and reranking."),
        Document(id="tools", text="Tool schemas define function names, arguments, and contracts."),
        Document(id="mcp", text="MCP exposes tools and resources through a standard protocol."),
    ]

    retriever = BM25Retriever.from_documents(docs)
    results = retriever.search("tool schema contract", top_k=2)

    assert results[0].document.id == "tools"
    assert results[0].keyword_score > results[1].keyword_score


def test_bm25_returns_empty_for_empty_query() -> None:
    retriever = BM25Retriever.from_documents(
        [Document(id="rag", text="RAG retrieval")]
    )

    assert retriever.search("   ", top_k=3) == []
