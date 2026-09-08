from documents import Document
from hybrid import HybridRetriever, RetrievalResult
from keyword_search import BM25Retriever


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def semantic_search(self, query, top_k=10, where=None):
        return self.results[:top_k]


def test_hybrid_search_blends_semantic_and_keyword_scores() -> None:
    retrieval_doc = Document(id="retrieval", text="Hybrid search combines semantic and keyword matching.")
    tool_doc = Document(id="tools", text="Tool schemas define callable contracts.")
    vector_store = FakeVectorStore(
        [
            RetrievalResult(document=retrieval_doc, semantic_score=0.55, distance=0.8),
            RetrievalResult(document=tool_doc, semantic_score=0.95, distance=0.05),
        ]
    )
    keyword = BM25Retriever.from_documents([retrieval_doc, tool_doc])
    retriever = HybridRetriever(vector_store=vector_store, keyword_retriever=keyword, alpha=0.45)

    results = retriever.search("hybrid keyword search", top_k=2)

    assert [result.document.id for result in results] == ["retrieval", "tools"]
    assert results[0].hybrid_score > results[1].hybrid_score
    assert results[0].keyword_score is not None
    assert results[0].semantic_score is not None


def test_hybrid_search_rejects_invalid_alpha() -> None:
    doc = Document(id="x", text="x")
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([]),
        keyword_retriever=BM25Retriever.from_documents([doc]),
        alpha=1.5,
    )

    try:
        retriever.search("x")
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("Expected invalid alpha to raise")
