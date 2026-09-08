from documents import Document
from hybrid import RetrievalResult
from openrouter import RerankResult
from rerank import OpenRouterReranker


class FakeClient:
    def rerank(self, query, documents, top_n=None, model=None):
        return [
            RerankResult(index=1, relevance_score=0.91, document={"text": documents[1]}),
            RerankResult(index=0, relevance_score=0.12, document={"text": documents[0]}),
        ][:top_n]


def test_openrouter_reranker_reorders_candidates_and_keeps_metadata() -> None:
    candidates = [
        RetrievalResult(document=Document(id="rag", text="RAG retrieval")),
        RetrievalResult(document=Document(id="hybrid", text="Hybrid search")),
    ]

    ranked = OpenRouterReranker(FakeClient()).rerank("keyword and semantic", candidates, top_n=2)

    assert [result.document.id for result in ranked] == ["hybrid", "rag"]
    assert ranked[0].rerank_score == 0.91
