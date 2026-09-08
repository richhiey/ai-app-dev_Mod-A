"""Hybrid retrieval utilities that blend Chroma and BM25 results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from documents import Document
from keyword_search import BM25Retriever


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    distance: float | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None


class SemanticStore(Protocol):
    def semantic_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        ...


class HybridRetriever:
    """Combine semantic and lexical retrieval with a tunable blend weight."""

    def __init__(
        self,
        *,
        vector_store: SemanticStore,
        keyword_retriever: BM25Retriever,
        alpha: float = 0.65,
    ) -> None:
        self.vector_store = vector_store
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        semantic_k: int = 20,
        keyword_k: int = 20,
        alpha: float | None = None,
        where: dict | None = None,
        semantic_query: str | None = None,
        keyword_query: str | None = None,
    ) -> list[RetrievalResult]:
        blend = self.alpha if alpha is None else alpha
        if not 0 <= blend <= 1:
            raise ValueError("alpha must be between 0 and 1.")

        semantic_results = self.vector_store.semantic_search(
            semantic_query or query,
            top_k=semantic_k,
            where=where,
        )
        keyword_results = self.keyword_retriever.search(keyword_query or query, top_k=keyword_k)

        semantic_by_id = {
            result.document.id: _semantic_score(result)
            for result in semantic_results
        }
        keyword_by_id = {
            result.document.id: result.keyword_score
            for result in keyword_results
        }
        normalized_semantic = _normalize(semantic_by_id)
        normalized_keyword = _normalize(keyword_by_id)

        merged: dict[str, RetrievalResult] = {}
        for result in semantic_results:
            merged[result.document.id] = result
        for result in keyword_results:
            if result.document.id in merged:
                base = merged[result.document.id]
                merged[result.document.id] = replace(
                    base,
                    keyword_score=result.keyword_score,
                )
            else:
                merged[result.document.id] = RetrievalResult(
                    document=result.document,
                    keyword_score=result.keyword_score,
                )

        rescored: list[RetrievalResult] = []
        for document_id, result in merged.items():
            semantic_score = result.semantic_score
            if semantic_score is None and result.distance is not None:
                semantic_score = _distance_to_score(result.distance)
            keyword_score = result.keyword_score or keyword_by_id.get(document_id)
            hybrid_score = (
                blend * normalized_semantic.get(document_id, 0.0)
                + (1 - blend) * normalized_keyword.get(document_id, 0.0)
            )
            rescored.append(
                replace(
                    result,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    hybrid_score=hybrid_score,
                )
            )

        rescored.sort(key=lambda result: result.hybrid_score or 0.0, reverse=True)
        return rescored[:top_k]


def _semantic_score(result: RetrievalResult) -> float:
    if result.semantic_score is not None:
        return result.semantic_score
    if result.distance is not None:
        return _distance_to_score(result.distance)
    return 0.0


def _distance_to_score(distance: float) -> float:
    return 1 / (1 + max(distance, 0.0))


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    positive = {key: value for key, value in scores.items() if value > 0}
    if not positive:
        return {}
    values = list(positive.values())
    low, high = min(values), max(values)
    if high == low:
        return {key: 1.0 for key in positive}
    return {key: (value - low) / (high - low) for key, value in positive.items()}
