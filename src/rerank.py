"""Reranking helpers for retrieved candidates."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from hybrid import RetrievalResult
from models import RerankModel
from openrouter import RerankResult


class RerankClient(Protocol):
    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        model: RerankModel | str | None = None,
    ) -> list[RerankResult]:
        ...


class OpenRouterReranker:
    def __init__(
        self,
        client: RerankClient,
        *,
        model: RerankModel | str = RerankModel.COHERE_RERANK,
    ) -> None:
        self.client = client
        self.model = model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_n: int | None = None,
    ) -> list[RetrievalResult]:
        documents = [candidate.document.text for candidate in candidates]
        remote_results = self.client.rerank(
            query,
            documents,
            top_n=top_n,
            model=self.model,
        )
        ranked: list[RetrievalResult] = []
        for item in remote_results:
            if 0 <= item.index < len(candidates):
                ranked.append(
                    replace(
                        candidates[item.index],
                        rerank_score=item.relevance_score,
                    )
                )
        return ranked
