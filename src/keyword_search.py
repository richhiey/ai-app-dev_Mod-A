"""Small BM25 keyword retriever for hybrid search lessons."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from documents import Document


@dataclass(frozen=True)
class KeywordResult:
    document: Document
    keyword_score: float


class BM25Retriever:
    def __init__(
        self,
        documents: list[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._tokens = [_tokenize(doc.text) for doc in documents]
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._idf = self._build_idf()

    @classmethod
    def from_documents(cls, documents: list[Document], **kwargs: float) -> "BM25Retriever":
        return cls(documents, **kwargs)

    def search(self, query: str, *, top_k: int = 10) -> list[KeywordResult]:
        query_terms = _tokenize(query)
        if not query_terms or top_k <= 0:
            return []
        scored = [
            KeywordResult(document=document, keyword_score=self._score_doc(i, query_terms))
            for i, document in enumerate(self.documents)
        ]
        scored = [result for result in scored if result.keyword_score > 0]
        scored.sort(key=lambda result: result.keyword_score, reverse=True)
        return scored[:top_k]

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self.documents)
        dfs: Counter[str] = Counter()
        for tokens in self._tokens:
            dfs.update(set(tokens))
        return {
            term: math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            for term, df in dfs.items()
        }

    def _score_doc(self, index: int, query_terms: list[str]) -> float:
        if not self._avg_doc_length:
            return 0.0
        counts = self._term_counts[index]
        doc_length = self._doc_lengths[index]
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self._avg_doc_length)
            score += self._idf.get(term, 0.0) * numerator / denominator
        return score


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [_simple_stem(token) for token in tokens if token]


def _simple_stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token
