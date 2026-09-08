"""ChromaDB-backed vector storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from documents import Document
from hybrid import RetrievalResult


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class ChromaStore:
    """Store course documents and embeddings in a persistent Chroma collection."""

    def __init__(
        self,
        *,
        path: str | Path,
        collection_name: str,
        embedder: Embedder,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedder = embedder
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("Install chromadb to use ChromaStore.") from exc

        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def index(self, documents: list[Document], *, batch_size: int = 32) -> int:
        """Upsert documents and their embeddings into Chroma."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        count = 0
        for batch in _batches(documents, batch_size):
            texts = [doc.text for doc in batch]
            embeddings = self.embedder.embed(texts)
            self._collection.upsert(
                ids=[doc.id for doc in batch],
                documents=texts,
                metadatas=[_sanitize_metadata(doc.metadata) for doc in batch],
                embeddings=embeddings,
            )
            count += len(batch)
        return count

    def semantic_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Query Chroma by embedding the query with the configured embedder."""

        if top_k <= 0:
            return []
        query_embedding = self.embedder.embed([query])[0]
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return _results_from_chroma(raw)

    def all_documents(self) -> list[Document]:
        """Return all stored documents for keyword retriever construction."""

        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        return [
            Document(id=str(doc_id), text=text, metadata=metadata or {})
            for doc_id, text, metadata in zip(ids, documents, metadatas)
            if text
        ]

    def reset_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)


def _results_from_chroma(raw: dict[str, Any]) -> list[RetrievalResult]:
    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    results: list[RetrievalResult] = []
    for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        if text is None:
            continue
        score = 1 / (1 + max(float(distance), 0.0)) if distance is not None else None
        results.append(
            RetrievalResult(
                document=Document(id=str(doc_id), text=text, metadata=metadata or {}),
                distance=float(distance) if distance is not None else None,
                semantic_score=score,
            )
        )
    return results


def _batches(items: list[Document], size: int) -> list[list[Document]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, bool | int | float | str):
            sanitized[key] = value
        elif value is not None:
            sanitized[key] = str(value)
    return sanitized
