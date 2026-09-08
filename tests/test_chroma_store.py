import sys
from types import SimpleNamespace

import pytest

from documents import Document
from vector_store import ChromaStore, ChromaStoreError


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "retrieval" in lower or "rag" in lower else 0.0,
                    1.0 if "tool" in lower or "schema" in lower else 0.0,
                    1.0 if "mcp" in lower else 0.0,
                ]
            )
        return vectors


def test_chroma_store_indexes_and_queries_documents(tmp_path) -> None:
    store = ChromaStore(
        path=tmp_path / "chroma",
        collection_name="course_test",
        embedder=FakeEmbedder(),
    )
    docs = [
        Document(id="rag", text="RAG retrieval uses embeddings and reranking.", metadata={"lesson": "LS 5"}),
        Document(id="tools", text="Tool schemas make function calls reliable.", metadata={"lesson": "LS 9"}),
    ]

    store.index(docs)
    results = store.semantic_search("retrieval quality", top_k=1)

    assert results[0].document.id == "rag"
    assert results[0].document.metadata["lesson"] == "LS 5"
    assert results[0].semantic_score is not None


def test_chroma_store_explains_corrupt_tenant_schema(monkeypatch, tmp_path) -> None:
    class BrokenClient:
        def __init__(self, *, path):
            self.path = path

        def get_or_create_collection(self, name):
            raise RuntimeError("Database error: no such table: tenants")

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=lambda path: BrokenClient(path=path)),
    )

    with pytest.raises(ChromaStoreError, match="ChromaDB could not open"):
        ChromaStore(
            path=tmp_path / "broken_chroma",
            collection_name="course_test",
            embedder=FakeEmbedder(),
        )
