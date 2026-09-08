import pytest

from documents import Document, chunk_text


def test_chunk_text_preserves_source_metadata_and_order() -> None:
    text = "First paragraph explains model access.\n\nSecond paragraph explains retrieval and reranking."

    chunks = chunk_text(
        text,
        source_id="lesson-1",
        chunk_size=45,
        overlap=8,
        metadata={"lesson": "LS 1"},
    )

    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["source_id"] == "lesson-1" for chunk in chunks)
    assert all(chunk.metadata["lesson"] == "LS 1" for chunk in chunks)
    assert chunks[0].id == "lesson-1:chunk-0000"
    assert "model access" in chunks[0].text


def test_chunk_text_rejects_overlap_larger_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("small text", source_id="x", chunk_size=20, overlap=20)


def test_document_requires_non_empty_text() -> None:
    with pytest.raises(ValueError):
        Document(id="empty", text="   ")
