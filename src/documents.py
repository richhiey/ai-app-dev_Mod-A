"""Document models and notebook-friendly chunking helpers."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Document id must not be empty.")
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Document text must not be empty.")
        return value


def chunk_text(
    text: str,
    *,
    source_id: str,
    chunk_size: int = 900,
    overlap: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Split text into stable, overlapping chunks.

    This uses character counts rather than tokenization so it works in Colab
    without extra tokenizer packages. The chunk sizes are deliberately visible
    and easy for students to tune.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be zero or positive.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    normalized = _normalize_text(text)
    if not normalized:
        return []

    chunks: list[Document] = []
    start = 0
    while start < len(normalized):
        target_end = min(start + chunk_size, len(normalized))
        end = _best_boundary(normalized, start, target_end)
        chunk = normalized[start:end].strip()
        if chunk:
            chunk_metadata = dict(metadata or {})
            chunk_metadata.update(
                {
                    "source_id": source_id,
                    "chunk_index": len(chunks),
                    "start_char": start,
                    "end_char": end,
                }
            )
            chunks.append(
                Document(
                    id=f"{source_id}:chunk-{len(chunks):04d}",
                    text=chunk,
                    metadata=chunk_metadata,
                )
            )
        if end >= len(normalized):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _normalize_text(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]
    return "\n\n".join(paragraphs)


def _best_boundary(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)
    window = text[start:target_end]
    paragraph_break = window.rfind("\n\n")
    if paragraph_break > len(window) // 2:
        return start + paragraph_break
    whitespace = max(window.rfind(" "), window.rfind("\n"))
    if whitespace > len(window) // 2:
        return start + whitespace
    return target_end
