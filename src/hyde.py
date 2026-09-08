"""HyDE query rewriting workflow."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from models import ChatModel


class HyDEQuery(BaseModel):
    original_query: str = Field(description="The learner or user query exactly as received.")
    rewritten_query: str = Field(description="A keyword-rich search query.")
    hypothetical_document: str = Field(
        description="A concise hypothetical answer document for semantic retrieval."
    )


class StructuredClient(Protocol):
    def structured(
        self,
        messages: list[dict[str, Any]],
        *,
        output_model: type[BaseModel],
        schema_name: str,
        model: ChatModel | str | None = None,
        **kwargs: Any,
    ) -> BaseModel:
        ...


class HyDERewriter:
    """Generate a hypothetical document that can improve recall."""

    def __init__(
        self,
        client: StructuredClient,
        *,
        model: ChatModel | str = ChatModel.GEMINI_25_FLASH_LITE,
    ) -> None:
        self.client = client
        self.model = model

    def rewrite(self, query: str, *, context_hint: str | None = None) -> HyDEQuery:
        system = (
            "You rewrite retrieval queries for AI application lessons. "
            "Return a hypothetical answer document that contains likely terminology, "
            "but do not invent citations or external facts."
        )
        user = f"Query: {query}"
        if context_hint:
            user += f"\nCourse context: {context_hint}"
        result = self.client.structured(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            output_model=HyDEQuery,
            schema_name="hyde_query",
            model=self.model,
        )
        return HyDEQuery.model_validate(result)
