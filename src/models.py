"""Course model allowlist and defaults.

All notebook helpers route through OpenRouter, but the course has a strict
model budget and access policy. Keep that policy here so scripts, notebooks,
and tests fail early if a disabled model slips in.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable


class ModelNotAllowedError(ValueError):
    """Raised when code attempts to use a model outside the course allowlist."""


class ChatModel(str, Enum):
    GEMINI_31_FLASH_LITE = "google/gemini-3.1-flash-lite"
    GEMINI_25_FLASH_LITE = "google/gemini-2.5-flash-lite"
    GEMINI_25_FLASH = "google/gemini-2.5-flash"


class EmbeddingModel(str, Enum):
    GEMINI_EMBEDDING = "google/gemini-embedding-001"


class RerankModel(str, Enum):
    COHERE_RERANK = "cohere/rerank-v3.5"


class ModelPurpose(str, Enum):
    CHAT = "chat"
    AGENT = "agent"
    REASONING = "reasoning"
    STRUCTURED = "structured"
    EMBEDDING = "embedding"
    RERANKING = "reranking"


_ALLOWED_MODEL_IDS: tuple[str, ...] = tuple(
    model.value
    for family in (ChatModel, EmbeddingModel, RerankModel)
    for model in family
)


_DEFAULTS: dict[ModelPurpose, ChatModel | EmbeddingModel | RerankModel] = {
    ModelPurpose.CHAT: ChatModel.GEMINI_25_FLASH_LITE,
    ModelPurpose.AGENT: ChatModel.GEMINI_31_FLASH_LITE,
    ModelPurpose.REASONING: ChatModel.GEMINI_25_FLASH,
    ModelPurpose.STRUCTURED: ChatModel.GEMINI_31_FLASH_LITE,
    ModelPurpose.EMBEDDING: EmbeddingModel.GEMINI_EMBEDDING,
    ModelPurpose.RERANKING: RerankModel.COHERE_RERANK,
}


def all_allowed_model_ids() -> tuple[str, ...]:
    """Return the complete course model allowlist."""

    return _ALLOWED_MODEL_IDS


def _model_value(model: str | Enum) -> str:
    if isinstance(model, Enum):
        return str(model.value)
    return str(model)


def ensure_allowed_model(model: str | Enum) -> str:
    """Return a model id if allowed, otherwise raise with a useful message."""

    model_id = _model_value(model)
    if model_id not in _ALLOWED_MODEL_IDS:
        allowed = ", ".join(_ALLOWED_MODEL_IDS)
        raise ModelNotAllowedError(
            f"Model '{model_id}' is not enabled for this course. Allowed models: {allowed}."
        )
    return model_id


def default_model_for(
    purpose: ModelPurpose | str,
) -> ChatModel | EmbeddingModel | RerankModel:
    """Return the default enabled model for a course workflow purpose."""

    purpose_key = ModelPurpose(purpose)
    return _DEFAULTS[purpose_key]


def ensure_all_allowed(models: Iterable[str | Enum]) -> list[str]:
    """Validate a sequence of models and return their string ids."""

    return [ensure_allowed_model(model) for model in models]
