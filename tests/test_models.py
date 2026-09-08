import pytest

from models import (
    ChatModel,
    EmbeddingModel,
    ModelPurpose,
    ModelNotAllowedError,
    RerankModel,
    all_allowed_model_ids,
    default_model_for,
    ensure_allowed_model,
)


def test_allowed_models_match_course_allowlist() -> None:
    assert set(all_allowed_model_ids()) == {
        "google/gemini-3.1-flash-lite",
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "google/gemini-embedding-001",
        "cohere/rerank-v3.5",
    }


def test_default_models_are_intentional_for_course_tasks() -> None:
    assert default_model_for(ModelPurpose.CHAT) == ChatModel.GEMINI_25_FLASH_LITE
    assert default_model_for(ModelPurpose.AGENT) == ChatModel.GEMINI_31_FLASH_LITE
    assert default_model_for(ModelPurpose.REASONING) == ChatModel.GEMINI_25_FLASH
    assert default_model_for(ModelPurpose.EMBEDDING) == EmbeddingModel.GEMINI_EMBEDDING
    assert default_model_for(ModelPurpose.RERANKING) == RerankModel.COHERE_RERANK


def test_rejects_models_outside_the_course_allowlist() -> None:
    with pytest.raises(ModelNotAllowedError):
        ensure_allowed_model("openai/gpt-5")


def test_accepts_enabled_model_strings_and_enums() -> None:
    assert ensure_allowed_model("google/gemini-2.5-flash") == "google/gemini-2.5-flash"
    assert ensure_allowed_model(ChatModel.GEMINI_31_FLASH_LITE) == "google/gemini-3.1-flash-lite"
