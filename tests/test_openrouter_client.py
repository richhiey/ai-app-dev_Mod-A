import json

import httpx
import pytest
from pydantic import BaseModel

from models import ChatModel
from openrouter import OpenRouterClient, OpenRouterError


class RouteDecision(BaseModel):
    intent: str
    confidence: float


def make_client(handler):
    return OpenRouterClient(
        api_key="test-key",
        app_title="course-core-tests",
        http_referer="https://example.test",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_posts_openrouter_chat_payload() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-test",
                "model": "google/gemini-2.5-flash-lite",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "native_finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hello"},
                    }
                ],
                "usage": {"total_tokens": 3},
            },
        )

    client = make_client(handler)
    result = client.chat([{"role": "user", "content": "Say hello"}])

    assert seen["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert seen["headers"]["authorization"] == "Bearer test-key"
    assert seen["headers"]["x-openrouter-title"] == "course-core-tests"
    assert seen["json"]["model"] == "google/gemini-2.5-flash-lite"
    assert result.content == "Hello"
    assert result.usage == {"total_tokens": 3}


def test_client_raises_on_openrouter_error_even_with_http_200() -> None:
    client = make_client(
        lambda request: httpx.Response(
            200,
            json={"error": {"code": 529, "message": "Provider overloaded"}},
        )
    )

    with pytest.raises(OpenRouterError, match="Provider overloaded"):
        client.chat([{"role": "user", "content": "Hi"}])


def test_structured_output_sends_json_schema_and_validates_response() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-structured",
                "model": "google/gemini-3.1-flash-lite",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"intent":"retrieval","confidence":0.87}',
                        },
                    }
                ],
            },
        )

    client = make_client(handler)
    parsed = client.structured(
        [{"role": "user", "content": "Classify: add reranking"}],
        output_model=RouteDecision,
        schema_name="route_decision",
    )

    assert parsed == RouteDecision(intent="retrieval", confidence=0.87)
    assert seen["json"]["model"] == ChatModel.GEMINI_31_FLASH_LITE.value
    assert seen["json"]["provider"]["require_parameters"] is True
    assert seen["json"]["response_format"]["type"] == "json_schema"
    assert seen["json"]["response_format"]["json_schema"]["strict"] is True


def test_embeddings_post_batch_input() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "google/gemini-embedding-001",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                ],
                "usage": {"total_tokens": 5},
            },
        )

    client = make_client(handler)
    embeddings = client.embed(["retrieval", "tool schemas"])

    assert seen["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert seen["json"]["model"] == "google/gemini-embedding-001"
    assert seen["json"]["input"] == ["retrieval", "tool schemas"]
    assert embeddings == [[1.0, 0.0], [0.0, 1.0]]


def test_rerank_posts_to_openrouter_rerank_endpoint() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "gen-rerank",
                "model": "cohere/rerank-v3.5",
                "results": [
                    {
                        "index": 1,
                        "relevance_score": 0.98,
                        "document": {"text": "Hybrid search combines lexical and semantic signals."},
                    }
                ],
            },
        )

    client = make_client(handler)
    ranked = client.rerank(
        query="keyword plus semantic search",
        documents=["A model can call tools.", "Hybrid search combines lexical and semantic signals."],
        top_n=1,
    )

    assert seen["url"] == "https://openrouter.ai/api/v1/rerank"
    assert seen["json"]["model"] == "cohere/rerank-v3.5"
    assert seen["json"]["top_n"] == 1
    assert ranked[0].index == 1
    assert ranked[0].relevance_score == 0.98
