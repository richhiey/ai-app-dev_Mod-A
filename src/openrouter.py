"""Thin, testable OpenRouter HTTP client."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from models import (
    ChatModel,
    EmbeddingModel,
    ModelPurpose,
    RerankModel,
    default_model_for,
    ensure_allowed_model,
)

TModel = TypeVar("TModel", bound=BaseModel)


class OpenRouterError(RuntimeError):
    """Error returned by OpenRouter or raised while preparing a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.metadata = metadata or {}


class StructuredOutputError(OpenRouterError):
    """Raised when a structured output cannot be parsed or validated."""


@dataclass(frozen=True)
class ChatResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]
    usage: dict[str, Any] | None = None
    model: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float
    document: dict[str, Any] | str | None = None


class OpenRouterClient:
    """Small wrapper around the OpenRouter REST API.

    The client uses direct HTTP rather than a vendor SDK so notebooks can show
    the request shape clearly and tests can mock every endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        app_title: str = "ms-ai-ml-helper-core",
        http_referer: str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
        max_retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.app_title = app_title
        self.http_referer = http_referer
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: ChatModel | str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider: dict[str, Any] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Call a course-enabled chat model."""

        model_id = ensure_allowed_model(model or default_model_for(ModelPurpose.CHAT))
        payload: dict[str, Any] = {"model": model_id, "messages": messages}
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if provider is not None:
            payload["provider"] = provider
        payload.update(extra)

        data = self._post("/chat/completions", payload)
        choice = self._first_choice(data)
        message = choice.get("message") or {}
        self._raise_choice_error(choice)
        return ChatResponse(
            content=message.get("content"),
            tool_calls=list(message.get("tool_calls") or []),
            raw=data,
            usage=data.get("usage"),
            model=data.get("model"),
            finish_reason=choice.get("finish_reason"),
        )

    def structured(
        self,
        messages: list[dict[str, Any]],
        *,
        output_model: type[TModel],
        schema_name: str,
        model: ChatModel | str | None = None,
        strict: bool = True,
        provider: dict[str, Any] | None = None,
        **extra: Any,
    ) -> TModel:
        """Call a chat model with OpenRouter JSON Schema structured output."""

        model_id = ensure_allowed_model(model or default_model_for(ModelPurpose.STRUCTURED))
        provider_prefs = {"require_parameters": True}
        if provider:
            provider_prefs.update(provider)
        schema = output_model.model_json_schema()
        response = self.chat(
            messages,
            model=model_id,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": strict,
                    "schema": schema,
                },
            },
            provider=provider_prefs,
            **extra,
        )
        if response.content is None:
            raise StructuredOutputError("Structured response did not include content.")
        try:
            data = _parse_json_content(response.content)
            return output_model.model_validate(data)
        except (JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise StructuredOutputError(f"Structured response failed validation: {exc}") from exc

    def embed(
        self,
        texts: list[str] | tuple[str, ...] | str,
        *,
        model: EmbeddingModel | str | None = None,
        input_type: str | None = None,
        dimensions: int | None = None,
        provider: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with the course-enabled embedding model."""

        model_id = ensure_allowed_model(model or default_model_for(ModelPurpose.EMBEDDING))
        inputs: list[str] | str = texts if isinstance(texts, str) else list(texts)
        payload: dict[str, Any] = {"model": model_id, "input": inputs}
        if input_type is not None:
            payload["input_type"] = input_type
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if provider is not None:
            payload["provider"] = provider

        data = self._post("/embeddings", payload)
        items = sorted(data.get("data", []), key=lambda item: item.get("index", 0))
        return [list(map(float, item["embedding"])) for item in items]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        model: RerankModel | str | None = None,
        provider: dict[str, Any] | None = None,
    ) -> list[RerankResult]:
        """Rerank candidate document strings with the enabled Cohere reranker."""

        model_id = ensure_allowed_model(model or default_model_for(ModelPurpose.RERANKING))
        payload: dict[str, Any] = {"model": model_id, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        if provider is not None:
            payload["provider"] = provider

        data = self._post("/rerank", payload)
        return [
            RerankResult(
                index=int(item["index"]),
                relevance_score=float(item["relevance_score"]),
                document=item.get("document"),
            )
            for item in data.get("results", [])
        ]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is required. Set it in the environment or pass api_key."
            )
        url = f"{self.base_url}{path}"
        last_error: OpenRouterError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(url, json=payload, headers=self._headers())
                data = self._decode_json(response)
                self._raise_for_error_response(response, data)
                return data
            except OpenRouterError as exc:
                last_error = exc
                if attempt >= self.max_retries or not _is_retryable(exc.status_code or exc.code):
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
        assert last_error is not None
        raise last_error

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": self.app_title,
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        return headers

    @staticmethod
    def _decode_json(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise OpenRouterError(
                f"OpenRouter returned non-JSON response with status {response.status_code}.",
                status_code=response.status_code,
            ) from exc
        if not isinstance(data, dict):
            raise OpenRouterError("OpenRouter response was not a JSON object.")
        return data

    @staticmethod
    def _raise_for_error_response(response: httpx.Response, data: dict[str, Any]) -> None:
        if response.status_code >= 400 or "error" in data:
            error = data.get("error") if isinstance(data.get("error"), dict) else {}
            message = error.get("message") or f"OpenRouter request failed with {response.status_code}."
            code = error.get("code")
            metadata = error.get("metadata") if isinstance(error.get("metadata"), dict) else {}
            raise OpenRouterError(
                message,
                status_code=response.status_code,
                code=code,
                metadata=metadata,
            )

    @staticmethod
    def _first_choice(data: dict[str, Any]) -> dict[str, Any]:
        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter response did not include choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise OpenRouterError("OpenRouter choice was not a JSON object.")
        return choice

    @staticmethod
    def _raise_choice_error(choice: dict[str, Any]) -> None:
        error = choice.get("error")
        if isinstance(error, dict):
            raise OpenRouterError(error.get("message", "OpenRouter choice failed."), code=error.get("code"))


class OpenRouterEmbedder:
    """Adapter that gives ChromaStore a simple `.embed(texts)` interface."""

    def __init__(
        self,
        client: OpenRouterClient,
        *,
        model: EmbeddingModel | str = EmbeddingModel.GEMINI_EMBEDDING,
        input_type: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.input_type = input_type

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts, model=self.model, input_type=self.input_type)


def _parse_json_content(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return json.loads(stripped)


def _is_retryable(status_or_code: int | None) -> bool:
    return status_or_code in {408, 429, 500, 502, 503, 524, 529}
