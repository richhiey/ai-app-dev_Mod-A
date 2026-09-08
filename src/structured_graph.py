"""LangGraph workflow for structured LLM outputs."""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel

from models import ChatModel


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


class _GraphState(TypedDict, total=False):
    prompt: str
    messages: list[dict[str, Any]]
    result: BaseModel


class StructuredOutputGraph:
    """Wrap a single structured-output model call in a LangGraph graph."""

    def __init__(
        self,
        *,
        client: StructuredClient,
        output_model: type[BaseModel],
        schema_name: str,
        system_prompt: str,
        model: ChatModel | str = ChatModel.GEMINI_31_FLASH_LITE,
    ) -> None:
        self.client = client
        self.output_model = output_model
        self.schema_name = schema_name
        self.system_prompt = system_prompt
        self.model = model
        self._graph = self._build_graph()

    def invoke(self, prompt: str) -> BaseModel:
        state = self._graph.invoke({"prompt": prompt})
        return state["result"]

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise ImportError("Install langgraph to use StructuredOutputGraph.") from exc

        def prepare_messages(state: _GraphState) -> _GraphState:
            return {
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": state["prompt"]},
                ]
            }

        def call_model(state: _GraphState) -> _GraphState:
            result = self.client.structured(
                state["messages"],
                output_model=self.output_model,
                schema_name=self.schema_name,
                model=self.model,
            )
            return {"result": result}

        builder = StateGraph(_GraphState)
        builder.add_node("prepare_messages", prepare_messages)
        builder.add_node("call_model", call_model)
        builder.set_entry_point("prepare_messages")
        builder.add_edge("prepare_messages", "call_model")
        builder.add_edge("call_model", END)
        return builder.compile()
