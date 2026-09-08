"""Minimal multi-step tool-calling loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from models import ChatModel
from openrouter import ChatResponse
from tools import ToolExecutionError, ToolExecutionResult, ToolRegistry


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        ...


@dataclass(frozen=True)
class AgentRunResult:
    final_content: str | None
    messages: list[dict[str, Any]]
    tool_results: list[ToolExecutionResult]
    steps: int


class ToolCallingAgent:
    """Run model -> tools -> model loops with explicit tool error messages."""

    def __init__(
        self,
        *,
        client: ChatClient,
        tools: ToolRegistry,
        model: ChatModel | str = ChatModel.GEMINI_31_FLASH_LITE,
        max_steps: int = 6,
        system_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.system_prompt = system_prompt

    def run(self, prompt_or_messages: str | list[dict[str, Any]]) -> AgentRunResult:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        messages = self._initial_messages(prompt_or_messages)
        tool_results: list[ToolExecutionResult] = []

        for step in range(1, self.max_steps + 1):
            response = self.client.chat(
                messages,
                model=self.model,
                tools=self.tools.to_openrouter_tools(),
                tool_choice="auto",
            )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = response.tool_calls
            messages.append(assistant_message)

            if not response.tool_calls:
                return AgentRunResult(
                    final_content=response.content,
                    messages=messages,
                    tool_results=tool_results,
                    steps=step,
                )

            for tool_call in response.tool_calls:
                result = self.tools.execute_tool_call(tool_call)
                tool_results.append(result)
                messages.append(result.as_message())

        raise ToolExecutionError(f"Tool loop exceeded max_steps={self.max_steps}.")

    def _initial_messages(self, prompt_or_messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if isinstance(prompt_or_messages, str):
            messages.append({"role": "user", "content": prompt_or_messages})
        else:
            messages.extend(prompt_or_messages)
        return messages
