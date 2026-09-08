import json

from tools import ToolRegistry


def test_tool_registry_exposes_openrouter_tool_schema_and_executes_call() -> None:
    registry = ToolRegistry()
    registry.register(
        name="lesson_lookup",
        description="Look up a lesson by topic.",
        parameters={
            "type": "object",
            "properties": {"topic": {"type": "string"}},
            "required": ["topic"],
            "additionalProperties": False,
        },
        handler=lambda topic: {"topic": topic, "lesson": "Hybrid Search"},
    )

    tools = registry.to_openrouter_tools()
    result = registry.execute_tool_call(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lesson_lookup", "arguments": '{"topic":"hybrid"}'},
        }
    )

    assert tools[0]["function"]["name"] == "lesson_lookup"
    assert result.ok is True
    assert json.loads(result.content)["lesson"] == "Hybrid Search"
    assert result.as_message()["tool_call_id"] == "call_1"


def test_tool_registry_returns_structured_error_for_malformed_arguments() -> None:
    registry = ToolRegistry()
    registry.register(
        name="lookup",
        description="Lookup.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda: "ok",
    )

    result = registry.execute_tool_call(
        {"id": "call_bad", "function": {"name": "lookup", "arguments": "{bad json"}}
    )

    assert result.ok is False
    assert result.retryable is False
    assert "Malformed JSON" in result.content


def test_tool_registry_marks_timeout_errors_retryable() -> None:
    def timeout_tool():
        raise TimeoutError("upstream timeout")

    registry = ToolRegistry()
    registry.register(
        name="slow_tool",
        description="Slow.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=timeout_tool,
    )

    result = registry.execute_tool_call(
        {"id": "call_timeout", "function": {"name": "slow_tool", "arguments": "{}"}}
    )

    assert result.ok is False
    assert result.retryable is True
    assert "upstream timeout" in result.content
