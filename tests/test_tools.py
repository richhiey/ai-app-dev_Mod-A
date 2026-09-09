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


def test_tool_registry_rejects_enum_values_before_handler_runs() -> None:
    handler_ran = False

    def handler(request_type: str):
        nonlocal handler_ran
        handler_ran = True
        return {"request_type": request_type}

    registry = ToolRegistry()
    registry.register(
        name="authorization_lookup",
        description="Check authorization.",
        parameters={
            "type": "object",
            "properties": {
                "request_type": {
                    "type": "string",
                    "enum": ["external_auditor_export", "internal_review"],
                }
            },
            "required": ["request_type"],
            "additionalProperties": False,
        },
        handler=handler,
    )

    result = registry.execute_tool_call(
        {
            "id": "call_bad_enum",
            "function": {
                "name": "authorization_lookup",
                "arguments": '{"request_type":"public_link"}',
            },
        }
    )

    assert result.ok is False
    assert handler_ran is False
    assert "must be one of" in result.content


def test_tool_registry_rejects_basic_type_mismatches() -> None:
    registry = ToolRegistry()
    registry.register(
        name="top_k_lookup",
        description="Lookup with a result limit.",
        parameters={
            "type": "object",
            "properties": {"top_k": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["top_k"],
            "additionalProperties": False,
        },
        handler=lambda top_k: {"top_k": top_k},
    )

    wrong_type = registry.execute_tool_call(
        {
            "id": "call_wrong_type",
            "function": {"name": "top_k_lookup", "arguments": '{"top_k":"2"}'},
        }
    )
    out_of_range = registry.execute_tool_call(
        {
            "id": "call_out_of_range",
            "function": {"name": "top_k_lookup", "arguments": '{"top_k":6}'},
        }
    )

    assert wrong_type.ok is False
    assert "must be integer" in wrong_type.content
    assert out_of_range.ok is False
    assert "must be <= 5" in out_of_range.content
