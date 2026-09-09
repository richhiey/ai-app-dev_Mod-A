"""Tool schemas, execution, and error handling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


class ToolExecutionError(RuntimeError):
    """Raised when a tool loop cannot proceed."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_call_id: str
    name: str
    ok: bool
    content: str
    retryable: bool = False
    error_type: str | None = None

    def as_message(self) -> dict[str, str]:
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ValueError("Tool names may contain letters, numbers, underscores, and hyphens.")
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolExecutionError(f"Tool '{name}' is not registered.") from exc

    def to_openrouter_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> ToolExecutionResult:
        tool_call_id = str(tool_call.get("id") or "")
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "")
        try:
            spec = self.get(name)
            args = _parse_arguments(function.get("arguments", "{}"))
            _validate_args(spec.parameters, args)
            result = spec.handler(**args)
            return ToolExecutionResult(
                tool_call_id=tool_call_id,
                name=name,
                ok=True,
                content=_to_json_content(result),
            )
        except TimeoutError as exc:
            return _tool_error(tool_call_id, name, exc, retryable=True)
        except Exception as exc:
            return _tool_error(tool_call_id, name, exc, retryable=False)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValueError("Tool arguments must be a JSON object or JSON string.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON arguments: {exc.msg}.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to a JSON object.")
    return parsed


def _validate_args(schema: dict[str, Any], args: dict[str, Any]) -> None:
    required = schema.get("required") or []
    for key in required:
        if key not in args:
            raise ValueError(f"Missing required tool argument: {key}.")
    if schema.get("additionalProperties") is False:
        allowed = set((schema.get("properties") or {}).keys())
        extras = set(args) - allowed
        if extras:
            raise ValueError(f"Unexpected tool argument(s): {', '.join(sorted(extras))}.")
    properties = schema.get("properties") or {}
    for key, value in args.items():
        property_schema = properties.get(key) or {}
        _validate_property_schema(key, value, property_schema)


def _validate_property_schema(key: str, value: Any, property_schema: dict[str, Any]) -> None:
    if "enum" in property_schema and value not in property_schema["enum"]:
        allowed = ", ".join(map(str, property_schema["enum"]))
        raise ValueError(f"Tool argument '{key}' must be one of: {allowed}.")

    expected_type = property_schema.get("type")
    if expected_type and not _matches_json_type(value, expected_type):
        if isinstance(expected_type, list):
            expected = " or ".join(map(str, expected_type))
        else:
            expected = str(expected_type)
        raise ValueError(f"Tool argument '{key}' must be {expected}.")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = property_schema.get("minimum")
        maximum = property_schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ValueError(f"Tool argument '{key}' must be >= {minimum}.")
        if maximum is not None and value > maximum:
            raise ValueError(f"Tool argument '{key}' must be <= {maximum}.")


def _matches_json_type(value: Any, expected_type: str | list[str]) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(value, item) for item in expected_type)

    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def _to_json_content(result: Any) -> str:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    return json.dumps(result, ensure_ascii=False, default=str)


def _tool_error(
    tool_call_id: str,
    name: str,
    exc: Exception,
    *,
    retryable: bool,
) -> ToolExecutionResult:
    payload = {
        "ok": False,
        "error_type": exc.__class__.__name__,
        "message": str(exc),
        "retryable": retryable,
    }
    return ToolExecutionResult(
        tool_call_id=tool_call_id,
        name=name,
        ok=False,
        content=json.dumps(payload, ensure_ascii=False),
        retryable=retryable,
        error_type=exc.__class__.__name__,
    )
