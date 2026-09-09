"""High-level helpers for the Sprint 3 Tools and MCP Colab."""

from __future__ import annotations

import json
import os
import sys
from getpass import getpass
from typing import Any

try:
    from IPython.display import JSON, Markdown, display
except ImportError:  # pragma: no cover - used only outside notebooks
    class JSON:  # type: ignore[no-redef]
        def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
            self.data = data

    class Markdown(str):  # type: ignore[no-redef]
        pass

    def display(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        return None


from agent import ToolCallingAgent
from mcp_client import inspect_and_call_stdio_tool
from mcp_server import build_mcp_server, course_core_health, keyword_search_documents
from models import ChatModel
from openrouter import ChatResponse, OpenRouterClient
from tools import ToolExecutionResult, ToolRegistry

HELIODESK_USER_REQUEST = (
    "Can I share the customer export for workspace-acme-ops with Maya Singh "
    "at maya.singh@acme.example for an external auditor review?"
)

ALLOWED_REQUEST_TYPES = ["external_auditor_export", "customer_export", "internal_review"]

HELIODESK_POLICY_SNIPPETS = [
    (
        "External customer exports may be shared only when written account-owner "
        "authorization is on record for the workspace and requester."
    ),
    (
        "Support must use the secure customer portal for customer export delivery. "
        "Public links and ticket comments are not approved delivery paths."
    ),
    (
        "External auditor requests must be logged in the audit channel before any "
        "export package is released."
    ),
    (
        "If authorization is missing, expired, unclear, or unavailable, support must "
        "ask for written account-owner authorization or route the case."
    ),
]

AUTHORIZATION_RECORDS = {
    ("workspace-acme-ops", "maya.singh@acme.example", "external_auditor_export"): {
        "workspace_id": "workspace-acme-ops",
        "requester_email": "maya.singh@acme.example",
        "request_type": "external_auditor_export",
        "authorization": {
            "present": True,
            "authorized_by": "Jordan Lee, Account Owner",
            "recorded_on": "2026-08-29",
            "expires_on": "2026-10-31",
            "evidence_id": "AUTH-8841",
        },
        "message": "Written account-owner authorization is on record for this requester.",
        "recommended_next_step": "retrieve_policy_evidence_before_reply",
    }
}

REQUIRED_AUTHORIZATION_OUTPUT_FIELDS = {
    "status",
    "workspace_id",
    "requester_email",
    "request_type",
    "authorization",
    "message",
    "recommended_next_step",
}

RECOVERY_BY_FAILURE = {
    "invalid_input": "ask_for_missing_input",
    "malformed_response": "reject_and_log",
    "missing_data": "reject_and_log",
    "timeout": "retry_once_then_route",
    "empty_result": "honest_no_result",
    "permission_failure": "route_to_authorized_owner",
}

ACCEPTED_RECOVERY_BY_FAILURE = {
    "invalid_input": {"ask_for_missing_input", "repair_arguments_before_execution"},
    "malformed_response": {"reject_and_log"},
    "missing_data": {"reject_and_log"},
    "timeout": {"retry_once_then_route"},
    "empty_result": {"honest_no_result", "ask_for_missing_input"},
    "permission_failure": {"route_to_authorized_owner"},
}


def pretty(obj: Any) -> None:
    serializable = json.loads(json.dumps(obj, ensure_ascii=False, default=str))
    display(JSON(serializable))


def load_openrouter_key(required: bool = False) -> str | None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        try:
            from google.colab import userdata

            key = userdata.get("OPENROUTER_API_KEY")
        except Exception:
            key = None

    if not key and required:
        key = getpass("OpenRouter API key: ").strip()

    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        return key
    return None


def show_setup_status() -> bool:
    key_available = load_openrouter_key(required=False) is not None
    display(JSON({"openrouter_key_available": key_available, "value_hidden": key_available}))
    return key_available


def show_case_brief() -> dict[str, Any]:
    return {
        "business_case": "HelioDesk external auditor export request",
        "user_request": HELIODESK_USER_REQUEST,
        "sources_of_truth": [
            {
                "source": "check_export_authorization",
                "proves": "current authorization state for a workspace and requester",
            },
            {
                "source": "search_heliodesk_policy / MCP keyword_search",
                "proves": "stable policy evidence from the support handbook",
            },
        ],
        "policy_snippets": HELIODESK_POLICY_SNIPPETS,
    }


def check_export_authorization(
    workspace_id: str,
    requester_email: str,
    request_type: str,
) -> dict[str, Any]:
    workspace_id = workspace_id.strip().lower()
    requester_email = requester_email.strip().lower()

    if workspace_id == "workspace-timeout":
        raise TimeoutError("The authorization service did not respond inside the client timeout.")
    if workspace_id == "workspace-malformed-response":
        return "AUTH_OK=YES;approved_by=account_owner"
    if workspace_id == "workspace-missing-data":
        return {
            "status": "found",
            "workspace_id": workspace_id,
            "requester_email": requester_email,
            "request_type": request_type,
            "message": "A record was found, but the authorization object is missing.",
            "recommended_next_step": "stop_before_downstream_use",
        }
    if requester_email == "permission.denied@heliodesk.example":
        return {
            "status": "error",
            "workspace_id": workspace_id,
            "requester_email": requester_email,
            "request_type": request_type,
            "authorization": None,
            "message": "The caller is not allowed to inspect this authorization record.",
            "recommended_next_step": "route_to_authorized_owner",
            "error_type": "permission_failure",
        }

    record = AUTHORIZATION_RECORDS.get((workspace_id, requester_email, request_type))
    if not record:
        return {
            "status": "not_found",
            "workspace_id": workspace_id,
            "requester_email": requester_email,
            "request_type": request_type,
            "authorization": None,
            "message": "No matching written authorization record was found.",
            "recommended_next_step": "ask_for_confirmed_workspace_or_requester",
        }

    return {"status": "found", **record}


def search_heliodesk_policy(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    return keyword_search_documents(query=query, documents=HELIODESK_POLICY_SNIPPETS, top_k=top_k)


def build_heliodesk_tool_registry() -> ToolRegistry:
    authorization_parameters = {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "Confirmed HelioDesk workspace identifier, such as workspace-acme-ops.",
            },
            "requester_email": {
                "type": "string",
                "description": "Email address of the person asking for the export.",
            },
            "request_type": {
                "type": "string",
                "enum": ALLOWED_REQUEST_TYPES,
                "description": "Supported authorization request category.",
            },
        },
        "required": ["workspace_id", "requester_email", "request_type"],
        "additionalProperties": False,
    }

    policy_search_parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Policy question or source-like search text."},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 4},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    registry = ToolRegistry()
    registry.register(
        name="check_export_authorization",
        description=(
            "Check whether HelioDesk has written account-owner authorization for a customer export "
            "request from a specific workspace and requester. Use this before answering external "
            "auditor export questions when the request includes confirmed identifiers. The tool is "
            "read-only and does not create links, approve sharing, send messages, or replace policy evidence."
        ),
        parameters=authorization_parameters,
        handler=check_export_authorization,
    )
    registry.register(
        name="search_heliodesk_policy",
        description=(
            "Search the HelioDesk support policy snippets for stable handbook evidence. Use this to "
            "ground the final support answer after current authorization state is known. The tool "
            "does not check live authorization or perform any customer-data action."
        ),
        parameters=policy_search_parameters,
        handler=search_heliodesk_policy,
    )
    return registry


def display_registered_tools(registry: ToolRegistry) -> None:
    names = [tool["function"]["name"] for tool in registry.to_openrouter_tools()]
    display(Markdown("**Registered tools**\n\n" + "\n".join(f"- `{name}`" for name in names)))


def inspect_tool_contracts(registry: ToolRegistry) -> list[dict[str, Any]]:
    report = []
    for tool in registry.to_openrouter_tools():
        function = tool["function"]
        parameters = function["parameters"]
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))
        description = function.get("description", "")
        report.append(
            {
                "tool": function["name"],
                "description_words": len(description.split()),
                "has_boundary_language": "does not" in description.lower(),
                "required": sorted(required),
                "rejects_extra_args": parameters.get("additionalProperties") is False,
                "enum_fields": {
                    name: spec.get("enum")
                    for name, spec in properties.items()
                    if "enum" in spec
                },
            }
        )
    return report


def inspect_and_display_contracts(registry: ToolRegistry) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    schemas = registry.to_openrouter_tools()
    report = inspect_tool_contracts(registry)
    display(Markdown("### OpenRouter tool schemas"))
    pretty(schemas)
    display(Markdown("### Contract report"))
    pretty(report)
    return schemas, report


def make_authorization_tool_call(
    *,
    workspace_id: str = "workspace-acme-ops",
    requester_email: str = "maya.singh@acme.example",
    request_type: str = "external_auditor_export",
    call_id: str = "call_authorization_1",
    extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arguments = {
        "workspace_id": workspace_id,
        "requester_email": requester_email,
        "request_type": request_type,
    }
    if extra_args:
        arguments.update(extra_args)
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "check_export_authorization",
            "arguments": json.dumps(arguments),
        },
    }


def run_direct_authorization_check(registry: ToolRegistry) -> tuple[ToolExecutionResult, dict[str, Any]]:
    result = registry.execute_tool_call(make_authorization_tool_call())
    payload = json.loads(result.content)
    return result, payload


def display_direct_authorization_check(result: ToolExecutionResult, payload: dict[str, Any]) -> None:
    pretty(
        {
            "ok": result.ok,
            "tool_message": result.as_message(),
            "parsed_payload": payload,
        }
    )


def assert_direct_authorization_check(result: ToolExecutionResult, payload: dict[str, Any]) -> None:
    assert result.ok is True
    assert payload["status"] == "found"
    assert payload["authorization"]["evidence_id"] == "AUTH-8841"


def classify_authorization_result(result: ToolExecutionResult) -> dict[str, Any]:
    if not result.ok:
        error_payload = json.loads(result.content)
        message = error_payload.get("message", "")
        if result.retryable:
            family = "timeout"
        elif "Malformed JSON" in message or "argument" in message:
            family = "invalid_input"
        else:
            family = "malformed_response"
        return {
            "ok": False,
            "failure_family": family,
            "retryable": result.retryable,
            "downstream_safe": False,
            "message": message,
        }

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:
        return {
            "ok": True,
            "failure_family": "malformed_response",
            "retryable": False,
            "downstream_safe": False,
            "message": str(exc),
        }

    if not isinstance(payload, dict):
        return {
            "ok": True,
            "failure_family": "malformed_response",
            "retryable": False,
            "downstream_safe": False,
            "message": "Tool returned parseable JSON, but not the expected object shape.",
            "payload": payload,
        }

    missing = sorted(REQUIRED_AUTHORIZATION_OUTPUT_FIELDS - set(payload))
    if missing:
        return {
            "ok": True,
            "failure_family": "missing_data",
            "retryable": False,
            "downstream_safe": False,
            "message": f"Missing required output field(s): {missing}",
            "payload": payload,
        }

    status = payload.get("status")
    if status == "not_found":
        family = "empty_result"
        downstream_safe = False
    elif status == "error" and payload.get("error_type") == "permission_failure":
        family = "permission_failure"
        downstream_safe = False
    else:
        family = None
        downstream_safe = status == "found" and bool(payload.get("authorization"))

    return {
        "ok": True,
        "failure_family": family,
        "retryable": False,
        "downstream_safe": downstream_safe,
        "message": payload.get("message"),
        "payload": payload,
    }


def failure_case_calls() -> list[dict[str, Any]]:
    return [
        {
            "case": "invalid_input_bad_json",
            "call": {
                "id": "call_bad_json",
                "type": "function",
                "function": {"name": "check_export_authorization", "arguments": '{"workspace_id": '},
            },
        },
        {
            "case": "invalid_input_extra_arg",
            "call": make_authorization_tool_call(call_id="call_extra_arg", extra_args={"approve_share": True}),
        },
        {
            "case": "invalid_input_unsupported_request_type",
            "call": make_authorization_tool_call(call_id="call_bad_enum", request_type="public_link_request"),
        },
        {
            "case": "timeout",
            "call": make_authorization_tool_call(call_id="call_timeout", workspace_id="workspace-timeout"),
        },
        {
            "case": "malformed_response",
            "call": make_authorization_tool_call(call_id="call_malformed_response", workspace_id="workspace-malformed-response"),
        },
        {
            "case": "missing_data",
            "call": make_authorization_tool_call(call_id="call_missing_data", workspace_id="workspace-missing-data"),
        },
        {
            "case": "empty_result",
            "call": make_authorization_tool_call(
                call_id="call_empty",
                workspace_id="workspace-nova-finance",
                requester_email="sam.rivera@nova.example",
            ),
        },
        {
            "case": "permission_failure",
            "call": make_authorization_tool_call(
                call_id="call_permission",
                requester_email="permission.denied@heliodesk.example",
            ),
        },
    ]


def run_failure_scenarios(registry: ToolRegistry) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failure_evidence = []
    failure_raw = {}
    for item in failure_case_calls():
        outcome = registry.execute_tool_call(item["call"])
        classification = classify_authorization_result(outcome)
        family = classification.get("failure_family")
        row = {
            "case": item["case"],
            "ok": outcome.ok,
            "failure_family": family,
            "retryable": classification["retryable"],
            "downstream_safe": classification["downstream_safe"],
            "recovery": RECOVERY_BY_FAILURE.get(family, "continue_with_validated_result"),
        }
        failure_evidence.append(row)
        failure_raw[item["case"]] = {
            "tool_call": item["call"],
            "tool_result": {
                "name": outcome.name,
                "ok": outcome.ok,
                "content": outcome.content,
                "retryable": outcome.retryable,
                "error_type": outcome.error_type,
            },
            "classification": classification,
        }
    return failure_evidence, failure_raw


def inspect_failure_case(failure_raw: dict[str, Any], case_name: str) -> dict[str, Any]:
    if case_name not in failure_raw:
        available = ", ".join(sorted(failure_raw))
        raise ValueError(f"Unknown case. Choose one of: {available}")
    return failure_raw[case_name]


def display_failure_evidence(
    failure_evidence: list[dict[str, Any]],
    failure_raw: dict[str, Any],
    *,
    focus_case: str = "missing_data",
) -> None:
    display(Markdown("### Failure evidence"))
    pretty(failure_evidence)
    display(Markdown(f"### Inspect one failure: `{focus_case}`"))
    pretty(inspect_failure_case(failure_raw, focus_case))


def assert_failure_scenarios(failure_evidence: list[dict[str, Any]]) -> None:
    observed = {row["failure_family"] for row in failure_evidence if row["failure_family"]}
    expected = {"invalid_input", "malformed_response", "missing_data", "timeout", "empty_result", "permission_failure"}
    assert expected.issubset(observed)


def review_recovery_plan(
    failure_evidence: list[dict[str, Any]],
    decisions: dict[str, str],
) -> list[dict[str, Any]]:
    observed = sorted({row["failure_family"] for row in failure_evidence if row["failure_family"]})
    review = []
    for family in observed:
        accepted = ACCEPTED_RECOVERY_BY_FAILURE[family]
        selected = decisions.get(family, "")
        review.append(
            {
                "failure_family": family,
                "selected_recovery": selected or "<missing>",
                "passes": selected in accepted,
                "expected_pattern": " or ".join(sorted(accepted)),
            }
        )
    return review


def display_recovery_review(recovery_review: list[dict[str, Any]]) -> None:
    display(Markdown("### Recovery review"))
    pretty(recovery_review)


class ScriptedHelioDeskClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls += 1
        tool_messages = [message for message in messages if message.get("role") == "tool"]

        if not tool_messages:
            return ChatResponse(
                content=None,
                tool_calls=[
                    {
                        "id": "call_auth_from_model",
                        "type": "function",
                        "function": {
                            "name": "check_export_authorization",
                            "arguments": json.dumps(
                                {
                                    "workspace_id": "workspace-acme-ops",
                                    "requester_email": "maya.singh@acme.example",
                                    "request_type": "external_auditor_export",
                                }
                            ),
                        },
                    }
                ],
                raw={"source": "scripted-first-call"},
            )

        last_tool = tool_messages[-1]
        if last_tool["name"] == "check_export_authorization":
            payload = json.loads(last_tool["content"])
            if payload.get("status") == "found":
                return ChatResponse(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_policy_from_model",
                            "type": "function",
                            "function": {
                                "name": "search_heliodesk_policy",
                                "arguments": json.dumps(
                                    {
                                        "query": "external auditor export written authorization secure portal audit channel",
                                        "top_k": 3,
                                    }
                                ),
                            },
                        }
                    ],
                    raw={"source": "scripted-second-call"},
                )
            return ChatResponse(
                content="Stop before sharing: current authorization is not confirmed.",
                tool_calls=[],
                raw={"source": "scripted-stop"},
            )

        policy_results = json.loads(last_tool["content"])
        top_policy = policy_results[0]["text"] if policy_results else "No policy evidence found."
        return ChatResponse(
            content=(
                "Authorization is on record for the Acme requester. Use the policy evidence before replying: "
                f"{top_policy} Do not share through a public link or ticket comment."
            ),
            tool_calls=[],
            raw={"source": "scripted-final"},
        )


def summarize_tool_trace(run: Any) -> list[dict[str, Any]]:
    return [
        {
            "step": index,
            "tool": result.name,
            "ok": result.ok,
            "content_preview": result.content[:500],
        }
        for index, result in enumerate(run.tool_results, start=1)
    ]


def run_scripted_tool_loop(registry: ToolRegistry) -> tuple[Any, list[dict[str, Any]]]:
    scripted_agent = ToolCallingAgent(
        client=ScriptedHelioDeskClient(),
        tools=registry,
        model=ChatModel.GEMINI_31_FLASH_LITE,
        max_steps=4,
        system_prompt=(
            "You are the HelioDesk support assistant. Check current authorization first, "
            "then retrieve policy evidence before writing a final answer."
        ),
    )
    run = scripted_agent.run(HELIODESK_USER_REQUEST)
    return run, summarize_tool_trace(run)


def display_scripted_tool_loop(run: Any, tool_trace: list[dict[str, Any]]) -> None:
    display(Markdown(run.final_content or "[no final content returned]"))
    display(JSON({"tool_calls_executed": len(run.tool_results), "tool_trace": tool_trace}))


def run_live_tool_loop(registry: ToolRegistry, *, enabled: bool = False) -> Any | None:
    if not enabled:
        display(Markdown("Skipped. Set `RUN_LIVE_OPENROUTER = True` when you want to spend one live model run."))
        return None
    if not load_openrouter_key(required=True):
        display(Markdown("No OpenRouter key loaded."))
        return None

    live_client = OpenRouterClient(app_title="ai-app-dev-module-a-sprint-3")
    live_agent = ToolCallingAgent(
        client=live_client,
        tools=registry,
        model=ChatModel.GEMINI_31_FLASH_LITE,
        max_steps=4,
        system_prompt=(
            "You are the HelioDesk support assistant. For external auditor export questions, "
            "check current authorization and retrieve policy evidence before answering. "
            "Do not invent missing authorization or delivery rules."
        ),
    )
    live_run = live_agent.run(HELIODESK_USER_REQUEST)
    display(Markdown(live_run.final_content or "[no text returned]"))
    display(JSON({"tool_calls_executed": len(live_run.tool_results), "tool_trace": summarize_tool_trace(live_run)}))
    return live_run


async def connect_to_heliodesk_policy_mcp() -> tuple[Any, dict[str, Any]]:
    server_preview = build_mcp_server("heliodesk-policy-mcp-demo")
    health = course_core_health()
    demo = await inspect_and_call_stdio_tool(
        command=sys.executable,
        args=["-m", "mcp_server"],
        tool_name="keyword_search",
        arguments={
            "query": "external auditor export authorization secure portal audit channel",
            "documents": HELIODESK_POLICY_SNIPPETS,
            "top_k": 3,
        },
    )
    summary = {
        "created_mcp_server_object": type(server_preview).__name__,
        "helper_server_health": health,
        "connected_server": demo.server_name,
        "advertised_tools": [tool.name for tool in demo.tools],
        "called_tool": demo.call.tool_name,
        "call_is_error": demo.call.is_error,
    }
    return demo, summary


def parse_mcp_tool_payload(content_texts: list[str]) -> list[Any]:
    rows = []
    for text in content_texts:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            rows.extend(parsed)
        else:
            rows.append(parsed)
    if not rows:
        raise ValueError("MCP tool response did not contain parseable JSON text.")
    return rows


def validate_mcp_policy_response(mcp_demo: Any) -> tuple[list[dict[str, Any]], str]:
    tool_names = [tool.name for tool in mcp_demo.tools]
    assert mcp_demo.server_name == "ms-ai-ml-helper-core"
    assert "health" in tool_names
    assert "keyword_search" in tool_names
    assert mcp_demo.call.tool_name == "keyword_search"
    assert mcp_demo.call.is_error is False
    assert mcp_demo.call.content_texts

    mcp_payload = parse_mcp_tool_payload(mcp_demo.call.content_texts)
    validated_rows = [
        {
            "index": int(row["index"]),
            "score": float(row["score"]),
            "text": str(row["text"]),
        }
        for row in mcp_payload
    ]
    downstream_note = (
        "Use the direct authorization result as current-state evidence and the MCP keyword_search "
        "result as handbook evidence. The downstream response may proceed only if both are present "
        "and the delivery rule is supported by the policy rows."
    )
    return validated_rows, downstream_note


def display_mcp_validation(validated_rows: list[dict[str, Any]], downstream_note: str) -> None:
    display(Markdown("### Validated MCP rows"))
    pretty(validated_rows)
    display(Markdown("**Downstream note**\n\n" + downstream_note))


def review_decision_note(note: str) -> dict[str, Any]:
    lowered = note.lower()
    checks = {
        "mentions_contract": "contract" in lowered or "required" in lowered,
        "mentions_failure": "failure" in lowered or "timeout" in lowered or "empty result" in lowered,
        "mentions_mcp_boundary": "mcp" in lowered,
        "mentions_direct_boundary": "direct" in lowered or "authorization" in lowered,
        "long_enough": len(note.split()) >= 45,
    }
    return {"passes": all(checks.values()), "checks": checks, "note": note}


def build_task_answer_signals() -> dict[str, str]:
    return {
        "task_a_direct_tool_contract": (
            "check_export_authorization is a read-only direct tool. It requires workspace_id, "
            "requester_email, and request_type, rejects unsupported fields or request types, "
            "and returns status='found' with authorization evidence AUTH-8841 for the HelioDesk case. "
            "This proves current authorization state only, not the full sharing decision."
        ),
        "task_b_failure_handling": (
            "invalid_input should be repaired or clarified before execution; malformed_response "
            "and missing_data should be rejected and logged; timeout can be retried once for this "
            "read-only fixture before routing; empty_result is not authorization; permission_failure "
            "routes to an authorized owner."
        ),
        "task_c_multi_step_and_mcp": (
            "The deterministic loop should call check_export_authorization before search_heliodesk_policy. "
            "The MCP server should advertise keyword_search over stdio, and validated_mcp_rows should contain "
            "policy evidence. Direct tool output proves current state; MCP output supplies policy evidence."
        ),
    }


def build_checkpoint_evidence(
    *,
    contract_report: list[dict[str, Any]],
    authorization_result: ToolExecutionResult,
    authorization_payload: dict[str, Any],
    failure_evidence: list[dict[str, Any]],
    recovery_review: list[dict[str, Any]],
    scripted_run: Any,
    mcp_demo: Any,
    validated_mcp_rows: list[dict[str, Any]],
    mcp_downstream_note: str,
    decision_note_review: dict[str, Any],
) -> dict[str, Any]:
    return {
        "business_case": "HelioDesk external auditor export request",
        "task_a_direct_tool_contract": {
            "direct_tool_contracts": contract_report,
            "direct_tool_success": {
                "tool": authorization_result.name,
                "ok": authorization_result.ok,
                "status": authorization_payload["status"],
                "evidence_id": authorization_payload["authorization"]["evidence_id"],
            },
        },
        "task_b_failure_handling": {
            "failure_behavior": failure_evidence,
            "recovery_notes": recovery_review,
        },
        "task_c_multi_step_and_mcp": {
            "multi_step_tool_loop": {
                "steps": scripted_run.steps,
                "tools_called": [result.name for result in scripted_run.tool_results],
                "final_content_preview": scripted_run.final_content,
            },
            "mcp_connection": {
                "server": mcp_demo.server_name,
                "advertised_tools": [tool.name for tool in mcp_demo.tools],
                "called_tool": mcp_demo.call.tool_name,
                "validated_rows": validated_mcp_rows,
            },
            "downstream_rule": mcp_downstream_note,
            "decision_note": decision_note_review,
        },
    }


def validate_checkpoint_evidence(
    *,
    authorization_payload: dict[str, Any],
    recovery_review: list[dict[str, Any]],
    scripted_run: Any,
    validated_mcp_rows: list[dict[str, Any]],
    decision_note_review: dict[str, Any],
) -> None:
    assert authorization_payload["authorization"]["evidence_id"] == "AUTH-8841"
    assert all(row["passes"] for row in recovery_review)
    assert [result.name for result in scripted_run.tool_results] == [
        "check_export_authorization",
        "search_heliodesk_policy",
    ]
    assert validated_mcp_rows
    assert decision_note_review["passes"]


__all__ = [
    "HELIODESK_USER_REQUEST",
    "pretty",
    "show_setup_status",
    "show_case_brief",
    "build_heliodesk_tool_registry",
    "display_registered_tools",
    "inspect_and_display_contracts",
    "run_direct_authorization_check",
    "display_direct_authorization_check",
    "assert_direct_authorization_check",
    "run_failure_scenarios",
    "display_failure_evidence",
    "assert_failure_scenarios",
    "review_recovery_plan",
    "display_recovery_review",
    "run_scripted_tool_loop",
    "display_scripted_tool_loop",
    "run_live_tool_loop",
    "connect_to_heliodesk_policy_mcp",
    "validate_mcp_policy_response",
    "display_mcp_validation",
    "review_decision_note",
    "build_task_answer_signals",
    "build_checkpoint_evidence",
    "validate_checkpoint_evidence",
]
