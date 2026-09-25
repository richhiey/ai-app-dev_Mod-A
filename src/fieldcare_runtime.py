"""Execute the FieldCare sprint contracts through direct tools and MCP."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from fieldcare import (
    FIELDCARE_DIRECT_TOOLS,
    FieldCareArtifactError,
    call_fieldcare_mcp,
    find_fieldcare_asset_dir,
    validate_fieldcare_artifact,
    validate_retrieval_settings,
)


def configure_contracts(
    pipeline: Any, app: dict, retrieval: dict, capabilities: dict
) -> None:
    for payload, kind in [
        (app, "app_contract"),
        (retrieval, "retrieval_config"),
        (capabilities, "capability_contracts"),
    ]:
        validate_fieldcare_artifact(payload, kind)
    validate_retrieval_settings(retrieval)
    unknown_rerank = set(retrieval["rerank_config"]) - pipeline.rerank_config.keys()
    if unknown_rerank:
        raise FieldCareArtifactError(
            "Unsupported reranking settings: " + ", ".join(sorted(unknown_rerank))
        )
    pipeline.rerank_config.update(deepcopy(retrieval["rerank_config"]))
    rules = capabilities["orchestration_rules"]
    by_type = {rule.get("request_type"): rule for rule in rules}
    if len(by_type) != len(rules) or any(
        name not in by_type for name in app["supported_request_types"]
    ):
        raise FieldCareArtifactError(
            "Provide one orchestration rule for every supported request type."
        )
    allowed = {*FIELDCARE_DIRECT_TOOLS, "search_service_docs"}
    for rule in rules:
        order = rule.get("call_order")
        if (
            not isinstance(order, list)
            or not order
            or any(name not in allowed for name in order)
        ):
            raise FieldCareArtifactError(
                "Every call_order must contain known capabilities."
            )
        if len(order) != len(set(order)):
            raise FieldCareArtifactError("call_order must not duplicate capabilities.")
        if not set(rule["required_inputs"]) <= set(
            rule["available_inputs_before_call"]
        ):
            raise FieldCareArtifactError(
                "Required inputs must be available before the call."
            )
    if set(app["supported_request_types"]) & set(
        app.get("out_of_scope_request_types", [])
    ):
        raise FieldCareArtifactError(
            "Supported and out-of-scope request types must not overlap."
        )
    for rule in app["routing_rules"]:
        if rule.get("when") not in {
            "safety_indicator",
            "repeat_fault",
            "warranty_unavailable",
            "permission_failure",
            "unsupported_model",
        } or not rule.get("route_to"):
            raise FieldCareArtifactError(
                "Unsupported routing condition or missing route_to."
            )
    policy = capabilities["failure_policy"]
    for key, choices in {
        "missing_input": {"ask", "abstain"},
        "timeout": {"retry_once_then_route", "abstain_and_route"},
        "unavailable": {"abstain_and_route"},
    }.items():
        if policy.get(key) not in choices:
            raise FieldCareArtifactError(
                f"Unsupported or missing failure_policy.{key}."
            )
    pipeline.app_contract = deepcopy(app)
    pipeline.retrieval_contract = deepcopy(retrieval)
    pipeline.capability_contract = deepcopy(capabilities)
    pipeline.contract_rules = deepcopy(by_type)
    pipeline.asset_dir = find_fieldcare_asset_dir()
    pipeline.last_execution = None


def request_boundary(pipeline: Any, request: dict, ids: dict) -> tuple[bool, list[str]]:
    app = pipeline.app_contract
    supported = request["request_type"] in app["supported_request_types"]
    rule = pipeline.contract_rules.get(request["request_type"], {})
    required = set(app["required_inputs"]) | set(rule.get("required_inputs", []))
    values = {**request, **ids}
    return supported, sorted(key for key in required if not values.get(key))


def execute_contract_plan(
    pipeline: Any, request: dict, ids: dict, build_args: Callable, execute: Callable
) -> list[dict]:
    supported, missing = request_boundary(pipeline, request, ids)
    trace = {
        "request": deepcopy(request),
        "supported": supported,
        "missing_inputs": missing,
        "calls": [],
        "mcp_rows": [],
        "failed": False,
    }
    pipeline.last_execution = trace
    if not supported or missing:
        return []
    results = []
    config = pipeline.retrieval_contract
    for name in pipeline.contract_rules[request["request_type"]]["call_order"]:
        if name == "search_service_docs":
            attempts = (
                2
                if pipeline.capability_contract["failure_policy"]["timeout"]
                == "retry_once_then_route"
                else 1
            )
            rows = []
            succeeded = False
            for attempt in range(attempts):
                try:
                    rows = call_fieldcare_mcp(
                        query=request["request_text"],
                        top_k=config["baseline_config"]["candidate_k"],
                        asset_dir=pipeline.asset_dir,
                        retrieval_config=config,
                    )
                    succeeded = True
                    break
                except Exception as exc:
                    status = (
                        "timeout" if isinstance(exc, TimeoutError) else "unavailable"
                    )
                    trace["calls"].append(
                        {
                            "boundary": "mcp",
                            "tool": name,
                            "query": request["request_text"],
                            "status": status,
                            "attempt": attempt + 1,
                        }
                    )
                    if status != "timeout" or attempt + 1 == attempts:
                        trace["failed"] = True
                        break
            if not succeeded:
                continue
            trace["mcp_rows"] = rows
            trace["calls"].append(
                {
                    "boundary": "mcp",
                    "tool": name,
                    "query": request["request_text"],
                    "status": "ok",
                    "doc_ids": [row["doc_id"] for row in rows],
                    "chunk_ids": [row["chunk_id"] for row in rows],
                }
            )
            continue
        args = build_args(pipeline, name, request, results)
        if args is None:
            trace["calls"].append(
                {"boundary": "direct", "tool": name, "status": "not_applicable"}
            )
            continue
        attempts = (
            2
            if pipeline.capability_contract["failure_policy"]["timeout"]
            == "retry_once_then_route"
            else 1
        )
        for attempt in range(attempts):
            result = execute(pipeline, name, args, len(trace["calls"]) + 1)
            status = result["result"]["status"]
            trace["calls"].append(
                {
                    "boundary": "direct",
                    "tool": name,
                    "status": status,
                    "attempt": attempt + 1,
                }
            )
            if status != "timeout":
                break
        results.append(result)
        if status in {
            "timeout",
            "unavailable",
            "permission_denied",
            "invalid_input",
            "not_found",
        }:
            trace["failed"] = True
    return results


def contract_evidence(
    pipeline: Any, request: dict, tool_results: list, rerank: Callable
) -> list[dict]:
    trace = pipeline.last_execution
    if trace is None or trace["request"] != request:
        raise RuntimeError(
            "Execute the request's contract plan before selecting evidence."
        )
    if not trace["supported"] or trace["missing_inputs"]:
        return []
    rows = deepcopy(trace["mcp_rows"])
    equipment = next(
        (
            row["result"]["data"]
            for row in tool_results
            if row["tool_name"] == "get_equipment_record"
            and row["result"]["status"] == "found"
        ),
        None,
    )
    selected = pipeline.retrieval_contract["selected_config"]
    if selected["rerank"]:
        rows = rerank(
            pipeline, request["request_text"], rows, equipment_record=equipment
        )
    else:
        rows = [{**row, "rerank_score": row["baseline_score"]} for row in rows]
    accepted, seen = [], set()
    for row in rows:
        if not row["current"] or row["doc_id"] in seen:
            continue
        if (
            equipment
            and row["applies_to_models"]
            and equipment["model"] not in row["applies_to_models"]
        ):
            continue
        accepted.append(row)
        seen.add(row["doc_id"])
        if len(accepted) == selected["accepted_k"]:
            break
    return accepted


def contract_context(pipeline: Any, request: dict, context: dict) -> dict:
    trace = pipeline.last_execution
    if trace is None or trace["request"] != request:
        raise RuntimeError("Context must use the execution trace for this request.")
    flags = set(context["decision_flags"])
    if not trace["supported"]:
        flags.update(["scope_boundary", "abstain"])
    if trace["missing_inputs"]:
        flags.add("abstain")
        if pipeline.capability_contract["failure_policy"]["missing_input"] == "ask":
            flags.add("ask_for_more_information")
    if trace["failed"]:
        flags.update(["mention_uncertainty", "abstain", "escalate"])
    context["decision_flags"] = sorted(flags)
    context["missing_inputs"] = trace["missing_inputs"]
    context["execution_trace"] = deepcopy(trace["calls"])
    return context


def contract_response(
    pipeline: Any, context: dict, response: dict, reason: str | None
) -> dict:
    """Reject model output that violates the contract instead of rewriting its answer."""
    if not set(context["decision_flags"]) <= set(response["decision_flags"]):
        raise FieldCareArtifactError("Model response omitted required decision flags.")
    for rule in pipeline.app_contract["routing_rules"]:
        if rule["when"] == reason:
            if response["escalation_path"] != rule["route_to"]:
                raise FieldCareArtifactError(
                    "Model response does not follow the contracted escalation route."
                )
            break
    missing = (
        set(pipeline.app_contract["response_contract"].get("required_fields", []))
        - response.keys()
    )
    if missing:
        raise FieldCareArtifactError(
            "Response is missing contracted fields: " + ", ".join(sorted(missing))
        )
    return response
