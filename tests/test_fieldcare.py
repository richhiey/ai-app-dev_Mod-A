import asyncio
import json
from pathlib import Path

import pytest

from fieldcare import (
    FIELDCARE_DIRECT_TOOLS,
    FieldCareArtifactError,
    artifact_record,
    build_fieldcare_direct_registry,
    inspect_and_call_fieldcare_mcp,
    search_fieldcare_service_docs,
    validate_fieldcare_artifact,
    write_fieldcare_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "data" / "fieldcare"


def app_contract() -> dict:
    return {
        "schema_version": "1.0",
        "project_id": "fieldcare",
        "supported_request_types": ["troubleshooting_plus_warranty"],
        "required_inputs": ["request_text", "equipment_id", "ticket_id"],
        "response_contract": {"required_fields": ["answer", "evidence"]},
        "routing_rules": [
            {"when": "safety_indicator", "route_to": "Safety Engineering"}
        ],
        "success_criteria": [
            "Every factual claim is traceable to a document or tool result."
        ],
    }


def retrieval_config() -> dict:
    return {
        "schema_version": "1.0",
        "project_id": "fieldcare",
        "source_revision": "fieldcare-assets-v1.0.0",
        "chunking": {"strategy": "document", "max_chars": 1200},
        "baseline_config": {"method": "bm25", "top_k": 8},
        "hybrid_config": {"alpha": 0.65, "top_k": 8},
        "rerank_config": {"top_n": 5, "reject_superseded": True},
        "selected_config": {"retrieval": "hybrid", "rerank": True},
        "evaluation_queries": [
            {"query_id": "fc-1", "query": "overheating after filter work"}
        ],
        "evidence_summary": [
            {
                "query_id": "fc-1",
                "before_doc_ids": ["DOC-FC-TS-006"],
                "after_doc_ids": ["DOC-FC-TS-001"],
            }
        ],
    }


def capability_contracts() -> dict:
    direct_tools = [
        {"name": name, "purpose": "Read FieldCare state."}
        for name in FIELDCARE_DIRECT_TOOLS
    ]
    return {
        "schema_version": "1.0",
        "project_id": "fieldcare",
        "direct_tools": direct_tools,
        "mcp_server": {
            "name": "fieldcare-service-docs",
            "tools": [{"name": "search_service_docs"}],
        },
        "orchestration_rules": [
            {
                "request_type": "troubleshooting_plus_warranty",
                "required_inputs": ["request_text", "equipment_id"],
                "available_inputs_before_call": [
                    "request_text",
                    "equipment_id",
                    "ticket_id",
                ],
                "call_order": [
                    "search_service_docs",
                    "get_equipment_record",
                    "get_warranty_status",
                ],
            }
        ],
        "failure_policy": {"missing_input": "ask", "timeout": "retry_once_then_route"},
        "trace_evidence": [
            {
                "tool": "search_service_docs",
                "status": "ok",
                "doc_ids": ["DOC-FC-TS-001"],
            }
        ],
    }


def test_fieldcare_direct_tools_exclude_document_search() -> None:
    registry = build_fieldcare_direct_registry(ASSET_DIR)
    names = [item["function"]["name"] for item in registry.to_openrouter_tools()]

    assert names == list(FIELDCARE_DIRECT_TOOLS)
    assert "search_service_docs" not in names
    result = registry.execute_tool_call(
        {
            "id": "equipment-1",
            "function": {
                "name": "get_equipment_record",
                "arguments": json.dumps({"equipment_id": "EQ-FC-1001"}),
            },
        }
    )
    assert result.ok is True
    assert json.loads(result.content)["data"]["model"] == "HX-200"


def test_fieldcare_document_search_is_exposed_only_by_mcp() -> None:
    local_rows = search_fieldcare_service_docs(
        "HX overheating after filter replacement airflow",
        top_k=3,
        asset_dir=ASSET_DIR,
    )
    assert local_rows[0]["doc_id"] in {"DOC-FC-TS-001", "DOC-FC-MP-014"}

    result = asyncio.run(
        inspect_and_call_fieldcare_mcp(
            query="HX overheating after filter replacement airflow",
            top_k=3,
            asset_dir=ASSET_DIR,
        )
    )
    assert [tool.name for tool in result.tools] == ["search_service_docs"]
    assert result.call.is_error is False
    assert "DOC-FC-" in "".join(result.call.content_texts)


def test_incomplete_warranty_record_is_not_reported_as_confirmed_coverage():
    result = (
        build_fieldcare_direct_registry(ASSET_DIR)
        .get("get_warranty_status")
        .handler(equipment_id="EQ-FC-1006")
    )
    assert result["status"] == "unavailable"
    assert result["data"] is None
    assert result["error_type"] == "incomplete_record"


def test_fieldcare_artifacts_validate_and_record_digests(tmp_path) -> None:
    artifacts = [
        ("app_contract", "fieldcare_app_contract.json", app_contract()),
        ("retrieval_config", "fieldcare_retrieval_config.json", retrieval_config()),
        (
            "capability_contracts",
            "fieldcare_capability_contracts.json",
            capability_contracts(),
        ),
    ]
    records = []
    for artifact_type, filename, payload in artifacts:
        path = write_fieldcare_artifact(payload, artifact_type, tmp_path / filename)
        records.append(artifact_record(path))

    manifest = {
        "schema_version": "1.0",
        "project_id": "fieldcare",
        "artifacts": records,
        "final_trace": {"request_id": "REQ-FC-001", "status": "completed"},
        "evaluation_summary": {"passed": 14, "total": 16},
        "selected_improvement": {
            "before_evidence": "DOC-FC-LEG-2019 ranked first.",
            "after_evidence": "DOC-FC-TS-001 ranked first and the legacy document was rejected.",
        },
        "limitations": ["Live model output still requires evidence review."],
    }
    validate_fieldcare_artifact(manifest, "submission_manifest")


def test_capability_artifact_rejects_mcp_direct_tool_overlap() -> None:
    payload = capability_contracts()
    payload["mcp_server"]["tools"].append({"name": "get_ticket_status"})

    with pytest.raises(FieldCareArtifactError, match="must not overlap"):
        validate_fieldcare_artifact(payload, "capability_contracts")


def test_artifact_rejects_unresolved_learner_placeholder() -> None:
    payload = app_contract()
    payload["success_criteria"] = ["Learner task: write a criterion"]

    with pytest.raises(FieldCareArtifactError, match="unresolved learner placeholder"):
        validate_fieldcare_artifact(payload, "app_contract")
