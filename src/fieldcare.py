"""FieldCare project assets, direct tools, MCP search, and artifact validation."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from documents import Document, chunk_text
from keyword_search import BM25Retriever
from mcp_client import MCPInspectAndCallResult, inspect_and_call_stdio_tool
from tools import ToolRegistry

FIELDCARE_PROJECT_ID = "fieldcare"
FIELDCARE_SCHEMA_VERSION = "1.0"
FIELDCARE_RELEASE_REF = "main"
FIELDCARE_ASSET_FILES = (
    "equipment_records.csv",
    "eval_cases.jsonl",
    "fieldcare_manifest.json",
    "maintenance_history.csv",
    "service_docs.jsonl",
    "service_tickets.csv",
    "tool_schemas.json",
    "user_requests.jsonl",
)
FIELDCARE_ARTIFACT_FILENAMES = (
    "fieldcare_app_contract.json",
    "fieldcare_retrieval_config.json",
    "fieldcare_capability_contracts.json",
)
FIELDCARE_DIRECT_TOOLS = (
    "get_equipment_record",
    "get_warranty_status",
    "get_maintenance_history",
    "get_ticket_status",
    "recommend_escalation_path",
)


class FieldCareArtifactError(ValueError):
    """Raised when a FieldCare project artifact is incomplete or inconsistent."""


def find_fieldcare_asset_dir(candidate: str | Path | None = None) -> Path:
    candidates = [
        candidate,
        os.getenv("FIELDCARE_ASSET_DIR"),
        Path(__file__).resolve().parents[1] / "data" / "fieldcare",
        Path.cwd() / "data" / "fieldcare",
        Path.cwd() / "fieldcare-assets",
    ]
    for item in candidates:
        if not item:
            continue
        path = Path(item).expanduser().resolve()
        if all((path / name).is_file() for name in FIELDCARE_ASSET_FILES):
            return path
    raise FileNotFoundError(
        "FieldCare assets were not found. Set FIELDCARE_ASSET_DIR or download the course assets."
    )


def download_fieldcare_assets(
    destination: str | Path = "fieldcare-assets",
    *,
    repo_ref: str = FIELDCARE_RELEASE_REF,
) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    local_source = Path(__file__).resolve().parents[1] / "data" / "fieldcare"
    base_url = (
        "https://raw.githubusercontent.com/richhiey/ai-app-dev_Mod-A/"
        f"{repo_ref}/data/fieldcare"
    )
    for filename in FIELDCARE_ASSET_FILES:
        target = destination / filename
        if target.is_file():
            continue
        local_file = local_source / filename
        if local_file.is_file():
            shutil.copy2(local_file, target)
            continue
        with urlopen(f"{base_url}/{filename}", timeout=30) as response:
            target.write_bytes(response.read())
    return find_fieldcare_asset_dir(destination)


def load_fieldcare_assets(asset_dir: str | Path | None = None) -> dict[str, Any]:
    root = find_fieldcare_asset_dir(asset_dir)
    return {
        "asset_dir": root,
        "equipment": _read_csv(root / "equipment_records.csv"),
        "maintenance": _read_csv(root / "maintenance_history.csv"),
        "tickets": _read_csv(root / "service_tickets.csv"),
        "service_docs": _read_jsonl(root / "service_docs.jsonl"),
        "tool_schemas": _read_json(root / "tool_schemas.json"),
        "requests": _read_jsonl(root / "user_requests.jsonl"),
        "eval_cases": _read_jsonl(root / "eval_cases.jsonl"),
    }


def build_fieldcare_direct_registry(
    asset_dir: str | Path | None = None,
) -> ToolRegistry:
    assets = load_fieldcare_assets(asset_dir)
    schemas = {item["name"]: item for item in assets["tool_schemas"]["tools"]}
    equipment = {row["equipment_id"]: row for row in assets["equipment"]}
    tickets = {row["ticket_id"]: row for row in assets["tickets"]}

    def equipment_record(equipment_id: str) -> dict[str, Any]:
        row = equipment.get(equipment_id)
        return _tool_response(
            status="found" if row else "not_found",
            data=row,
            message=(
                f"Equipment record found for {equipment_id}."
                if row
                else f"No equipment record found for {equipment_id}."
            ),
            source_id="equipment_records.csv",
        )

    def warranty_status(
        equipment_id: str, service_date: str | None = None
    ) -> dict[str, Any]:
        """Read the course equipment snapshot, not a simulated remote service."""
        row = equipment.get(equipment_id)
        if not row:
            return _tool_response(
                "not_found",
                None,
                f"No warranty record found for {equipment_id}.",
                "equipment_records.csv",
            )
        coverage = row["warranty_status"]
        if coverage == "unknown":
            return {
                **_tool_response(
                    "unavailable",
                    None,
                    f"The local course record for {equipment_id} has unknown warranty state; contact Warranty Operations.",
                    "equipment_records.csv",
                ),
                "error_type": "incomplete_record",
            }
        return _tool_response(
            "found",
            {
                "coverage_state": coverage,
                "service_date": service_date
                or datetime.now(timezone.utc).date().isoformat(),
                "service_eligibility": row["service_eligibility"],
                "confidence": "high",
                "recommended_next_step": "combine_with_current_policy_conditions",
            },
            f"Warranty state for {equipment_id}: {coverage}.",
            "equipment_records.csv",
        )

    def maintenance_history(
        equipment_id: str,
        limit: int = 5,
        issue_category: str | None = None,
    ) -> dict[str, Any]:
        rows = [
            row for row in assets["maintenance"] if row["equipment_id"] == equipment_id
        ]
        if issue_category:
            needle = issue_category.replace("_", " ").lower()
            rows = [
                row
                for row in rows
                if needle in " ".join(row.values()).replace("_", " ").lower()
            ]
        rows = sorted(rows, key=lambda row: row["service_date"], reverse=True)[:limit]
        return _tool_response(
            "found" if rows else "not_found",
            rows,
            f"Found {len(rows)} maintenance event(s) for {equipment_id}.",
            "maintenance_history.csv",
        )

    def ticket_status(ticket_id: str) -> dict[str, Any]:
        row = tickets.get(ticket_id)
        return _tool_response(
            "found" if row else "not_found",
            row,
            (
                f"Ticket record found for {ticket_id}."
                if row
                else f"No ticket record found for {ticket_id}."
            ),
            "service_tickets.csv",
        )

    def escalation_path(
        equipment_id: str, ticket_id: str, reason_code: str
    ) -> dict[str, Any]:
        teams = {
            "safety_indicator": "Safety Engineering",
            "repeat_fault": "Field Engineering",
            "unsupported_model": "Legacy Service Desk",
            "warranty_unavailable": "Warranty Operations",
            "permission_failure": "Warranty Operations",
        }
        return _tool_response(
            "found",
            {
                "recommended_team": teams[reason_code],
                "equipment_id": equipment_id,
                "ticket_id": ticket_id,
                "reason_code": reason_code,
                "action_performed": False,
            },
            "Escalation path recommended; no ticket action was performed.",
            "routing_rules",
        )

    handlers = {
        "get_equipment_record": equipment_record,
        "get_warranty_status": warranty_status,
        "get_maintenance_history": maintenance_history,
        "get_ticket_status": ticket_status,
        "recommend_escalation_path": escalation_path,
    }
    registry = ToolRegistry()
    for name in FIELDCARE_DIRECT_TOOLS:
        schema = schemas[name]
        registry.register(
            name=name,
            description=schema["description"],
            parameters=schema["parameters"],
            handler=handlers[name],
        )
    return registry


def search_fieldcare_service_docs(
    query: str,
    *,
    top_k: int = 5,
    asset_dir: str | Path | None = None,
    retrieval_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = retrieval_config or {
        "chunking": {
            "strategy": "document_sections",
            "max_chars": 1200,
            "overlap": 120,
        },
        "baseline_config": {"method": "bm25", "candidate_k": top_k},
        "selected_config": {"retrieval": "bm25", "rerank": True, "accepted_k": top_k},
    }
    validate_retrieval_settings(config)
    docs = load_fieldcare_assets(asset_dir)["service_docs"]
    helper_docs = fieldcare_chunks(docs, config["chunking"])
    searchable_docs = [
        Document(
            id=doc.id,
            text=f"{doc.metadata['title']}\n{doc.text}",
            metadata=doc.metadata,
        )
        for doc in helper_docs
    ]
    keyword = BM25Retriever.from_documents(searchable_docs)
    mode = config["selected_config"]["retrieval"]
    if mode in {"semantic", "hybrid"}:
        from hybrid import HybridRetriever
        from openrouter import OpenRouterClient, OpenRouterEmbedder
        from vector_store import ChromaStore

        if not os.getenv("OPENROUTER_API_KEY"):
            raise FieldCareArtifactError(
                "Semantic and hybrid retrieval require OPENROUTER_API_KEY."
            )
        # The corpus hash isolates indexes when the corpus or chunk settings change.
        corpus_digest = hashlib.sha256(
            json.dumps(
                [doc.model_dump() for doc in searchable_docs], sort_keys=True
            ).encode()
        ).hexdigest()[:24]
        directory = Path(os.getenv("FIELDCARE_INDEX_DIR", ".chroma/fieldcare"))
        with OpenRouterClient() as client:
            store = ChromaStore(
                path=directory,
                collection_name=f"fieldcare-{corpus_digest}",
                embedder=OpenRouterEmbedder(client),
            )
            if {doc.id for doc in store.all_documents()} != {
                doc.id for doc in searchable_docs
            }:
                store.index(searchable_docs)
            if mode == "semantic":
                results = store.semantic_search(query, top_k=top_k)
            else:
                blend = config["hybrid_config"]
                results = HybridRetriever(
                    vector_store=store, keyword_retriever=keyword
                ).search(
                    query,
                    top_k=top_k,
                    alpha=blend["semantic_weight"],
                    semantic_k=blend["candidate_k"],
                    keyword_k=config["baseline_config"]["candidate_k"],
                )
    else:
        results = keyword.search(query, top_k=top_k)
    score_field = {
        "hybrid": "hybrid_score",
        "semantic": "semantic_score",
        "bm25": "keyword_score",
    }[mode]
    scores = [float(getattr(result, score_field) or 0) for result in results]
    maximum = max(scores, default=1) or 1
    originals = {doc.id: doc for doc in helper_docs}
    return [
        {
            **originals[result.document.id].metadata,
            "chunk_id": result.document.id,
            "text": originals[result.document.id].text,
            "score": score,
            "baseline_score": score / maximum,
            "preview": originals[result.document.id].text[:280],
            "retrieval_backend": config["selected_config"]["retrieval"],
        }
        for result, score in zip(results, scores)
    ]


def fieldcare_chunks(docs: list[dict], config: dict) -> list[Document]:
    chunks = []
    for row in docs:
        chunks.extend(
            chunk_text(
                row["text"],
                source_id=row["doc_id"],
                chunk_size=config["max_chars"],
                overlap=config["overlap"],
                metadata={key: value for key, value in row.items() if key != "text"},
            )
        )
    return chunks


def validate_retrieval_settings(config: dict) -> None:
    chunking = config.get("chunking", {})
    if chunking.get("strategy") != "document_sections":
        raise FieldCareArtifactError("chunking.strategy must be document_sections.")
    size, overlap = chunking.get("max_chars"), chunking.get("overlap")
    if type(size) is not int or type(overlap) is not int or not 0 <= overlap < size:
        raise FieldCareArtifactError("chunking requires 0 <= overlap < max_chars.")
    selected = config.get("selected_config", {})
    if (
        selected.get("retrieval") not in {"bm25", "semantic", "hybrid"}
        or type(selected.get("rerank")) is not bool
    ):
        raise FieldCareArtifactError(
            "Select bm25, semantic, or hybrid and a boolean rerank setting."
        )
    for value in [
        selected.get("accepted_k"),
        config.get("baseline_config", {}).get("candidate_k"),
    ]:
        if type(value) is not int or value < 1:
            raise FieldCareArtifactError(
                "accepted_k and candidate_k must be positive integers."
            )
    if config.get("baseline_config", {}).get("method") != "bm25":
        raise FieldCareArtifactError("baseline_config.method must be bm25.")
    if selected["retrieval"] == "hybrid":
        blend = config.get("hybrid_config", {})
        weights = [blend.get("semantic_weight"), blend.get("keyword_weight")]
        if (
            any(type(w) not in (float, int) or not 0 <= w <= 1 for w in weights)
            or abs(sum(weights) - 1) > 1e-8
        ):
            raise FieldCareArtifactError(
                "Hybrid weights must be between zero and one and sum to one."
            )
        if type(blend.get("candidate_k")) is not int or blend["candidate_k"] < 1:
            raise FieldCareArtifactError("hybrid_config.candidate_k must be positive.")


async def inspect_and_call_fieldcare_mcp(
    *,
    query: str,
    top_k: int = 5,
    asset_dir: str | Path | None = None,
    retrieval_config: dict[str, Any] | None = None,
) -> MCPInspectAndCallResult:
    root = find_fieldcare_asset_dir(asset_dir)
    env = dict(os.environ)
    env["FIELDCARE_ASSET_DIR"] = str(root)
    return await inspect_and_call_stdio_tool(
        command=os.sys.executable,
        args=["-m", "fieldcare_mcp_server"],
        tool_name="search_service_docs",
        arguments={
            "query": query,
            "top_k": top_k,
            "retrieval_config": retrieval_config,
        },
        env=env,
    )


def call_fieldcare_mcp(**kwargs: Any) -> list[dict[str, Any]]:
    # A worker owns the event loop so synchronous lesson helpers also work in ipykernel.
    with ThreadPoolExecutor(max_workers=1) as worker:
        result = worker.submit(
            lambda: asyncio.run(inspect_and_call_fieldcare_mcp(**kwargs))
        ).result()
    if result.call.is_error:
        raise RuntimeError(
            "FieldCare MCP search failed: " + " ".join(result.call.content_texts)
        )
    rows = []
    for content in result.call.content_texts:
        parsed = json.loads(content)
        rows.extend(parsed if isinstance(parsed, list) else [parsed])
    return rows


def validate_fieldcare_artifact(payload: dict[str, Any], artifact_type: str) -> None:
    if payload.get("schema_version") != FIELDCARE_SCHEMA_VERSION:
        raise FieldCareArtifactError(
            f"{artifact_type}: schema_version must be {FIELDCARE_SCHEMA_VERSION}."
        )
    if payload.get("project_id") != FIELDCARE_PROJECT_ID:
        raise FieldCareArtifactError(
            f"{artifact_type}: project_id must be {FIELDCARE_PROJECT_ID}."
        )
    validators = {
        "app_contract": _validate_app_contract,
        "retrieval_config": _validate_retrieval_config,
        "capability_contracts": _validate_capability_contracts,
        "submission_manifest": _validate_submission_manifest,
    }
    try:
        validator = validators[artifact_type]
    except KeyError as exc:
        raise FieldCareArtifactError(
            f"Unknown artifact type: {artifact_type}."
        ) from exc
    validator(payload)
    _reject_placeholders(payload, artifact_type)


def write_fieldcare_artifact(
    payload: dict[str, Any],
    artifact_type: str,
    destination: str | Path,
) -> Path:
    validate_fieldcare_artifact(payload, artifact_type)
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def load_fieldcare_artifact(path: str | Path, artifact_type: str) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_fieldcare_artifact(payload, artifact_type)
    return payload


def artifact_record(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _validate_app_contract(payload: dict[str, Any]) -> None:
    _require_nonempty(payload, "supported_request_types", list, "app_contract")
    _require_nonempty(payload, "required_inputs", list, "app_contract")
    _require_nonempty(payload, "response_contract", dict, "app_contract")
    _require_nonempty(
        payload["response_contract"], "required_fields", list, "response_contract"
    )
    _require_nonempty(payload, "routing_rules", list, "app_contract")
    _require_nonempty(payload, "success_criteria", list, "app_contract")


def _validate_retrieval_config(payload: dict[str, Any]) -> None:
    _require_nonempty(payload, "source_revision", str, "retrieval_config")
    _require_nonempty(payload, "chunking", dict, "retrieval_config")
    _require_nonempty(payload, "baseline_config", dict, "retrieval_config")
    _require_nonempty(payload, "hybrid_config", dict, "retrieval_config")
    _require_nonempty(payload, "rerank_config", dict, "retrieval_config")
    _require_nonempty(payload, "selected_config", dict, "retrieval_config")
    _require_nonempty(payload, "evaluation_queries", list, "retrieval_config")
    _require_nonempty(payload, "evidence_summary", list, "retrieval_config")


def _validate_capability_contracts(payload: dict[str, Any]) -> None:
    direct_tools = payload.get("direct_tools")
    if not isinstance(direct_tools, list):
        raise FieldCareArtifactError(
            "capability_contracts: direct_tools must be a list."
        )
    names = {tool.get("name") for tool in direct_tools if isinstance(tool, dict)}
    if names != set(FIELDCARE_DIRECT_TOOLS):
        raise FieldCareArtifactError(
            "capability_contracts: direct_tools must contain the five FieldCare state/action contracts."
        )
    mcp_server = payload.get("mcp_server")
    if not isinstance(mcp_server, dict):
        raise FieldCareArtifactError(
            "capability_contracts: mcp_server must be an object."
        )
    mcp_names = {
        tool.get("name")
        for tool in mcp_server.get("tools", [])
        if isinstance(tool, dict)
    }
    if "search_service_docs" not in mcp_names:
        raise FieldCareArtifactError(
            "capability_contracts: MCP must expose search_service_docs."
        )
    if names & mcp_names:
        raise FieldCareArtifactError(
            "capability_contracts: direct and MCP tools must not overlap."
        )
    if mcp_names != {"search_service_docs"}:
        raise FieldCareArtifactError(
            "capability_contracts: MCP must expose only search_service_docs."
        )
    rules = payload.get("orchestration_rules")
    if not isinstance(rules, list) or not rules:
        raise FieldCareArtifactError(
            "capability_contracts: orchestration_rules must be non-empty."
        )
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise FieldCareArtifactError(
                f"capability_contracts: rule {index} must be an object."
            )
        _require_nonempty(rule, "required_inputs", list, f"orchestration rule {index}")
        if not isinstance(rule.get("available_inputs_before_call"), list):
            raise FieldCareArtifactError(
                f"orchestration rule {index}: available_inputs_before_call must be a list."
            )
    _require_nonempty(payload, "failure_policy", dict, "capability_contracts")
    _require_nonempty(payload, "trace_evidence", list, "capability_contracts")


def _validate_submission_manifest(payload: dict[str, Any]) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or {
        item.get("filename") for item in artifacts
    } != set(FIELDCARE_ARTIFACT_FILENAMES):
        raise FieldCareArtifactError(
            "submission_manifest: artifacts must list the three sprint JSON files."
        )
    for item in artifacts:
        if len(str(item.get("sha256", ""))) != 64:
            raise FieldCareArtifactError(
                "submission_manifest: every artifact needs a SHA-256 digest."
            )
    _require_nonempty(payload, "final_trace", dict, "submission_manifest")
    _require_nonempty(payload, "evaluation_summary", dict, "submission_manifest")
    improvement = payload.get("selected_improvement")
    if not isinstance(improvement, dict):
        raise FieldCareArtifactError(
            "submission_manifest: selected_improvement must be an object."
        )
    _require_nonempty(improvement, "before_evidence", str, "selected_improvement")
    _require_nonempty(improvement, "after_evidence", str, "selected_improvement")
    _require_nonempty(payload, "limitations", list, "submission_manifest")


def _require_nonempty(
    payload: dict[str, Any], key: str, expected: type, label: str
) -> None:
    value = payload.get(key)
    if not isinstance(value, expected) or not value:
        raise FieldCareArtifactError(
            f"{label}: {key} must be a non-empty {expected.__name__}."
        )


def _reject_placeholders(payload: Any, label: str) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _reject_placeholders(value, label)
    elif isinstance(payload, list):
        for value in payload:
            _reject_placeholders(value, label)
    elif isinstance(payload, str):
        lowered = payload.strip().lower()
        if lowered.startswith(("learner task:", "todo", "replace me")):
            raise FieldCareArtifactError(
                f"{label}: unresolved learner placeholder: {payload!r}."
            )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _tool_response(
    status: str,
    data: Any,
    message: str,
    source_id: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "data": data,
        "message": message,
        "error_type": None if status == "found" else "missing_record",
        "source_id": source_id,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
