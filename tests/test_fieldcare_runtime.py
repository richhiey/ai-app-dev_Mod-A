import ast
import copy
import json
import sys
import types
from pathlib import Path

import pytest

import fieldcare_runtime
from fieldcare import FieldCareArtifactError, search_fieldcare_service_docs

ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / "notebooks" / name).read_text())


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv("FIELDCARE_ASSET_DIR", str(ROOT / "data" / "fieldcare"))
    monkeypatch.chdir(ROOT)
    module = types.ModuleType("fieldcare_test_notebook")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec((ROOT / "src" / "fieldcare_project.py").read_text(), module.__dict__)
    ns = module.__dict__
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-key")

    # Unit tests isolate contract logic. The separate notebook runner uses real APIs.
    class ContractClient:
        def __init__(self, **kwargs):
            self.last_chat_response = types.SimpleNamespace(
                model="unit-test",
                raw={"id": "unit-test"},
                usage={"total_tokens": 1},
                content=None,
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def structured(self, messages, *, output_model, **kwargs):
            context = json.loads(messages[1]["content"])
            return output_model.model_validate(
                {
                    "request_id": context["request"]["request_id"],
                    "answer_summary": "Unit-test response",
                    "next_checks": [],
                    "coverage_statement": "not_applicable",
                    "evidence": [
                        {
                            "doc_id": row["doc_id"],
                            "chunk_id": row["chunk_id"],
                            "reason": "test",
                        }
                        for row in context["retrieved_evidence"]
                    ],
                    "tool_state": context["required_tool_state"],
                    "decision_flags": context["required_decision_flags"],
                    "escalation_path": context["required_escalation_path"],
                    "limitations": ["unit test"],
                }
            )

    ns["OpenRouterClient"] = ContractClient
    ns["rerank_service_docs"] = lambda pipeline, query, rows, **kwargs: [
        {**row, "rerank_score": row["baseline_score"]} for row in rows
    ]
    # Load the literal contract exported by Sprint 1, rather than a second copy.
    for cell in notebook("sprint_1_llm_structured_outputs.ipynb")["cells"]:
        text = "".join(cell["source"])
        if "FIELDCARE_APP_CONTRACT =" in text:
            node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign))
            app = ast.literal_eval(node.value)
    retrieval = {
        "schema_version": "1.0",
        "project_id": "fieldcare",
        "source_revision": "fieldcare-assets-v1.0.0",
        "chunking": {
            "strategy": "document_sections",
            "max_chars": 1200,
            "overlap": 120,
        },
        "baseline_config": {"method": "bm25", "candidate_k": 32},
        "hybrid_config": {
            "semantic_weight": 0.65,
            "keyword_weight": 0.35,
            "candidate_k": 32,
        },
        "rerank_config": {"model": "cohere/rerank-v3.5", "top_n": 8},
        "selected_config": {"retrieval": "bm25", "rerank": True, "accepted_k": 5},
        "evaluation_queries": [{"query": "HX overheating"}],
        "evidence_summary": [{"doc_id": "DOC-FC-TS-001"}],
    }
    names = [
        "get_equipment_record",
        "get_warranty_status",
        "get_maintenance_history",
        "get_ticket_status",
        "recommend_escalation_path",
    ]
    capabilities = {
        "schema_version": "1.0",
        "project_id": "fieldcare",
        "direct_tools": [{"name": name} for name in names],
        "mcp_server": {
            "name": "fieldcare-service-docs",
            "tools": [{"name": "search_service_docs"}],
        },
        "orchestration_rules": [
            {
                "request_type": request_type,
                "required_inputs": app["required_inputs"],
                "available_inputs_before_call": app["required_inputs"],
                "call_order": ["search_service_docs", *names],
            }
            for request_type in app["supported_request_types"]
        ],
        "failure_policy": {
            "missing_input": "ask",
            "timeout": "retry_once_then_route",
            "unavailable": "abstain_and_route",
        },
        "trace_evidence": [{"boundary": "mcp"}],
    }
    design = ns["default_pipeline_design"]()
    design.update(
        {
            "application_scope": app,
            "response_contract": app["response_contract"],
            "success_criteria": app["success_criteria"],
            "app_contract": app,
            "retrieval_contract": retrieval,
            "capability_contract": capabilities,
        }
    )
    pipeline = ns["build_fieldcare_pipeline"](
        ns["load_fieldcare_environment"](), design
    )
    monkeypatch.setattr(
        fieldcare_runtime,
        "call_fieldcare_mcp",
        lambda **kw: search_fieldcare_service_docs(**kw),
    )
    return ns, pipeline


def test_request_uses_only_mcp_evidence_and_preserves_call_order(runtime, monkeypatch):
    ns, pipeline = runtime
    sentinel = search_fieldcare_service_docs(
        "HX overheating", top_k=1, asset_dir=pipeline.asset_dir
    )
    monkeypatch.setattr(fieldcare_runtime, "call_fieldcare_mcp", lambda **kw: sentinel)
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    assert {row["chunk_id"] for row in result["context"]["retrieved_evidence"]} <= {
        sentinel[0]["chunk_id"]
    }
    assert [
        call["tool"] for call in result["context"]["execution_trace"]
    ] == pipeline.contract_rules["troubleshooting_plus_warranty"]["call_order"]
    assert "search_service_docs" not in pipeline.toolbox


def test_mcp_failure_cannot_fall_back_to_local_evidence(runtime, monkeypatch):
    ns, _ = runtime

    def unavailable(**kwargs):
        raise RuntimeError("disabled")

    monkeypatch.setattr(fieldcare_runtime, "call_fieldcare_mcp", unavailable)
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    assert result["context"]["retrieved_evidence"] == []
    assert {"abstain", "mention_uncertainty", "escalate"} <= set(
        result["response"]["decision_flags"]
    )
    assert result["context"]["execution_trace"][0]["status"] == "unavailable"


def test_accepted_k_and_rerank_switch_change_execution(runtime, monkeypatch):
    ns, pipeline = runtime
    pipeline.retrieval_contract["selected_config"].update(accepted_k=1, rerank=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reranker must not run when disabled")

    ns["rerank_service_docs"] = forbidden
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    assert len(result["context"]["retrieved_evidence"]) == 1


def test_chunking_and_candidate_limit_are_executed(runtime):
    _, pipeline = runtime
    config = pipeline.retrieval_contract
    config["chunking"].update(max_chars=60, overlap=10)
    rows = search_fieldcare_service_docs(
        "HX overheating", top_k=2, asset_dir=pipeline.asset_dir, retrieval_config=config
    )
    assert len(rows) == 2
    assert all(len(row["text"]) <= 60 for row in rows)


def test_required_inputs_stop_calls_and_output_fields_are_enforced(runtime):
    ns, pipeline = runtime
    pipeline.app_contract["required_inputs"].append("technician_id")
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    assert result["context"]["execution_trace"] == []
    assert result["context"]["missing_inputs"] == ["technician_id"]
    assert "ask_for_more_information" in result["response"]["decision_flags"]
    pipeline.app_contract["required_inputs"].remove("technician_id")
    pipeline.app_contract["response_contract"]["required_fields"].append(
        "nonexistent_output"
    )
    with pytest.raises(FieldCareArtifactError, match="nonexistent_output"):
        ns["run_fieldcare_app"]("REQ-FC-001")


def test_routing_rule_is_used(runtime):
    ns, pipeline = runtime
    pipeline.app_contract["routing_rules"] = [
        {"when": "repeat_fault", "route_to": "Specialist Review"}
    ]
    assert (
        ns["run_fieldcare_app"]("REQ-FC-001")["response"]["escalation_path"]
        == "Specialist Review"
    )


@pytest.mark.parametrize(
    "policy, expected", [("retry_once_then_route", 2), ("abstain_and_route", 1)]
)
def test_timeout_policy_controls_mcp_retry(runtime, monkeypatch, policy, expected):
    ns, pipeline = runtime
    pipeline.capability_contract["failure_policy"]["timeout"] = policy
    calls = []

    def timeout(**kwargs):
        calls.append(kwargs)
        raise TimeoutError()

    monkeypatch.setattr(fieldcare_runtime, "call_fieldcare_mcp", timeout)
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    assert len(calls) == expected
    assert "abstain" in result["response"]["decision_flags"]


def test_hybrid_requires_credentials_instead_of_silent_bm25(runtime, monkeypatch):
    _, pipeline = runtime
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config = copy.deepcopy(pipeline.retrieval_contract)
    config["selected_config"]["retrieval"] = "hybrid"
    with pytest.raises(FieldCareArtifactError, match="OPENROUTER_API_KEY"):
        search_fieldcare_service_docs(
            "HX overheating", asset_dir=pipeline.asset_dir, retrieval_config=config
        )


def test_hybrid_weights_change_ranking_using_existing_hybrid_retriever(
    runtime, monkeypatch
):
    import fieldcare
    import vector_store
    from hybrid import RetrievalResult

    _, pipeline = runtime
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder-not-a-real-key")
    docs = [
        {
            "doc_id": "LEXICAL",
            "title": "filter filter",
            "text": "filter repair",
            "current": True,
        },
        {
            "doc_id": "SEMANTIC",
            "title": "airflow",
            "text": "restricted intake cleaning",
            "current": True,
        },
    ]
    monkeypatch.setattr(
        fieldcare, "load_fieldcare_assets", lambda *args: {"service_docs": docs}
    )

    class SemanticFixture:
        def __init__(self, **kwargs):
            pass

        def index(self, documents):
            self.documents = documents

        def all_documents(self):
            return []

        def semantic_search(self, query, **kwargs):
            return [
                RetrievalResult(
                    document=doc,
                    semantic_score=1.0 if doc.metadata["doc_id"] == "SEMANTIC" else 0.1,
                )
                for doc in self.documents
            ]

    monkeypatch.setattr(vector_store, "ChromaStore", SemanticFixture)
    config = copy.deepcopy(pipeline.retrieval_contract)
    config["selected_config"]["retrieval"] = "hybrid"
    config["hybrid_config"].update(semantic_weight=0.0, keyword_weight=1.0)
    lexical = search_fieldcare_service_docs("filter", top_k=2, retrieval_config=config)
    config["hybrid_config"].update(semantic_weight=1.0, keyword_weight=0.0)
    semantic = search_fieldcare_service_docs("filter", top_k=2, retrieval_config=config)
    assert lexical[0]["doc_id"] == "LEXICAL"
    assert semantic[0]["doc_id"] == "SEMANTIC"


def test_direct_timeout_retries_and_then_abstains(runtime):
    ns, pipeline = runtime
    original = ns["execute_registered_tool"]
    calls = []

    def timeout(pipeline, name, args, index):
        calls.append(name)
        result = original(pipeline, name, args, index)
        if name == "get_warranty_status":
            result["result"].update(status="timeout", data=None)
        return result

    ns["execute_registered_tool"] = timeout
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    assert calls.count("get_warranty_status") == 2
    assert "abstain" in result["response"]["decision_flags"]


def test_out_of_scope_request_does_not_execute_capabilities(runtime):
    ns, pipeline = runtime
    request = ns["request_by_id"](pipeline.env, "REQ-FC-001")
    request["request_type"] = "unsupported_business_promise"
    result = ns["run_fieldcare_request"](pipeline, request)
    assert result["context"]["execution_trace"] == []
    assert not result["context"]["retrieved_evidence"]
    assert "scope_boundary" in result["response"]["decision_flags"]


def test_real_mcp_request_has_traceable_evidence(runtime, monkeypatch):
    import fieldcare

    ns, _ = runtime
    monkeypatch.setattr(
        fieldcare_runtime, "call_fieldcare_mcp", fieldcare.call_fieldcare_mcp
    )
    result = ns["run_fieldcare_app"]("REQ-FC-001")
    evidence = result["context"]["retrieved_evidence"]
    trace = result["context"]["execution_trace"]
    assert evidence and trace[0]["status"] == "ok"
    assert {row["chunk_id"] for row in evidence} <= set(trace[0]["chunk_ids"])


def test_published_notebooks_have_unique_ids_and_no_saved_outputs():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        cells = json.loads(path.read_text())["cells"]
        ids = [cell["id"] for cell in cells]
        assert all(ids) and len(ids) == len(set(ids)), path.name
        for cell in cells:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [], path.name
                assert cell["execution_count"] is None, path.name


@pytest.mark.parametrize("recovers", [True, False])
def test_model_citation_validation_retries_without_rewriting(runtime, recovers):
    ns, _ = runtime
    original = ns["OpenRouterClient"]
    calls = []

    class CitationClient(original):
        def structured(self, messages, **kwargs):
            calls.append(copy.deepcopy(messages))
            result = super().structured(messages, **kwargs)
            if len(calls) == 1 or not recovers:
                result.evidence[0].chunk_id = "not-retrieved"
            self.last_chat_response.content = result.model_dump_json()
            return result

    ns["OpenRouterClient"] = CitationClient
    if recovers:
        result = ns["run_fieldcare_app"]("REQ-FC-001")
        assert result["response"]["model_call"]["validation_retries"]
        assert all(
            row["chunk_id"] != "not-retrieved" for row in result["response"]["evidence"]
        )
    else:
        with pytest.raises(ValueError, match="Unretrieved citation"):
            ns["run_fieldcare_app"]("REQ-FC-001")
    assert len(calls) == 2
    assert "Validation failed" in calls[1][-1]["content"]


def test_sprint4_answers_only_defined_inside_reveal_function():
    nb = notebook("sprint_4_fieldcare_project.ipynb")
    source = next(
        "".join(c["source"])
        for c in nb["cells"]
        if "def reveal_project_answer_signals" in "".join(c["source"])
    )
    tree = ast.parse(source)
    assert not any(
        isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "PROJECT_ANSWER_SIGNALS"
            for t in n.targets
        )
        for n in tree.body
    )
    reveal = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "reveal_project_answer_signals"
    )
    assert isinstance(reveal.body[0], ast.If)
