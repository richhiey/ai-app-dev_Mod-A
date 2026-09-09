import ast
import json
import re
from pathlib import Path

from models import all_allowed_model_ids


ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/richhiey/ai-app-dev_Mod-A.git"
PIP_REQUIREMENT = (
    "ms-ai-ml-helper-core @ "
    "git+https://github.com/richhiey/ai-app-dev_Mod-A.git@main"
)
NOTEBOOKS = [
    "notebooks/sprint_1_llm_structured_outputs.ipynb",
    "notebooks/sprint_2_rag_hybrid_hyde.ipynb",
    "notebooks/sprint_3_tools_mcp.ipynb",
]
REQUIRED_TOPICS = {
    "notebooks/sprint_1_llm_structured_outputs.ipynb": [
        "OpenRouterClient",
        "StructuredOutputGraph",
        "APPLICATION_COMPONENTS",
        "MODEL_ACCESS_PATHS",
        "client.chat",
        "client.structured",
    ],
    "notebooks/sprint_2_rag_hybrid_hyde.ipynb": [
        "ChromaStore",
        "BM25Retriever",
        "HybridRetriever",
        "OpenRouterReranker",
        "HyDERewriter",
        "uuid.uuid4().hex",
    ],
    "notebooks/sprint_3_tools_mcp.ipynb": [
        "sprint3_tools_mcp",
        "build_heliodesk_tool_registry",
        "run_failure_scenarios",
        "run_scripted_tool_loop",
        "connect_to_heliodesk_policy_mcp",
        "validate_mcp_policy_response",
    ],
}
LIVE_SESSION_ARCS = {
    "notebooks/sprint_1_llm_structured_outputs.ipynb": [
        "## 3. Map the application boundary",
        "## 4. Compare the three access paths",
        "## 5. Make a direct LLM call",
        "## 6. Move from text to validated JSON",
        "## 7. Wrap the same idea in LangGraph",
        "## 8. Sprint 1 checkpoint defense",
    ],
    "notebooks/sprint_2_rag_hybrid_hyde.ipynb": [
        "## 5. Baseline retrieval",
        "## 6. Rerank the candidate pool",
        "## 7. Add hybrid search",
        "## 8. Tune the blend",
        "## 9. Rewrite the query with HyDE",
        "## 10. Build the checkpoint defense",
        "## 11. Guided checkpoint: three retrieval tasks",
    ],
    "notebooks/sprint_3_tools_mcp.ipynb": [
        "## 4. Inspect the schemas sent to the model",
        "## 5. Execute a valid direct tool call",
        "## 6. Handle malformed and failed tool calls",
        "## 7. Let the model run a multi-step tool loop",
        "## 8. Connect to the MCP server",
        "## 9. Validate the MCP response",
        "## 10. Sprint 3 checkpoint answers and evidence",
    ],
}
MODEL_SLUG_RE = re.compile(r"\b(?:google|cohere)/[a-z0-9._:-]+\b")


def _load_notebook(relative_path: str) -> dict:
    path = ROOT / relative_path
    return json.loads(path.read_text(encoding="utf-8"))


def _code_sources(notebook: dict) -> list[str]:
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def test_colab_notebooks_are_valid_and_install_from_github() -> None:
    for relative_path in NOTEBOOKS:
        notebook = _load_notebook(relative_path)
        code_sources = _code_sources(notebook)
        setup_code = code_sources[0]
        code = "\n".join(code_sources)

        assert notebook["nbformat"] == 4
        assert REPO_URL in setup_code
        assert PIP_REQUIREMENT in setup_code
        assert "%pip install -q --force-reinstall --no-cache-dir" in setup_code
        assert "subprocess" not in setup_code
        assert "sys.path" not in setup_code
        assert "REPO_DIR" not in setup_code
        credential_source = code
        if relative_path == "notebooks/sprint_3_tools_mcp.ipynb":
            credential_source += (ROOT / "src" / "sprint3_tools_mcp.py").read_text(
                encoding="utf-8"
            )
        assert "userdata.get(\"OPENROUTER_API_KEY\")" in credential_source
        assert "ms_ai_ml_core" not in code
        assert "/Users/richhiey" not in code


def test_notebook_code_cells_parse_as_python() -> None:
    for relative_path in NOTEBOOKS:
        notebook = _load_notebook(relative_path)
        for index, source in enumerate(_code_sources(notebook)):
            if index == 0:
                assert source.lstrip().startswith("#@title Install helper core")
                continue
            ast.parse(source, filename=f"{relative_path}:cell-{index}")


def test_notebooks_are_clean_for_publication() -> None:
    for relative_path in NOTEBOOKS:
        notebook = _load_notebook(relative_path)
        assert notebook["metadata"]["kernelspec"]["name"] == "python3"
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_sprint_3_notebook_keeps_helper_plumbing_hidden() -> None:
    notebook = _load_notebook("notebooks/sprint_3_tools_mcp.ipynb")
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    visible_code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and cell.get("metadata", {}).get("cellView") != "form"
    ]

    assert any("from sprint3_tools_mcp import" in source for source in code_cells)
    assert not any("def check_export_authorization" in source for source in visible_code_cells)
    assert not any("class ScriptedHelioDeskClient" in source for source in visible_code_cells)
    assert all(len(source.splitlines()) <= 30 for source in visible_code_cells)


def test_notebooks_cover_required_sprint_topics() -> None:
    for relative_path, topics in REQUIRED_TOPICS.items():
        notebook = _load_notebook(relative_path)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        for topic in topics:
            assert topic in source


def test_notebooks_follow_live_session_arcs() -> None:
    for relative_path, headings in LIVE_SESSION_ARCS.items():
        notebook = _load_notebook(relative_path)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        positions = [source.index(heading) for heading in headings]
        assert positions == sorted(positions)


def test_notebooks_only_reference_enabled_openrouter_models() -> None:
    allowed = set(all_allowed_model_ids())
    for relative_path in NOTEBOOKS:
        notebook = _load_notebook(relative_path)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        for model_slug in MODEL_SLUG_RE.findall(source):
            assert model_slug in allowed
