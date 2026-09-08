import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/richhiey/ai-app-dev_Mod-A.git"
NOTEBOOKS = [
    "notebooks/sprint_1_llm_structured_outputs.ipynb",
    "notebooks/sprint_2_rag_hybrid_hyde.ipynb",
    "notebooks/sprint_3_tools_mcp.ipynb",
]


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
        code = "\n".join(_code_sources(notebook))

        assert notebook["nbformat"] == 4
        assert REPO_URL in code
        assert '"git", "clone", REPO_URL, str(REPO_DIR)' in code
        assert '"git", "-C", str(REPO_DIR), "pull", "--ff-only"' in code
        assert "\"pip\", \"install\", \"-q\", \"-e\"" in code
        assert "ms_ai_ml_core" not in code
        assert "/Users/richhiey" not in code


def test_notebook_code_cells_parse_as_python() -> None:
    for relative_path in NOTEBOOKS:
        notebook = _load_notebook(relative_path)
        for index, source in enumerate(_code_sources(notebook)):
            ast.parse(source, filename=f"{relative_path}:cell-{index}")
