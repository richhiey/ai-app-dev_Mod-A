import asyncio
from io import UnsupportedOperation
from pathlib import Path
import sys

from mcp_client import inspect_and_call_stdio_tool
from mcp_server import build_mcp_server, keyword_search_documents


def test_keyword_search_documents_is_usable_without_running_server() -> None:
    results = keyword_search_documents(
        query="tool schema",
        documents=[
            "RAG retrieval uses embeddings.",
            "Tool schemas describe function inputs and outputs.",
        ],
        top_k=1,
    )

    assert results[0]["index"] == 1
    assert results[0]["score"] > 0


def test_build_mcp_server_returns_server_instance() -> None:
    server = build_mcp_server()

    assert server is not None


def test_mcp_client_can_inspect_and_call_local_server(tmp_path) -> None:
    errlog_path = tmp_path / "mcp.stderr.log"

    result = asyncio.run(
        inspect_and_call_stdio_tool(
            command=sys.executable,
            args=[str(Path("src/mcp_server.py"))],
            tool_name="keyword_search",
            arguments={
                "query": "tool schema",
                "documents": [
                    "RAG retrieval uses embeddings.",
                    "Tool schemas describe function inputs and outputs.",
                ],
                "top_k": 1,
            },
            cwd=Path(__file__).resolve().parents[1],
            errlog_path=errlog_path,
        )
    )

    assert errlog_path.exists()
    assert result.server_name == "ms-ai-ml-helper-core"
    assert [tool.name for tool in result.tools] == ["health", "keyword_search"]
    assert result.call.tool_name == "keyword_search"
    assert result.call.is_error is False
    assert "Tool schemas describe" in "\n".join(result.call.content_texts)


def test_mcp_client_works_when_notebook_stderr_has_no_file_descriptor(
    monkeypatch,
) -> None:
    class NotebookStderr:
        def fileno(self):
            raise UnsupportedOperation("fileno")

        def write(self, value):
            return len(value)

        def flush(self):
            return None

    monkeypatch.setattr(sys, "stderr", NotebookStderr())
    result = asyncio.run(
        inspect_and_call_stdio_tool(
            command=sys.executable,
            args=[str(Path("src/mcp_server.py"))],
            tool_name="health",
            cwd=Path(__file__).resolve().parents[1],
        )
    )

    assert result.call.is_error is False
    assert '"status":"ok"' in "".join(result.call.content_texts).replace(" ", "")
