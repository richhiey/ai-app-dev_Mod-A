"""Run all four notebooks against real APIs. Requires a key and spends API credits."""

from __future__ import annotations

import json
import ast
import argparse
from contextlib import nullcontext
import shutil
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = [
    "fieldcare_app_contract.json",
    "fieldcare_retrieval_config.json",
    "fieldcare_capability_contracts.json",
]


def bootstrap_source() -> str:
    return (
        "import os, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
        f"os.environ['FIELDCARE_ASSET_DIR'] = {str(ROOT / 'data' / 'fieldcare')!r}\n"
        "import pandas as pd\n"
        "from IPython.display import display, JSON, Markdown\n"
    )


def execute_notebook(name: str, workdir: Path, *, experiment: bool = False):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    install = next(
        cell
        for cell in notebook.cells
        if "Install helper core from GitHub" in cell.source
    )
    install.source = bootstrap_source()
    if experiment:
        for cell in notebook.cells:
            if cell.cell_type != "code" or "IMPROVEMENT_ACCEPTED_K" not in cell.source:
                continue
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name)
                    and target.id == "IMPROVEMENT_ACCEPTED_K"
                    for target in node.targets
                ):
                    lines = cell.source.splitlines()
                    lines[node.lineno - 1 : node.end_lineno] = [
                        "fieldcare_pipeline.retrieval_contract['selected_config']['accepted_k'] = 1",
                        "IMPROVEMENT_ACCEPTED_K = 5",
                    ]
                    cell.source = "\n".join(lines)
                    break
        notebook.cells.append(
            nbformat.v4.new_code_cell("""
assert FIELDCARE_SUBMISSION_MANIFEST["final_trace"]["mcp_tool"] == "search_service_docs"
trace = FIELDCARE_SUBMISSION_MANIFEST["final_trace"]
mcp_calls = [call for call in trace["calls"] if call["boundary"] == "mcp"]
assert mcp_calls and set(trace["accepted_doc_ids"]) <= set(mcp_calls[0]["doc_ids"])
assert len(json.loads(PROJECT_IMPROVEMENT_NOTE["before_evidence"])) == 1
assert len(json.loads(PROJECT_IMPROVEMENT_NOTE["after_evidence"])) == 5
assert PROJECT_IMPROVEMENT_NOTE["before_trace"] and PROJECT_IMPROVEMENT_NOTE["after_trace"]
assert PROJECT_IMPROVEMENT_NOTE["config_before"] != PROJECT_IMPROVEMENT_NOTE["config_after"]
assert len(evaluation_df) == 16
assert guided_response["model_call"]["request_id"]
assert guided_response["model_call"]["usage"]["total_tokens"] > 0
# Incomplete written responses must not expose answer signals.
shown = []
saved_pretty = pretty
pretty = shown.append
reveal_project_answer_signals()
assert not shown
pretty = saved_pretty
""")
        )

    def report_cell(cell, cell_index):
        if cell.cell_type == "code":
            print(f"  {name}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
        on_cell_start=report_cell,
    )
    try:
        return client.execute()
    finally:
        nbformat.write(notebook, workdir / name)


def main() -> None:
    from fieldcare import artifact_record
    from notebook_setup import require_openrouter_key

    require_openrouter_key()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep executed notebooks and artifacts outside the repository.",
    )
    args = parser.parse_args()
    workspace = (
        nullcontext(str(args.output_dir))
        if args.output_dir
        else tempfile.TemporaryDirectory(prefix="module-a-notebook-smoke-")
    )
    with workspace as temp:
        workdir = Path(temp)
        workdir.mkdir(parents=True, exist_ok=True)
        names = [
            "sprint_1_llm_structured_outputs.ipynb",
            "sprint_2_rag_hybrid_hyde.ipynb",
            "sprint_3_tools_mcp.ipynb",
        ]
        records = []
        for index, (name, artifact) in enumerate(zip(names, ARTIFACTS), start=1):
            directory = workdir / f"sprint-{index}"
            directory.mkdir()
            shutil.copytree(ROOT / "data", directory / "data")
            print(f"Executing {name} with real APIs...", flush=True)
            execute_notebook(name, directory)
            path = directory / artifact
            records.append(artifact_record(path))
        final = workdir / "sprint-4"
        final.mkdir()
        shutil.copytree(ROOT / "data" / "fieldcare", final / "data" / "fieldcare")
        for index, artifact in enumerate(ARTIFACTS, start=1):
            shutil.copy2(workdir / f"sprint-{index}" / artifact, final / artifact)
        print(
            "Executing sprint_4_fieldcare_project.ipynb with real APIs...", flush=True
        )
        execute_notebook("sprint_4_fieldcare_project.ipynb", final, experiment=True)
        manifest = json.loads(
            (final / "fieldcare_submission_manifest.json").read_text()
        )
        assert manifest["artifacts"] == records, (
            "Sprint 4 must consume the exact earlier exports."
        )
        experiment = manifest["selected_improvement"]
        print(
            json.dumps(
                {
                    "artifact_handoff": "3/3 identical digests",
                    "mcp_tool": manifest["final_trace"]["mcp_tool"],
                    "experiment": {
                        "before": experiment["before_evidence"],
                        "after": experiment["after_evidence"],
                    },
                    "evaluation_summary": manifest["evaluation_summary"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    main()
