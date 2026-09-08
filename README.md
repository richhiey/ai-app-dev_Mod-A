# MS AI/ML Helper Core

Shared Python helpers for the AI & ML App Development course notebooks. The repo is intentionally small: each helper is easy to read in Colab, but production-shaped enough to teach the real patterns.

## What This Covers

- OpenRouter chat, structured output, embedding, and rerank calls
- ChromaDB document indexing and semantic search
- BM25 keyword retrieval and hybrid search blending
- Cohere reranking through OpenRouter
- HyDE query rewriting
- LangGraph structured-output workflows
- Multi-step tool calling with structured tool errors
- Small MCP server and stdio client helpers for tool exposure practice

## Course Model Policy

Only these OpenRouter models are allowed:

| Purpose | Default model |
| --- | --- |
| Cheap chat and HyDE | `google/gemini-2.5-flash-lite` |
| Agent and structured JSON | `google/gemini-3.1-flash-lite` |
| More reasoning | `google/gemini-2.5-flash` |
| Embeddings | `google/gemini-embedding-001` |
| Reranking | `cohere/rerank-v3.5` |

The package rejects any other model before making a network request.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` or in your notebook environment.

## Quick Checks

```bash
pytest
python scripts/call_llm.py "Say hello in one sentence."
```

## Index Documents

Put `.txt`, `.md`, or `.py` files in a folder, then run:

```bash
python scripts/index_directory.py ./docs --collection course_docs --db-path ./data/chroma
```

## Search Documents

```bash
python scripts/search.py "where does naive RAG break?" --collection course_docs --db-path ./data/chroma
python scripts/search.py "where does naive RAG break?" --rerank --hyde
```

## Notebook Pattern

The Colab notebooks in `notebooks/` begin by cloning
`https://github.com/richhiey/ai-app-dev_Mod-A.git` and installing the repo in
editable mode. This keeps live and campus notebooks on the shared source of
truth instead of copying helper functions into each notebook.

In Colab, add `OPENROUTER_API_KEY` in Secrets. The notebooks will use that value
automatically and fall back to a hidden prompt when the secret is not present.

Available notebooks:

- `notebooks/sprint_1_llm_structured_outputs.ipynb`
- `notebooks/sprint_2_rag_hybrid_hyde.ipynb`
- `notebooks/sprint_3_tools_mcp.ipynb`

```python
from hybrid import HybridRetriever
from keyword_search import BM25Retriever
from openrouter import OpenRouterClient, OpenRouterEmbedder
from rerank import OpenRouterReranker
from vector_store import ChromaStore

client = OpenRouterClient()
embedder = OpenRouterEmbedder(client)
store = ChromaStore(path="./data/chroma", collection_name="course_docs", embedder=embedder)

docs = store.all_documents()
keyword = BM25Retriever.from_documents(docs)
hybrid = HybridRetriever(vector_store=store, keyword_retriever=keyword, alpha=0.65)

candidates = hybrid.search("how can reranking improve RAG?", top_k=10)
top = OpenRouterReranker(client).rerank("how can reranking improve RAG?", candidates, top_n=3)
```

## MCP Server

```bash
mcp dev src/mcp_server.py
```

The server exposes `health` and `keyword_search`. These are deliberately small so students can inspect the generated schemas before adding course-specific tools.

For notebook-based MCP practice, use `inspect_and_call_stdio_tool` from
`mcp_client.py` to connect to the local server, inspect advertised tools, call
one tool, and validate the returned content.

## Tests

The unit tests use fake model clients and fake embeddings where possible, so they do not spend OpenRouter credits. ChromaDB, LangGraph, and MCP are imported through the installed dependencies.
