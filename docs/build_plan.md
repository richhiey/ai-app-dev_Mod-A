# Build Plan

## Goal

Create a shared Python helper core that campus and live Colab notebooks can import. The code should make the course concepts runnable without hiding the important application mechanics from students.

## Architecture

- `models.py`: one source of truth for allowed OpenRouter models and defaults.
- `openrouter.py`: direct HTTP client for chat, structured output, embeddings, and reranking.
- `documents.py`: document model and stable chunking.
- `vector_store.py`: ChromaDB persistence and semantic query helpers.
- `keyword_search.py`: BM25 keyword retrieval.
- `hybrid.py`: score blending for semantic plus keyword retrieval.
- `rerank.py`: candidate reranking adapter.
- `hyde.py`: query rewriting with a hypothetical document.
- `tools.py`: tool schemas, function execution, and structured errors.
- `agent.py`: multi-step model and tool loop.
- `structured_graph.py`: LangGraph workflow for structured output.
- `mcp_server.py`: minimal MCP server for inspection and extension.
- `mcp_client.py`: stdio MCP client helpers for connect, inspect, call, and validate labs.

## Notebook Sequence

1. Sprint 1 notebooks import `OpenRouterClient`, call a chat model, then switch to structured output with Pydantic.
2. Sprint 2 notebooks chunk files, index into ChromaDB, run semantic search, add BM25, tune hybrid alpha, then rerank.
3. Sprint 2 extension notebooks add HyDE and compare retrieved chunks before and after rewriting.
4. Sprint 3 notebooks define a tool schema, run one tool call, force a tool failure, then run a multi-step tool loop.
5. Sprint 3 MCP notebooks connect to the MCP server over stdio, inspect tools, call one capability, validate the response, and optionally use MCP Inspector locally.
6. Sprint 4 projects compose retrieval, reranking, HyDE, and at least one tool or MCP capability.

## Testing Strategy

- Unit tests use mock HTTP transports for OpenRouter payloads and error handling.
- Retrieval tests use fake embeddings where possible to avoid network calls.
- ChromaDB has a local persistent-client smoke test.
- LangGraph and MCP have construction-level tests so students can trust imports before notebook work.

## Extension Rules

- Add new models only after they are enabled for the course.
- Keep all external model calls in `OpenRouterClient`.
- Prefer pure functions and small adapters so live notebooks can reveal one concept at a time.
- Add tests before adding a new helper.
