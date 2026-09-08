# Course Concept Map

Source files read in full:

- `Copy of AI_ML_T5_T6_Final_Design - App Dev - Module A - Live.csv`
- `Copy of AI_ML_T5_T6_Final_Design - App Dev - Module A - Campus.csv`

## Baseline

Students arrive from T4 with Python, scikit-learn, MLflow, and Streamlit. The course does not assume prior LLM, RAG, MCP, or production service experience. The recurring skill is extending scaffolded AI services rather than writing everything from scratch.

## Sprint 1: AI Application Anatomy

- Identify model access, retrieval, tools, memory, and orchestration in an AI app.
- Compare subscription UI, embedded copilot, and direct API use.
- Read direct model request and response shapes.
- Explain why structured JSON matters when an AI app feeds another system.
- Label extension points beyond visual builders.

Repo support:

- `OpenRouterClient.chat`
- `OpenRouterClient.structured`
- `StructuredOutputGraph`
- model allowlist enforcement

## Sprint 2: Advanced Retrieval

- Diagnose naive RAG failures: weak chunks, irrelevant retrieval, and context overflow.
- Add reranking and compare before/after retrieval quality.
- Combine keyword and semantic search.
- Tune the hybrid blend on test queries.
- Use query rewriting and HyDE to improve recall.

Repo support:

- `Document` and `chunk_text`
- `ChromaStore`
- `BM25Retriever`
- `HybridRetriever`
- `OpenRouterReranker`
- `HyDERewriter`

## Sprint 3: Tool Use and MCP

- Explain tool use as application-level action.
- Distinguish multi-step tool reasoning from one-shot answers.
- Read and modify tool schemas.
- Handle malformed responses, timeouts, and missing data.
- Explain MCP, connect to a server, inspect a response, and decide when MCP is worthwhile.

Repo support:

- `ToolRegistry`
- `ToolCallingAgent`
- structured tool execution errors
- `mcp_server.py`

## Sprint 4: Project Application

- Choose a fresh business case.
- Scope inputs, outputs, and success criteria.
- Extend the scaffold with retrieval and at least one tool or MCP integration.
- Stress-test realistic edge cases and document limitations.

Repo support:

- composable scripts in `scripts/`
- unit tests as examples of expected behavior
- reusable docs and notebook snippets
