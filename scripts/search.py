#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid import HybridRetriever
from hyde import HyDERewriter
from keyword_search import BM25Retriever
from openrouter import OpenRouterClient, OpenRouterEmbedder
from rerank import OpenRouterReranker
from vector_store import ChromaStore


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid search over a ChromaDB collection.")
    parser.add_argument("query")
    parser.add_argument("--db-path", default="./data/chroma")
    parser.add_argument("--collection", default="course_docs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--hyde", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    with OpenRouterClient(
        app_title=os.getenv("OPENROUTER_APP_TITLE", "ms-ai-ml-helper-core"),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
    ) as client:
        store = ChromaStore(
            path=args.db_path,
            collection_name=args.collection,
            embedder=OpenRouterEmbedder(client, input_type="search_query"),
        )
        docs = store.all_documents()
        keyword = BM25Retriever.from_documents(docs)
        retriever = HybridRetriever(vector_store=store, keyword_retriever=keyword, alpha=args.alpha)

        semantic_query = None
        keyword_query = None
        if args.hyde:
            rewrite = HyDERewriter(client).rewrite(args.query)
            semantic_query = rewrite.hypothetical_document
            keyword_query = rewrite.rewritten_query
            print(f"HyDE query: {rewrite.rewritten_query}\n")

        results = retriever.search(
            args.query,
            top_k=max(args.top_k, 10) if args.rerank else args.top_k,
            semantic_query=semantic_query,
            keyword_query=keyword_query,
        )
        if args.rerank:
            results = OpenRouterReranker(client).rerank(args.query, results, top_n=args.top_k)

    for index, result in enumerate(results[: args.top_k], start=1):
        score = result.rerank_score or result.hybrid_score or result.semantic_score or 0.0
        source = result.document.metadata.get("source_path") or result.document.id
        print(f"{index}. score={score:.4f} source={source}")
        print(result.document.text[:500].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
