#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from documents import chunk_text
from openrouter import OpenRouterClient, OpenRouterEmbedder
from vector_store import ChromaStore

SUPPORTED_SUFFIXES = {".txt", ".md", ".py"}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def iter_source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk and index a directory into ChromaDB.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--db-path", default="./data/chroma")
    parser.add_argument("--collection", default="course_docs")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    load_dotenv()
    files = iter_source_files(args.source)
    if not files:
        raise SystemExit(f"No supported files found under {args.source}")

    docs = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        docs.extend(
            chunk_text(
                text,
                source_id=path.as_posix(),
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                metadata={"source_path": path.as_posix()},
            )
        )

    with OpenRouterClient(
        app_title=os.getenv("OPENROUTER_APP_TITLE", "ms-ai-ml-helper-core"),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
    ) as client:
        store = ChromaStore(
            path=args.db_path,
            collection_name=args.collection,
            embedder=OpenRouterEmbedder(client, input_type="search_document"),
        )
        count = store.index(docs)

    print(f"Indexed {count} chunks into collection '{args.collection}'.")


if __name__ == "__main__":
    main()
