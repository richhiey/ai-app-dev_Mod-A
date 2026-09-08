#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openrouter import OpenRouterClient


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
    parser = argparse.ArgumentParser(description="Call an enabled OpenRouter chat model.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    load_dotenv()
    with OpenRouterClient(
        app_title=os.getenv("OPENROUTER_APP_TITLE", "ms-ai-ml-helper-core"),
        http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
    ) as client:
        response = client.chat(
            [{"role": "user", "content": args.prompt}],
            model=args.model,
        )
    print(response.content)


if __name__ == "__main__":
    main()
