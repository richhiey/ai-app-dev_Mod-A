"""Credential setup shared by the course notebooks and live integration checks."""

from __future__ import annotations

import os
from getpass import getpass

from dotenv import load_dotenv


def require_openrouter_key(*, prompt: bool = False) -> str:
    """Load a real key without displaying it; missing credentials are an error."""
    load_dotenv(override=False)
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        try:
            from google.colab import userdata

            key = userdata.get("OPENROUTER_API_KEY") or ""
        except ImportError:
            pass
        except Exception:
            # Colab also raises when the secret is absent or access is not granted.
            key = ""
    if not key and prompt:
        key = getpass("OpenRouter API key: ").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required. Set it in Colab Secrets, the environment, "
            "or your local ignored .env file. No simulated responses are available."
        )
    os.environ["OPENROUTER_API_KEY"] = key
    return key
