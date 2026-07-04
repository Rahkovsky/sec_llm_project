#!/usr/bin/env python3
"""Environment configuration utilities."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv as _load_dotenv

    dotenv_available = True
except ImportError:
    _load_dotenv = None  # type: ignore[assignment]
    dotenv_available = False


def load_environment() -> None:
    """Load environment variables from a repo-root ``.env`` if present.

    Uses python-dotenv when installed; otherwise falls back to a tiny stdlib
    parser so ``.env`` works even without the optional dependency (e.g. the
    local Ollama stack). Existing environment variables are never overridden, so
    an explicit ``export`` still wins.
    """
    project_root = Path(__file__).parents[3]  # src/sec_llm_project/utils/ -> repo root
    env_file = project_root / ".env"
    if not env_file.exists():
        return

    if dotenv_available and _load_dotenv is not None:
        _load_dotenv(env_file)
        return

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, sep, value = line.partition("=")
        if sep:
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_sec_user_info() -> tuple[str, str]:
    """
    Get SEC user information from environment variables.

    Returns:
        Tuple of (name, email) for SEC API identification
    """
    load_environment()

    name = os.getenv("SEC_USER_NAME", "Anonymous User")
    email = os.getenv("SEC_USER_EMAIL", "user@example.com")

    return name, email


def get_sec_user_agent() -> str:
    """
    Get properly formatted User-Agent string for SEC API.

    Returns:
        User-Agent string in format "Name email"
    """
    name, email = get_sec_user_info()
    return f"{name} {email}"
