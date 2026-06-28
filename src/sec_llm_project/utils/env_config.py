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
    """Load environment variables from .env file if available."""
    if not dotenv_available:
        return

    # Look for .env file in project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists() and dotenv_available and _load_dotenv is not None:
        _load_dotenv(env_file)


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
