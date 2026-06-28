"""SEC Disclosure Intelligence Prototype.

A modular, configuration-driven system for retrieval, structured extraction,
disclosure-change monitoring, grounded answer generation, and evaluation over
official SEC filings.

The public surface is intentionally small; import submodules directly for the
full API. Heavy optional backends (Chroma, sentence-transformers, Ollama,
OpenAI, Anthropic) are lazy-imported so the core runs with only the standard
library installed.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
