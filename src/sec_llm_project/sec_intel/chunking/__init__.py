"""SEC-aware chunking (plan item 4)."""

from __future__ import annotations

from .sec_sections import (
    SECChunker,
    Section,
    metadata_from_filename,
    split_into_sections,
)

__all__ = ["SECChunker", "Section", "metadata_from_filename", "split_into_sections"]
