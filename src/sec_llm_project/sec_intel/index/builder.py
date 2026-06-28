"""Reproducible index construction (plan items 2, 11).

Ties the chunker and embedder to a :class:`SECIndex`. Indexing is deterministic:
the same corpus + config yields the same chunk ids and vectors, and the
embedding fingerprint is recorded for later verification.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..chunking.sec_sections import (
    SECChunker,
    metadata_from_filename,
    sections_from_sidecar,
)
from ..core.config import AppConfig
from ..core.types import Chunk, FilingMetadata
from ..embeddings.base import Embedder
from ..embeddings.factory import build_embedder
from .store import SECIndex

# Map a directory/file naming hint to a filing type for provenance.
_TYPE_HINTS = {
    "10-K": "10-K", "10K": "10-K", "10-Q": "10-Q", "10Q": "10-Q",
    "8-K": "8-K", "8K": "8-K", "DEF14A": "DEF 14A", "DEF 14A": "DEF 14A",
}


def _infer_filing_type(path: Path, default: str = "10-K") -> str:
    parts = {p.upper() for p in path.parts}
    for hint, ftype in _TYPE_HINTS.items():
        if hint.upper() in parts:
            return ftype
    return default


def _load_sidecar(txt_path: Path) -> dict | None:
    """Load the ``<stem>.json`` provenance sidecar written at download time, if present."""
    sidecar = txt_path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) and "metadata" in data else None


class IndexBuilder:
    def __init__(self, config: AppConfig | None = None, embedder: Embedder | None = None) -> None:
        self.config = config or AppConfig()
        self.embedder = embedder or build_embedder(self.config.embedding)
        self.chunker = SECChunker(self.config.chunking)

    # ----------------------------------------------------------- building
    def _new_index(self) -> SECIndex:
        return SECIndex(
            self.embedder.info,
            store=self.config.index.store,
            path=self.config.index.path,
            collection=self.config.index.collection,
        )

    def build_from_chunks(self, chunks: list[Chunk], index: SECIndex | None = None) -> SECIndex:
        # NB: an empty SECIndex is falsy (len == 0), so test identity explicitly.
        if index is None:
            index = self._new_index()
        if chunks:
            vectors = self.embedder.embed_documents([c.text for c in chunks])
            index.add(chunks, vectors)
        return index

    def chunk_text(self, text: str, metadata: FilingMetadata) -> list[Chunk]:
        return self.chunker.chunk_filing(text, metadata)

    def build_from_dir(self, input_dir: str | Path, *, pattern: str = "*.txt",
                       default_filing_type: str = "10-K",
                       max_files: int | None = None,
                       persist: bool = True,
                       _index: SECIndex | None = None) -> SECIndex:
        base = Path(input_dir)
        if not base.exists():
            raise FileNotFoundError(f"Input directory not found: {base}")
        index = _index if _index is not None else self._new_index()
        processed = 0
        for file_path in sorted(base.rglob(pattern)):
            if not file_path.is_file():
                continue
            if max_files is not None and processed >= max_files:
                break
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue
            sidecar = _load_sidecar(file_path)
            if sidecar:
                # Rich provenance + segmentation captured at download time.
                metadata = FilingMetadata.from_dict(sidecar["metadata"])
                sections = sections_from_sidecar(sidecar, text) or None
                chunks = self.chunker.chunk_filing(text, metadata, sections=sections)
            else:
                # Legacy fallback: derive what we can from the filename.
                filing_type = _infer_filing_type(file_path, default_filing_type)
                metadata = metadata_from_filename(file_path, filing_type=filing_type)
                chunks = self.chunker.chunk_filing(text, metadata)
            self.build_from_chunks(chunks, index)
            processed += 1
        if persist:
            index.persist()
        return index

    def build_from_dirs(self, input_dirs: list[str | Path], *, pattern: str = "*.txt",
                        default_filing_type: str = "10-K",
                        max_files: int | None = None,
                        persist: bool = True) -> SECIndex:
        """Build one unified index from multiple directories (e.g. 10-K + 10-Q + 8-K).

        Form types are inferred per-file from the directory path, so mixing
        form directories into one index works correctly.
        """
        index = self._new_index()
        for d in input_dirs:
            self.build_from_dir(d, pattern=pattern, default_filing_type=default_filing_type,
                                max_files=max_files, persist=False, _index=index)
        if persist:
            index.persist()
        return index
