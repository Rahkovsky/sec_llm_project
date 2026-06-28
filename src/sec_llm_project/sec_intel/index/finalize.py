"""Finalize a persisted Chroma-backed index for fast, restart-proof querying.

Runs once after data processing (embedding) completes. Chroma is the complete
source of truth (additive + idempotent across restarts), so we rebuild both
query-time artifacts *from Chroma*:

* ``chunks.jsonl`` — the chunk store read by ``SECIndex.load`` (feeds citation
  text and the lexical index). Rebuilding from Chroma makes it correct even if
  the indexer auto-restarted (whose in-memory set would otherwise be partial).
* ``bm25.json`` — the BM25 lexical index, built here once instead of being
  re-tokenized on the first query of every session.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.types import Chunk, FilingMetadata
from ..retrieval.bm25 import BM25Index


def finalize_index(
    path: str | Path,
    *,
    collection: str = "sec_filings",
    embedding_info: dict[str, Any] | None = None,
    batch: int = 20000,
    log: Callable[[str], None] = print,
) -> int:
    """Rebuild ``chunks.jsonl``, ``index_meta.json``, and ``bm25.json`` from the
    Chroma collection at ``<path>/chroma_db``. Returns the chunk count."""
    import chromadb

    path = Path(path)
    client = chromadb.PersistentClient(path=str(path / "chroma_db"))
    col = client.get_collection(collection)
    total = col.count()
    log(f"finalize: {total:,} chunks in Chroma -> rebuilding chunks.jsonl")

    chunks: list[Chunk] = []
    with open(path / "chunks.jsonl", "w", encoding="utf-8") as fh:
        offset = 0
        while offset < total:
            got = col.get(include=["documents", "metadatas"], limit=batch, offset=offset)
            ids = got.get("ids") or []
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            for cid, doc, meta in zip(ids, docs, metas, strict=False):
                m = meta or {}
                md = {k: v for k, v in m.items()
                      if k not in ("section_title", "item_number")}
                chunk = Chunk(
                    chunk_id=cid,
                    text=doc or "",
                    metadata=FilingMetadata.from_dict(md),
                    section_title=str(m.get("section_title", "UNKNOWN")),
                    item_number=str(m.get("item_number", "UNKNOWN")),
                )
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                chunks.append(chunk)
            offset += batch
            log(f"  reconstructed {min(offset, total):,}/{total:,}")

    # Sidecar: reuse the existing embedding_info if not supplied (avoids loading
    # the embedding model just to record its fingerprint).
    meta_path = path / "index_meta.json"
    info = embedding_info
    if info is None and meta_path.exists():
        info = json.loads(meta_path.read_text(encoding="utf-8")).get("embedding_info")
    meta_path.write_text(json.dumps({
        "embedding_info": info,
        "store": "chroma",
        "collection": collection,
        "chunk_count": len(chunks),
    }, indent=2), encoding="utf-8")

    log("finalize: building + saving BM25 lexical index")
    BM25Index(chunks).save(path / "bm25.json")
    log(f"finalize: done — chunks.jsonl ({len(chunks):,}), bm25.json, index_meta.json")
    return len(chunks)
