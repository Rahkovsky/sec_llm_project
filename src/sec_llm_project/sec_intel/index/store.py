# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Chunk catalog + vector store.

:class:`SECIndex` owns a catalog of chunks (with full provenance) plus a dense
vector store. Two store backends are provided:

* ``memory`` — pure-Python cosine search, persisted as JSON. No dependencies.
* ``chroma`` — delegates vector storage/search to Chroma when configured.

The catalog is always kept locally so the lexical (BM25) retriever and the
chunk lookups used for citations work uniformly regardless of backend. The
embedding fingerprint is persisted and verified on load to prevent querying an
index with a mismatched embedder (plan item 2).
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.types import Chunk
from ..embeddings.base import EmbeddingInfo


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _matches_filter(chunk: Chunk, where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    meta = chunk.metadata.to_dict()
    meta["section_title"] = chunk.section_title
    meta["item_number"] = chunk.item_number
    for key, want in where.items():
        have = meta.get(key)
        if isinstance(want, list | tuple | set):
            if have not in {str(w) for w in want}:
                return False
        elif str(have) != str(want):
            return False
    return True


class SECIndex:
    """Persistent chunk catalog with dense search."""

    def __init__(self, embedding_info: EmbeddingInfo, store: str = "memory",
                 path: str = "data/sec_index", collection: str = "sec_filings") -> None:
        self.embedding_info = embedding_info
        self.store = store
        self.path = path
        self.collection = collection
        # True only when populated by ``load`` from persisted artifacts, so the
        # sidecar ``bm25.json`` at ``path`` is known to match these chunks. A
        # freshly built in-memory index must not adopt a stale/foreign bm25.json.
        self.loaded_from_disk = False
        self._chunks: dict[str, Chunk] = {}
        # Lazy text (indexes loaded from disk): ``_chunks`` then holds metadata-only
        # stubs (text "") and ``_text_at`` maps chunk id -> (offset, length) of its
        # line in chunks.jsonl, read on demand by ``get``. Chunk text is ~2 GB of
        # Python strings on the 730k-chunk index (most chunks contain a curly
        # quote or dash, so CPython stores them at 2 bytes/char); a query reads
        # only the few dozen chunks it retrieves.
        self._text_at: dict[str, tuple[int, int]] = {}
        self._text_fd: int | None = None
        self._vectors: dict[str, list[float]] = {}
        self._chroma_collection: Any = None
        if store == "chroma":
            self._init_chroma()

    # ----------------------------------------------------------------- chroma
    def _init_chroma(self) -> None:
        import chromadb  # lazy

        os.makedirs(self.path, exist_ok=True)
        client = chromadb.PersistentClient(path=os.path.join(self.path, "chroma_db"))
        self._chroma_collection = client.get_or_create_collection(
            self.collection, metadata={"hnsw:space": "cosine"}
        )

    # -------------------------------------------------------------- mutation
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._text_at.pop(chunk.chunk_id, None)  # the new text is in memory
        if self.store == "chroma":
            self._chroma_collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[self._meta_for_chroma(c) for c in chunks],
            )
        else:
            for chunk, vec in zip(chunks, vectors, strict=False):
                self._vectors[chunk.chunk_id] = vec

    @staticmethod
    def _meta_for_chroma(chunk: Chunk) -> dict[str, Any]:
        meta = chunk.metadata.to_dict()
        meta["section_title"] = chunk.section_title
        meta["item_number"] = chunk.item_number
        return meta

    # --------------------------------------------------------------- queries
    def all_chunks(self) -> list[Chunk]:
        """Every chunk with its text. On a lazily loaded index this reads the whole
        catalog from disk (~2 GB of strings); use :meth:`catalog` when only
        metadata is needed."""
        if not self._text_at:
            return list(self._chunks.values())
        return [c for c in map(self.get, self._chunks) if c is not None]

    def catalog(self) -> list[Chunk]:
        """Every chunk's id and metadata (section, item, filing). ``text`` may be
        empty: do not read it from these objects, use :meth:`get`."""
        return list(self._chunks.values())

    def get(self, chunk_id: str) -> Chunk | None:
        chunk = self._chunks.get(chunk_id)
        at = self._text_at.get(chunk_id) if chunk is not None else None
        if at is None or self._text_fd is None:
            return chunk
        raw = os.pread(self._text_fd, at[1], at[0])
        return replace(chunk, text=str(json.loads(raw).get("text", "")))

    def __len__(self) -> int:
        return len(self._chunks)

    def vectors(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Stored embeddings for ``chunk_ids`` (missing ids are omitted)."""
        if self.store == "chroma":
            got = self._chroma_collection.get(ids=list(chunk_ids), include=["embeddings"])
            return {cid: list(v) for cid, v in zip(got["ids"], got["embeddings"], strict=False)}
        return {cid: self._vectors[cid] for cid in chunk_ids if cid in self._vectors}

    def dense_search(self, query_vector: list[float], k: int,
                     where: dict[str, Any] | None = None) -> list[tuple[str, float]]:
        if self.store == "chroma":
            return self._dense_search_chroma(query_vector, k, where)
        scored: list[tuple[str, float]] = []
        for cid, vec in self._vectors.items():
            chunk = self._chunks[cid]
            if not _matches_filter(chunk, where):
                continue
            scored.append((cid, _cosine(query_vector, vec)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def _dense_search_chroma(self, query_vector: list[float], k: int,
                             where: dict[str, Any] | None) -> list[tuple[str, float]]:
        chroma_where = None
        if where:
            clauses = [
                {key: {"$in": [str(v) for v in val]}}
                if isinstance(val, list | tuple | set)
                else {key: str(val)}
                for key, val in where.items()
            ]
            chroma_where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._chroma_collection.query(
            query_embeddings=[query_vector], n_results=k, where=chroma_where
        )
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        # Chroma cosine distance -> similarity.
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances, strict=False)]

    # ------------------------------------------------------------ persistence
    def persist(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        sidecar = {
            "embedding_info": self.embedding_info.to_dict(),
            "store": self.store,
            "collection": self.collection,
            "chunk_count": len(self._chunks),
        }
        Path(self.path, "index_meta.json").write_text(
            json.dumps(sidecar, indent=2), encoding="utf-8"
        )
        # Write a temp file and rename: a lazily loaded index reads its text from
        # the chunks.jsonl being replaced (the open descriptor keeps the old file).
        target = Path(self.path, "chunks.jsonl")
        tmp = target.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for cid in self._chunks:
                chunk = self.get(cid)
                assert chunk is not None
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, target)
        if self.store == "memory":
            with open(Path(self.path, "vectors.jsonl"), "w", encoding="utf-8") as fh:
                for cid, vec in self._vectors.items():
                    fh.write(json.dumps({"id": cid, "v": vec}) + "\n")

    @classmethod
    def load(cls, path: str, *, expected: EmbeddingInfo | None = None) -> SECIndex:
        sidecar = json.loads(Path(path, "index_meta.json").read_text(encoding="utf-8"))
        info = EmbeddingInfo(**sidecar["embedding_info"])
        if expected is not None and expected.fingerprint() != info.fingerprint():
            raise ValueError(
                "Embedding mismatch: index was built with "
                f"'{info.fingerprint()}' but query uses '{expected.fingerprint()}'. "
                "Rebuild the index or query with the matching embedder."
            )
        index = cls(info, store=sidecar.get("store", "memory"), path=path,
                    collection=sidecar.get("collection", "sec_filings"))
        chunks_file = Path(path, "chunks.jsonl")
        metas: dict[Any, Any] = {}  # one shared (frozen) FilingMetadata per filing
        offset = 0
        with open(chunks_file, "rb") as fh:
            for raw in fh:
                if raw.strip():
                    data = json.loads(raw)
                    data["text"] = ""
                    chunk = Chunk.from_dict(data)
                    chunk.metadata = metas.setdefault(chunk.metadata, chunk.metadata)
                    chunk.section_title = sys.intern(chunk.section_title)
                    chunk.item_number = sys.intern(chunk.item_number)
                    index._chunks[chunk.chunk_id] = chunk
                    index._text_at[chunk.chunk_id] = (offset, len(raw))
                offset += len(raw)
        index._text_fd = os.open(chunks_file, os.O_RDONLY)
        vec_file = Path(path, "vectors.jsonl")
        if index.store == "memory" and vec_file.exists():
            with open(vec_file, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        index._vectors[rec["id"]] = rec["v"]
        index.loaded_from_disk = True
        return index
