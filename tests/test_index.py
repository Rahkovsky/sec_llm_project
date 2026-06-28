from __future__ import annotations

import pytest

from sec_llm_project.sec_intel.core.types import Chunk, FilingMetadata
from sec_llm_project.sec_intel.embeddings.hashing import HashingEmbedder
from sec_llm_project.sec_intel.index.store import SECIndex


def _chunk(cid, text):
    return Chunk(chunk_id=cid, text=text,
                 metadata=FilingMetadata(ticker="ABC", filing_type="10-K"),
                 section_title="Risk Factors", item_number="1A")


def test_persist_and_load_roundtrip(tmp_path):
    emb = HashingEmbedder(dim=64)
    idx = SECIndex(emb.info, store="memory", path=str(tmp_path))
    chunks = [_chunk("c1", "alpha beta"), _chunk("c2", "gamma delta")]
    idx.add(chunks, emb.embed_documents([c.text for c in chunks]))
    idx.persist()

    loaded = SECIndex.load(str(tmp_path), expected=emb.info)
    assert len(loaded) == 2
    assert loaded.get("c1") is not None
    hits = loaded.dense_search(emb.embed_query("alpha"), k=1)
    assert hits and hits[0][0] == "c1"


def test_embedding_mismatch_rejected(tmp_path):
    emb = HashingEmbedder(dim=64)
    idx = SECIndex(emb.info, store="memory", path=str(tmp_path))
    idx.add([_chunk("c1", "alpha")], emb.embed_documents(["alpha"]))
    idx.persist()

    other = HashingEmbedder(dim=128)  # different fingerprint
    with pytest.raises(ValueError):
        SECIndex.load(str(tmp_path), expected=other.info)
