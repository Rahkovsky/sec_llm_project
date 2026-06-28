from __future__ import annotations

from sec_llm_project.sec_intel.core.config import AppConfig
from sec_llm_project.sec_intel.core.types import Chunk, FilingMetadata
from sec_llm_project.sec_intel.embeddings.hashing import HashingEmbedder
from sec_llm_project.sec_intel.index.store import SECIndex
from sec_llm_project.sec_intel.retrieval.bm25 import BM25Index
from sec_llm_project.sec_intel.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from sec_llm_project.sec_intel.retrieval.hybrid import HybridRetriever


def _chunk(cid, text, ticker="ABC", item="1A"):
    return Chunk(chunk_id=cid, text=text,
                 metadata=FilingMetadata(ticker=ticker, filing_type="10-K"),
                 section_title="Risk Factors", item_number=item)


CHUNKS = [
    _chunk("c1", "Supply chain disruptions could harm our manufacturing operations."),
    _chunk("c2", "We face intense competition from larger rivals.", item="1A"),
    _chunk("c3", "Revenue increased due to strong cloud demand.", item="7"),
    _chunk("c4", "Cybersecurity breaches could expose customer data.", item="1A"),
]


def test_bm25_finds_lexical_match():
    bm = BM25Index(CHUNKS)
    hits = bm.search("supply chain manufacturing", k=3)
    assert hits
    assert hits[0][0] == "c1"


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert set(fused) == {"a", "b", "c", "d"}
    # items appearing high in both lists score highest
    assert fused["b"] >= fused["c"]


def test_weighted_fusion_normalizes():
    fused = weighted_fusion([{"a": 10.0, "b": 0.0}, {"a": 0.0, "b": 5.0}], [0.5, 0.5])
    assert abs(fused["a"] - fused["b"]) < 1e-9


def _index():
    emb = HashingEmbedder(dim=128)
    idx = SECIndex(emb.info, store="memory")
    idx.add(CHUNKS, emb.embed_documents([c.text for c in CHUNKS]))
    return idx, emb


def test_hybrid_retrieve_supply_chain():
    idx, emb = _index()
    r = HybridRetriever(idx, emb, AppConfig().retrieval)
    results = r.retrieve("supply chain risk", top_k=3)
    assert results
    assert results[0].chunk.chunk_id == "c1"
    assert "dense" in results[0].components and "bm25" in results[0].components


def test_metadata_filter_restricts_items():
    idx, emb = _index()
    r = HybridRetriever(idx, emb, AppConfig().retrieval)
    results = r.retrieve("revenue", top_k=5, filters={"item_number": "7"})
    assert results
    assert all(res.chunk.item_number == "7" for res in results)
