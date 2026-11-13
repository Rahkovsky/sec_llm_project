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


def test_mmr_skips_redundant_duplicate():
    from dataclasses import replace
    dup = _chunk("c1b", CHUNKS[0].text)  # same text, another filing year
    emb = HashingEmbedder(dim=128)
    idx = SECIndex(emb.info, store="memory")
    chunks = [*CHUNKS, dup]
    idx.add(chunks, emb.embed_documents([c.text for c in chunks]))
    base = replace(AppConfig().retrieval, use_bm25=False)
    plain = HybridRetriever(idx, emb, base).retrieve("supply chain manufacturing", top_k=2)
    assert {r.chunk.chunk_id for r in plain} == {"c1", "c1b"}
    mmr = HybridRetriever(idx, emb, replace(base, mmr_lambda=0.5)).retrieve(
        "supply chain manufacturing", top_k=2)
    ids = [r.chunk.chunk_id for r in mmr]
    assert ids[0] in {"c1", "c1b"} and ids[1] not in {"c1", "c1b"}
    assert [r.rank for r in mmr] == [0, 1]


class _FakeExpander:
    def __init__(self, text):
        self.text, self.calls = text, []

    def expand(self, query, filters=None):
        self.calls.append((query, filters))
        return [self.text]


def test_expansion_pulls_in_vocabulary_match():
    from dataclasses import replace
    emb = HashingEmbedder(dim=256)
    idx = SECIndex(emb.info, store="memory")
    idx.add(CHUNKS, emb.embed_documents([c.text for c in CHUNKS]))
    cfg = replace(AppConfig().retrieval, use_bm25=False, mmr_lambda=0.85)
    q = "hackers"  # shares no word with the cybersecurity chunk
    exp = _FakeExpander("Cybersecurity breaches could expose customer data.")
    r = HybridRetriever(idx, emb, cfg, expander=exp)
    res = r.retrieve(q, top_k=1, filters={"ticker": "ABC"})
    assert res[0].chunk.chunk_id == "c4"
    assert exp.calls == [(q, {"ticker": "ABC"})]
    assert r.last_expansions == [exp.text]


def test_hyde_expander_prompt_and_failure():
    from sec_llm_project.sec_intel.llm.base import GenerationResult
    from sec_llm_project.sec_intel.retrieval.rewrite import HydeExpander

    class LLM:
        def __init__(self, fail=False):
            self.fail, self.prompts = fail, []

        def generate(self, messages, **kw):
            if self.fail:
                raise RuntimeError("down")
            self.prompts.append(messages[-1].content)
            return GenerationResult(text=" A passage. ", model="m")

    llm = LLM()
    from sec_llm_project.sec_intel.retrieval.entity import EntityResolver

    res = EntityResolver({"ES": "EVERSOURCE ENERGY", "AAPL": "Apple Inc."})
    x = HydeExpander(llm, res.mask)
    assert x.expand("Loss contingencies under ES", {"ticker": "ES"}) == ["A passage."]
    assert llm.prompts[0].endswith("Question: Loss contingencies under the company")
    assert "ES" not in llm.prompts[0].split("Question:")[1]  # company-blind
    x.expand("Loss contingencies under ES", {"ticker": "ES"})
    assert len(llm.prompts) == 1  # cached
    assert HydeExpander(LLM(fail=True)).expand("q") == []


def test_hyde_expander_drops_refusals():
    from sec_llm_project.sec_intel.llm.base import GenerationResult
    from sec_llm_project.sec_intel.retrieval.rewrite import HydeExpander, is_refusal

    class LLM:
        def generate(self, messages, **kw):
            return GenerationResult(text="Please provide the relevant text from the 10-K.",
                                    model="m")

    assert HydeExpander(LLM()).expand("How many acquisitions?") == []
    assert not is_refusal("The Company cannot guarantee that its hedging will be effective.")


def test_named_years_and_period_filter():
    from sec_llm_project.sec_intel.retrieval.period import named_years, period_filter

    assert named_years("What revenue in fiscal 2023 vs 2022?") == [2023, 2022]
    assert named_years("FY24 gross margin") == [2024]
    assert named_years("AWS revenue on March 3, 2024") == [2024]
    assert named_years("sales of $2023 million, 2,024 units") == []
    assert period_filter("risks in 2023", ["2022", "2023", "2024", "2025"]) == ["2023", "2024"]
    assert period_filter("forecast in fiscal 2028", ["2023", "2024"]) is None  # not covered
    assert period_filter("what are the risks", ["2024"]) is None


def test_period_filter_restricts_fiscal_year():
    from dataclasses import replace

    def fy(c, y):
        return replace(c, metadata=replace(c.metadata, fiscal_year=y))

    chunks = [fy(_chunk("old", "Revenue increased due to cloud demand.", item="7"), "2021"),
              fy(_chunk("new", "Revenue increased due to cloud demand in total.", item="7"), "2023")]
    emb = HashingEmbedder(dim=128)
    idx = SECIndex(emb.info, store="memory")
    idx.add(chunks, emb.embed_documents([c.text for c in chunks]))
    cfg = AppConfig().retrieval
    cfg.period_filter = True
    cfg.period_mode = "hard"
    r = HybridRetriever(idx, emb, cfg)
    got = r.retrieve("revenue growth in 2023", top_k=5, filters={"ticker": "ABC"})
    assert [g.chunk.chunk_id for g in got] == ["new"] and r.last_period == ["2023"]
    got = r.retrieve("revenue growth in 2030", top_k=5, filters={"ticker": "ABC"})
    assert {g.chunk.chunk_id for g in got} == {"old", "new"} and r.last_period is None
    # soft: every year stays, the in-period chunk is boosted to the top (also under MMR)
    for lam in (None, 0.85):
        cfg.period_mode, cfg.period_boost, cfg.mmr_lambda = "soft", 1.0, lam
        got = r.retrieve("revenue increased due to cloud demand in 2023", top_k=5,
                         filters={"ticker": "ABC"})
        assert [g.chunk.chunk_id for g in got] == ["new", "old"], lam
        got = r.retrieve("revenue increased due to cloud demand in 2021", top_k=5,
                         filters={"ticker": "ABC"})
        assert next(g.chunk.chunk_id for g in got) == "old", lam
