"""High-level platform facade wiring all components from one config.

This is the main entry point most callers want: build (or load) an index, then
ask grounded questions, run structured extraction, or compare filings — all with
dependency injection driven by :class:`AppConfig` (plan items 1, 11).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .comparison.differ import ComparisonReport, FilingComparer
from .core.audit import AuditLogger
from .core.config import AppConfig
from .core.tracing import Tracer, build_tracer
from .core.types import Answer, Chunk, FilingMetadata
from .embeddings.base import Embedder
from .embeddings.factory import build_embedder
from .extraction.extractor import ExtractionResult, StructuredExtractor
from .generation.grounded import GroundedAnswerer
from .generation.verifier import build_verifier
from .index.builder import IndexBuilder
from .index.store import SECIndex
from .llm.base import LLMProvider
from .llm.factory import build_llm
from .retrieval.entity import EntityResolver, Resolution
from .retrieval.hybrid import HybridRetriever


class SECIntelPipeline:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.embedder: Embedder = build_embedder(self.config.embedding)
        self.llm: LLMProvider = build_llm(self.config.llm)
        self.audit = AuditLogger.from_config(self.config.audit)
        self.tracer: Tracer = build_tracer(self.config.observability)
        self.builder = IndexBuilder(self.config, embedder=self.embedder)
        self.verifier = build_verifier(self.config.verification, generator=self.config.llm)
        self._index: SECIndex | None = None
        self._retriever: HybridRetriever | None = None
        self._resolver: EntityResolver | None = None

    # ----------------------------------------------------------- index setup
    def build_index(self, input_dir: str | Path | list[str | Path], **kwargs: Any) -> SECIndex:
        if isinstance(input_dir, list):
            self._index = self.builder.build_from_dirs(input_dir, **kwargs)
        else:
            self._index = self.builder.build_from_dir(input_dir, **kwargs)
        self._retriever = None
        return self._index

    def index_chunks(self, chunks: list[Chunk]) -> SECIndex:
        self._index = self.builder.build_from_chunks(chunks)
        self._retriever = None
        return self._index

    def index_text(self, text: str, metadata: FilingMetadata) -> SECIndex:
        return self.index_chunks(self.builder.chunk_text(text, metadata))

    def load_index(self) -> SECIndex:
        self._index = SECIndex.load(self.config.index.path, expected=self.embedder.info)
        self._retriever = None
        return self._index

    @property
    def index(self) -> SECIndex:
        if self._index is None:
            raise RuntimeError("No index loaded. Call build_index/load_index first.")
        return self._index

    def _build_reranker(self) -> Any:
        if not self.config.retrieval.rerank:
            return None
        from .retrieval.rerank import CrossEncoderReranker

        return CrossEncoderReranker(self.config.retrieval.reranker_model)

    def _build_expander(self) -> Any:
        mode = self.config.retrieval.query_expansion
        if not mode:
            return None
        if mode != "hyde":
            raise ValueError(f"unknown retrieval.query_expansion: {mode!r}")
        from .retrieval.rewrite import HydeExpander

        return HydeExpander(self.llm, self.resolver.mask)

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                self.index, self.embedder, self.config.retrieval,
                reranker=self._build_reranker(), expander=self._build_expander(),
            )
        return self._retriever

    # ------------------------------------------------------------- use cases
    @property
    def resolver(self) -> EntityResolver:
        """Company resolver over the issuers actually present in the loaded index."""
        if self._resolver is None:
            names: dict[str, str] = {}
            for c in self.index.catalog():
                m = c.metadata
                if m.ticker not in names and m.company not in ("", "UNKNOWN"):
                    names[m.ticker] = m.company
            self._resolver = EntityResolver(names)
        return self._resolver

    def resolve_filters(self, question: str, *, scope: str = "primary"
                        ) -> tuple[dict[str, Any] | None, Resolution]:
        """Ticker filter from the question text. ``scope``: "primary" (the main
        company only) or "all" (every company named, for comparisons)."""
        res = self.resolver.resolve(question)
        if not res.tickers:
            return None, res
        tickers = res.tickers[:1] if scope == "primary" else res.tickers
        return {"ticker": tickers[0] if len(tickers) == 1 else tickers}, res

    def ask(self, question: str, *, filters: dict[str, Any] | None = None,
            top_k: int | None = None) -> Answer:
        answerer = GroundedAnswerer(
            self.retriever, self.llm, self.config.generation, audit=self.audit,
            verifier=self.verifier, tracer=self.tracer,
        )
        return answerer.answer(question, filters=filters, top_k=top_k)

    def extract(self, target: str, *, ticker: str | None = None,
                filing_type: str | None = None, top_k: int = 8) -> ExtractionResult:
        extractor = StructuredExtractor(self.retriever, self.llm, audit=self.audit)
        return extractor.extract(target, ticker=ticker, filing_type=filing_type, top_k=top_k)

    def compare_years(self, ticker: str, year_a: str, year_b: str, *,
                      filing_type: str = "10-K",
                      items: list[str] | None = None) -> ComparisonReport:
        return FilingComparer(self.index).compare_years(
            ticker, year_a, year_b, filing_type=filing_type, items=items
        )

    def monitor(self, ticker: str, year_a: str, year_b: str, *,
                forms: list[str] | None = None, compare_form: str = "10-K",
                xbrl: bool = False) -> Any:
        """Run the Disclosure Change & Risk Signal Monitor (flagship workflow)."""
        from .monitor.monitor import DisclosureMonitor

        _ = self.index  # ensure an index is loaded before monitoring
        return DisclosureMonitor(self).monitor(
            ticker, year_a, year_b, forms=forms, compare_form=compare_form, xbrl=xbrl
        )
