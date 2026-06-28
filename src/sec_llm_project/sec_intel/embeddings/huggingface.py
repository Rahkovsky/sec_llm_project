# pyright: reportMissingImports=false, reportUnknownMemberType=false
"""Local neural embeddings via sentence-transformers (plan item 2).

Defaults to Nomic (``nomic-ai/nomic-embed-text-v1.5``), a US-origin open model,
for local/private deployments. The sentence-transformers dependency is imported
lazily so this module can be referenced without the package installed.
"""

from __future__ import annotations

from .base import EmbeddingInfo

# Per-model query/document instruction prefixes (improve retrieval quality).
_INSTRUCTIONS: dict[str, tuple[str, str]] = {
    "nomic": ("search_query: ", "search_document: "),
    "e5": ("query: ", "passage: "),
}

# Known-good commit pins for ``trust_remote_code`` models. Pinning runs a reviewed
# version of the custom modeling code (a security best-practice) and stops the
# remote code from being re-downloaded on every load. A config ``revision``
# overrides these.
_PINNED_REVISIONS: dict[str, str] = {
    "nomic-ai/nomic-embed-text-v1.5": "e9b6763023c676ca8431644204f50c2b100d9aab",
}


def _instructions(model: str) -> tuple[str, str]:
    name = model.lower()
    for key, val in _INSTRUCTIONS.items():
        if key in name:
            return val
    return ("", "")


class HuggingFaceEmbedder:
    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5", normalize: bool = True,
                 version: str = "v1", cache_folder: str | None = None,
                 revision: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        # Nomic ships custom modeling code; trust_remote_code is required. Pin the
        # revision (config override, else the built-in known-good commit) so a
        # reviewed version of that code runs and is not re-fetched each load.
        trust = "nomic" in model.lower()
        rev = revision or _PINNED_REVISIONS.get(model)
        # Nomic's pinned trust_remote_code calls get_extended_attention_mask, which
        # transformers ≥5.x deprecated. The transformers logger has propagate=False
        # and its own StreamHandler, so the filter must go on that handler directly.
        import logging

        class _DeprecFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "get_extended_attention_mask" not in record.getMessage()

        _f = _DeprecFilter()
        _tf = logging.getLogger("transformers")
        _tf.addFilter(_f)
        for _h in _tf.handlers:
            _h.addFilter(_f)
        self._model = SentenceTransformer(
            model, cache_folder=cache_folder, trust_remote_code=trust, revision=rev
        )
        self._q_instr, self._d_instr = _instructions(model)
        self.normalize = normalize
        # ``get_sentence_embedding_dimension`` was renamed to
        # ``get_embedding_dimension`` in sentence-transformers 5.x; prefer the new
        # name and fall back for older (>=2.7) versions.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        dim = int(get_dim())
        self.info = EmbeddingInfo(
            backend="huggingface", model=model, dim=dim,
            normalize=normalize, version=version,
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=self.normalize, convert_to_numpy=True
        )
        return [list(map(float, v)) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self._d_instr + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._q_instr + text])[0]
