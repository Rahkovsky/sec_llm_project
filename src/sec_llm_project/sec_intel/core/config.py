"""Configuration-driven architecture (plan item 11).

The whole platform is wired from a single :class:`AppConfig`. It can be loaded
from a dict, a JSON/YAML file, or environment variables, so backends are
selected through configuration rather than code changes (plan item 1). YAML is
optional; JSON always works with only the standard library.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints


@dataclass
class LLMConfig:
    # provider: mock | ollama | openai | anthropic
    provider: str = "mock"
    model: str = "mock-1"
    base_url: str = ""  # used by ollama / openai-compatible gateways
    api_key_env: str = ""  # name of env var holding the key (never the key itself)
    temperature: float = 0.0  # deterministic prompts by default (plan item 10)
    max_tokens: int = 768
    seed: int = 7
    timeout: float = 120.0


@dataclass
class EmbeddingConfig:
    # provider: openai (text-embedding-3-large) | nomic/huggingface (US-origin
    # local models, e.g. nomic-embed-text) | hashing (offline, no deps).
    provider: str = "hashing"
    model: str = "hashing-256"
    dim: int = 256  # used by hashing; informational for neural backends
    normalize: bool = True
    api_key_env: str = ""  # e.g. OPENAI_API_KEY for the openai provider
    base_url: str = ""
    # Pin a Hugging Face model to a specific commit. Important for
    # ``trust_remote_code`` models (e.g. Nomic): runs a reviewed version of the
    # custom modeling code and avoids re-downloading it each run. Empty falls back
    # to the embedder's built-in pin for known models, else the latest revision.
    revision: str = ""
    # Embedding version is recorded in the index sidecar for reproducibility
    # and to guard against querying with a mismatched model (plan item 2).
    version: str = "v1"


@dataclass
class ChunkingConfig:
    max_chars: int = 1800
    overlap_chars: int = 200
    min_chars: int = 200
    sec_aware: bool = True  # split on Item N. boundaries when present (plan item 4)


@dataclass
class RetrievalConfig:
    top_k: int = 8
    candidate_k: int = 40  # per-retriever candidate pool before fusion
    use_bm25: bool = True
    use_dense: bool = True
    fusion: str = "rrf"  # rrf | weighted
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class GenerationConfig:
    # Abstain when the best retrieval score is below this floor (plan item 10).
    min_evidence_score: float = 0.05
    min_citations: int = 1
    abstain_below_confidence: float = 0.15
    max_context_chunks: int = 6
    quote_chars: int = 240


@dataclass
class IndexConfig:
    # store: memory | chroma
    store: str = "memory"
    path: str = "data/sec_index"
    collection: str = "sec_filings"


@dataclass
class AuditConfig:
    enabled: bool = True
    path: str = "logs/audit.jsonl"


@dataclass
class FallbackConfig:
    """Local/private fallback stack (plan: compliance-aware, local mode).

    Used for air-gapped or API-unavailable deployments. Defaults to US-origin
    local models: Ollama Gemma 3 12B for generation and Nomic for embeddings.
    (12B fits 16-32 GB RAM; use gemma3:27b only on ~48 GB+ to avoid swapping.)
    """

    enabled: bool = False
    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(provider="ollama", model="gemma3:12b")
    )
    embedding: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(
            provider="nomic", model="nomic-ai/nomic-embed-text-v1.5", dim=768
        )
    )


@dataclass
class VerificationConfig:
    """Independent LLM-as-judge citation verification (claim-level entailment).

    The judge is configured separately from the generator so it can be a
    different model — ideally a different vendor — and never grades its own work.
    When ``judge.provider`` is an offline backend (mock/hashing), a deterministic
    lexical-overlap judge is used so the whole path runs in CI without keys.
    """

    enabled: bool = False
    # Cross-vendor by default in real configs (e.g. generator=openai, judge=anthropic);
    # the mock default keeps offline/tests deterministic and key-free.
    judge: LLMConfig = field(
        default_factory=lambda: LLMConfig(provider="mock", model="lexical-judge")
    )
    groundedness_floor: float = 0.6  # abstain below this fraction of supported claims
    repair: bool = True              # one-shot re-prompt to drop/fix flagged claims
    abstain_on_contradiction: bool = True


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    local_fallback: FallbackConfig = field(default_factory=FallbackConfig)

    def use_fallback(self) -> AppConfig:
        """Return a copy that promotes the local fallback stack to primary."""
        data = self.to_dict()
        data["llm"] = data["local_fallback"]["llm"]
        data["embedding"] = data["local_fallback"]["embedding"]
        return AppConfig.from_dict(data)

    # ------------------------------------------------------------------ loaders
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return _build_dataclass(cls, data or {})

    @classmethod
    def from_file(cls, path: str | Path) -> AppConfig:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        is_yaml = p.suffix.lower() in {".yaml", ".yml"}
        data = _load_yaml(text) if is_yaml else json.loads(text)
        return cls.from_dict(data if isinstance(data, dict) else {})

    @classmethod
    def from_env(cls, prefix: str = "SECI_") -> AppConfig:
        """Overlay environment variables onto defaults.

        Variables are dotted and case-insensitive, e.g.
        ``SECI_LLM__PROVIDER=ollama`` or ``SECI_RETRIEVAL__RERANK=true``.
        """
        overrides: dict[str, Any] = {}
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            path = key[len(prefix) :].lower().split("__")
            _assign(overrides, path, _coerce(value))
        base = cls().to_dict()
        _deep_merge(base, overrides)
        return cls.from_dict(base)

    @classmethod
    def load(cls, path: str | Path | None = None, env_prefix: str = "SECI_") -> AppConfig:
        """Load file (if given) then overlay environment variables on top."""
        cfg = cls.from_file(path) if path else cls()
        base = cfg.to_dict()
        env_cfg = cls.from_env(env_prefix).to_dict()
        # Only overlay env values that differ from defaults to avoid clobbering
        # file settings with defaults.
        default = cls().to_dict()
        diff: dict[str, Any] = {}
        _diff_against(default, env_cfg, diff)
        _deep_merge(base, diff)
        return cls.from_dict(base)


# ---------------------------------------------------------------------- helpers
def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    # ``from __future__ import annotations`` makes field.type a string, so resolve
    # real types via get_type_hints to detect nested dataclasses.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build_dataclass(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def _assign(tree: dict[str, Any], path: list[str], value: Any) -> None:
    node = tree
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _diff_against(default: dict[str, Any], candidate: dict[str, Any], out: dict[str, Any]) -> None:
    for k, v in candidate.items():
        if isinstance(v, dict) and isinstance(default.get(k), dict):
            sub: dict[str, Any] = {}
            _diff_against(default[k], v, sub)
            if sub:
                out[k] = sub
        elif default.get(k) != v:
            out[k] = v


def _coerce(value: str) -> Any:
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_yaml(text: str) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyYAML is required to load .yaml config; install pyyaml or use JSON."
        ) from exc
    return yaml.safe_load(text)
