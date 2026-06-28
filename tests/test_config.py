from __future__ import annotations

import os

from sec_llm_project.sec_intel.core.config import AppConfig


def test_defaults_offline():
    cfg = AppConfig()
    assert cfg.llm.provider == "mock"
    assert cfg.embedding.provider == "hashing"
    assert cfg.retrieval.use_bm25 and cfg.retrieval.use_dense


def test_from_dict_builds_nested_dataclasses():
    cfg = AppConfig.from_dict({"llm": {"provider": "ollama", "model": "mistral:7b"},
                               "retrieval": {"rerank": True}})
    assert cfg.llm.provider == "ollama"
    assert cfg.llm.model == "mistral:7b"
    assert cfg.retrieval.rerank is True
    # untouched fields keep defaults
    assert cfg.embedding.provider == "hashing"


def test_round_trip():
    cfg = AppConfig.from_dict({"index": {"store": "chroma"}})
    assert AppConfig.from_dict(cfg.to_dict()).index.store == "chroma"


def test_env_override(monkeypatch=None):
    os.environ["SECI_LLM__PROVIDER"] = "anthropic"
    os.environ["SECI_RETRIEVAL__TOP_K"] = "11"
    os.environ["SECI_RETRIEVAL__RERANK"] = "true"
    try:
        cfg = AppConfig.from_env()
        assert cfg.llm.provider == "anthropic"
        assert cfg.retrieval.top_k == 11
        assert cfg.retrieval.rerank is True
    finally:
        for k in ["SECI_LLM__PROVIDER", "SECI_RETRIEVAL__TOP_K", "SECI_RETRIEVAL__RERANK"]:
            os.environ.pop(k, None)
