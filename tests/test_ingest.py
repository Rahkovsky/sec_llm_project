from __future__ import annotations

import time

import pytest

from sec_llm_project.sec_intel.ingest.rate_limit import (
    SEC_MAX_RPS,
    RateLimiter,
    build_user_agent,
)


def test_rate_limiter_enforces_rate():
    limiter = RateLimiter(rate_per_sec=20.0 if False else 10.0)  # at cap is allowed
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    elapsed = time.monotonic() - start
    # 5 tokens at 10/s with burst 1 => at least ~0.4s of waiting.
    assert elapsed >= 0.3


def test_rate_limiter_rejects_above_cap():
    with pytest.raises(ValueError):
        RateLimiter(rate_per_sec=SEC_MAX_RPS + 1)


def test_user_agent_format():
    ua = build_user_agent("Jane Analyst", "jane@example.gov")
    assert "jane@example.gov" in ua
    assert "Jane Analyst" in ua
