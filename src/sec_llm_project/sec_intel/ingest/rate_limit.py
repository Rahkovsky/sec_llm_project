"""SEC EDGAR fair-access rate limiting.

SEC's fair-access policy caps automated traffic at 10 requests/second and
requires a descriptive User-Agent identifying the requester. This module
provides a conservative token-bucket limiter (default 8 req/s, below the cap)
and a helper to assemble a compliant User-Agent. Using these makes ingestion a
good citizen of official SEC data sources.
"""

from __future__ import annotations

import threading
import time

SEC_MAX_RPS = 10.0  # SEC fair-access hard cap
DEFAULT_RPS = 8.0   # stay safely under the cap


class RateLimiter:
    """Thread-safe token bucket enforcing a maximum request rate."""

    def __init__(self, rate_per_sec: float = DEFAULT_RPS, burst: int = 1) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        if rate_per_sec > SEC_MAX_RPS:
            raise ValueError(
                f"rate_per_sec={rate_per_sec} exceeds SEC fair-access cap of {SEC_MAX_RPS}/s"
            )
        self.rate = rate_per_sec
        self.capacity = float(max(1, burst))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a request token is available."""
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                sleep_for = (1.0 - self._tokens) / self.rate
                time.sleep(sleep_for)


def build_user_agent(name: str, email: str, app: str = "SEC-Disclosure-Intelligence-Prototype") -> str:
    """Compose a SEC-compliant User-Agent: 'App Name email'."""
    name = (name or "Anonymous").strip()
    email = (email or "anonymous@example.com").strip()
    return f"{app} {name} {email}"
