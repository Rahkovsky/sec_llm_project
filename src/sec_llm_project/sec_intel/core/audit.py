"""Audit logging and provenance (plan item 10).

Every retrieval and generation event can be appended to a JSONL audit trail so
that answers are reproducible and inspectable after the fact. The logger is
deliberately tiny and dependency-free; it degrades to a no-op when disabled.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class AuditLogger:
    enabled: bool = True
    path: str = "logs/audit.jsonl"

    def log(self, event: str, payload: dict[str, Any]) -> str:
        """Append an event; returns a unique event id (empty string if disabled)."""
        if not self.enabled:
            return ""
        event_id = uuid.uuid4().hex
        record = {
            "event_id": event_id,
            "event": event,
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "payload": payload,
        }
        p = Path(self.path)
        if p.parent and not p.parent.exists():
            os.makedirs(p.parent, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return event_id

    @classmethod
    def from_config(cls, audit_config: Any) -> AuditLogger:
        return cls(
            enabled=bool(getattr(audit_config, "enabled", True)),
            path=str(getattr(audit_config, "path", "logs/audit.jsonl")),
        )


_NULL = AuditLogger(enabled=False)


def null_logger() -> AuditLogger:
    return _NULL
