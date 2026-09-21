"""Append-only JSONL audit trail: every state transition and agent call."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **event: Any) -> None:
        event = {"ts": round(time.time(), 3), **event}
        self.events.append(event)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
