"""Graph cache: reuse a finished plan when a *similar* request arrives.
Key = goal similarity (token Jaccard) AND same level AND same known topics,
because the same goal yields different graphs for different learners."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_STOP = {"i", "want", "to", "learn", "the", "a", "an", "about", "how", "and", "of", "in", "understand"}


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP}


class GraphCache:
    def __init__(self, path: str | Path | None = None, threshold: float = 0.7):
        self.path, self.threshold = (Path(path) if path else None), threshold
        self.entries: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self.entries = json.loads(self.path.read_text())

    def _key(self, goal: str, level: str, known: list[str]) -> dict[str, Any]:
        return {"tokens": sorted(_tokens(goal)), "level": level,
                "known": sorted(k.lower() for k in known)}

    def get(self, goal: str, level: str, known: list[str]) -> dict[str, Any] | None:
        k = self._key(goal, level, known)
        q = set(k["tokens"])
        for e in self.entries:
            if e["key"]["level"] != k["level"] or e["key"]["known"] != k["known"]:
                continue
            t = set(e["key"]["tokens"])
            if q and t and len(q & t) / len(q | t) >= self.threshold:
                return e["plan"]
        return None

    def put(self, goal: str, level: str, known: list[str], plan: dict[str, Any]) -> None:
        self.entries.append({"key": self._key(goal, level, known), "plan": plan})
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries))
