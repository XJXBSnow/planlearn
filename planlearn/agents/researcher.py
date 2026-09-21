"""Topic Researcher: brings in fresh, external signal the static RAG/KG can't.

Used sparingly (ties, final recommendations). Findings are cached and shared
across users, since trends for "lasso regression" don't depend on who asks."""
from __future__ import annotations

from typing import Protocol

from ..models import AgentResult, LearningNode, NextAction


class WebSearch(Protocol):
    def search(self, query: str, k: int = 5) -> list[dict]: ...  # [{"title","url","snippet"}]


class NullSearch:
    """Offline default: no external signal, neutral trend scores."""
    def search(self, query: str, k: int = 5) -> list[dict]:
        return []


class ResearcherAgent:
    name = "researcher"

    def __init__(self, web: WebSearch | None = None):
        self.web = web or NullSearch()
        self.shared_cache: dict[str, list[dict]] = {}   # shared across users

    def _lookup(self, topic: str) -> list[dict]:
        if topic not in self.shared_cache:
            self.shared_cache[topic] = self.web.search(
                f"{topic} statistical learning tutorial OR survey 2026", k=5)
        return self.shared_cache[topic]

    def trends(self, nodes: list[LearningNode]) -> AgentResult:
        scores = {}
        for n in nodes:
            results = self._lookup(n.title.split(" [")[0])
            # crude popularity proxy: result count; replace with citation/recency signals
            scores[n.id] = min(1.0, 0.5 + 0.1 * len(results))
        signal = any(s != 0.5 for s in scores.values())
        return AgentResult(self.name, scores, NextAction.CONTINUE,
                           "external signal found" if signal else "no external signal (offline)")

    def recommend(self, leaves: list[LearningNode], k: int = 3) -> AgentResult:
        recs = []
        for n in leaves[:k]:
            for r in self._lookup(n.title.split(" [")[0])[:1]:
                recs.append({"for": n.id, **r})
        return AgentResult(self.name, recs, NextAction.DONE)
