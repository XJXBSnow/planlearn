"""The one seam every agent uses for RAG. Swap implementations freely:
TF-IDF (offline dev) -> local Chroma -> pgvector/Qdrant (multi-user)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..models import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float  # higher = more relevant, roughly 0..1


class Retriever(Protocol):
    def search(self, query: str, k: int = 8, labs: bool | None = None) -> list[Hit]: ...
