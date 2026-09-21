"""Dependency-light retriever for offline development and tests (scikit-learn only)."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from ..models import Chunk
from .base import Hit


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=2)
        # heading context is part of what we index, same as embed_text
        self.matrix = self.vec.fit_transform([f"{c.section}\n{c.text}" for c in chunks])

    def search(self, query: str, k: int = 8, labs: bool | None = None) -> list[Hit]:
        sims = linear_kernel(self.vec.transform([query]), self.matrix).ravel()
        order = sims.argsort()[::-1]
        hits = []
        for i in order:
            c = self.chunks[i]
            if labs is not None and c.is_lab != labs:
                continue
            if sims[i] <= 0:
                break
            hits.append(Hit(c, float(sims[i])))
            if len(hits) == k:
                break
        return hits
