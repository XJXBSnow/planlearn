"""Chroma-backed retriever over the index built by index_chunks.py.
pip install chromadb sentence-transformers"""
from __future__ import annotations

from ..models import Chunk
from .base import Hit

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class ChromaRetriever:
    def __init__(self, db_path: str, collection: str = "islp",
                 model: str = "BAAI/bge-base-en-v1.5"):
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.col = chromadb.PersistentClient(path=db_path).get_collection(collection)
        self.model = SentenceTransformer(model)

    def search(self, query: str, k: int = 8, labs: bool | None = None) -> list[Hit]:
        q = self.model.encode(BGE_QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        where = {"is_lab": labs} if labs is not None else None
        r = self.col.query(query_embeddings=[q], n_results=k, where=where)
        hits = []
        for cid, doc, meta, dist in zip(r["ids"][0], r["documents"][0],
                                        r["metadatas"][0], r["distances"][0]):
            chunk = Chunk(cid, doc, meta["section"], meta["page_start"], meta["page_end"],
                          meta["is_lab"], len(doc) // 4)
            hits.append(Hit(chunk, 1 - dist))
        return hits
