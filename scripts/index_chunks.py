"""
Embed islp_chunks.jsonl into a local Chroma store and run a test query.
pip install chromadb sentence-transformers
Usage: python index_chunks.py islp_chunks.jsonl ./islp_db "what is the bias-variance tradeoff?"
"""
import json, sys
import chromadb
from sentence_transformers import SentenceTransformer

src, db_path = sys.argv[1], sys.argv[2]
query = sys.argv[3] if len(sys.argv) > 3 else None

MODEL = "BAAI/bge-base-en-v1.5"          # swap for any embedding model you prefer
model = SentenceTransformer(MODEL)
client = chromadb.PersistentClient(path=db_path)
col = client.get_or_create_collection("islp", metadata={"hnsw:space": "cosine"})

if col.count() == 0:
    rows = [json.loads(l) for l in open(src)]
    for i in range(0, len(rows), 64):
        b = rows[i:i + 64]
        col.add(
            ids=[r["id"] for r in b],
            documents=[r["text"] for r in b],
            # embed_text = "Chapter > Section > Subsection" + body, better retrieval
            embeddings=model.encode([r["embed_text"] for r in b],
                                    normalize_embeddings=True).tolist(),
            metadatas=[{"section": r["section"], "page_start": r["page_start"],
                        "page_end": r["page_end"], "is_lab": r["is_lab"]} for r in b],
        )
    print(f"indexed {col.count()} chunks")

if query:
    # bge models expect this instruction prefix on queries (not on documents)
    q = model.encode("Represent this sentence for searching relevant passages: " + query,
                     normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[q], n_results=5)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"\n[{1 - dist:.3f}] {meta['section']} (pp. {meta['page_start']}-{meta['page_end']})")
        print(doc[:300], "...")
