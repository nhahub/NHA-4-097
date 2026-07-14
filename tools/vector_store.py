"""tools/vector_store.py"""
from __future__ import annotations
import os
import numpy as np
from typing import Any


class VectorStore:
    def __init__(self, collection: str = "default"):
        self.collection_name = collection
        self._backend  = self._init_backend()
        self._embedder = self._init_embedder()

    def _init_backend(self):
        try:
            import chromadb
            try:
                client = chromadb.EphemeralClient()
            except AttributeError:
                client = chromadb.Client()
            col = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return {"type": "chroma", "client": client, "col": col}
        except Exception:
            pass
        try:
            import faiss
            return {"type":"faiss","index":None,"texts":[],"metadatas":[],"ids":[],"dim":None}
        except Exception:
            pass
        return {"type":"simple","vectors":[],"texts":[],"metadatas":[],"ids":[]}

    def _init_embedder(self):
        # Groq doesn't have embeddings — use sentence-transformers or random
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "random"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder == "random":
            return [list(np.random.rand(384).astype(float)) for _ in texts]
        if self._embedder == "openai":
            import openai
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            r = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [d.embedding for d in r.data]
        vecs = self._embedder.encode(texts, convert_to_numpy=True)
        return vecs.tolist()

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]):
        if not texts:
            return
        vectors = self._embed(texts)
        b = self._backend
        if b["type"] == "chroma":
            b["col"].add(ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors)
        elif b["type"] == "faiss":
            self._faiss_add(b, ids, texts, metadatas, vectors)
        else:
            for uid,t,m,v in zip(ids,texts,metadatas,vectors):
                b["ids"].append(uid); b["texts"].append(t)
                b["metadatas"].append(m); b["vectors"].append(v)

    def _faiss_add(self, b, ids, texts, metadatas, vectors):
        import faiss
        dim = len(vectors[0])
        arr = np.array(vectors, dtype=np.float32)
        if b["index"] is None:
            b["dim"]   = dim
            b["index"] = faiss.IndexFlatL2(dim)
        b["index"].add(arr)
        b["texts"].extend(texts); b["metadatas"].extend(metadatas); b["ids"].extend(ids)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        vec = self._embed([query])[0]
        b   = self._backend
        if b["type"] == "chroma":
            n = b["col"].count()
            if n == 0: return []
            r = b["col"].query(query_embeddings=[vec], n_results=min(top_k, n))
            return [{"text":d,"metadata":m}
                    for d,m in zip(r["documents"][0], r["metadatas"][0])]
        elif b["type"] == "faiss":
            return self._faiss_search(b, vec, top_k)
        else:
            return self._cosine_search(b, vec, top_k)

    def _faiss_search(self, b, vec, top_k):
        if b["index"] is None or b["index"].ntotal == 0: return []
        import faiss
        arr = np.array([vec], dtype=np.float32)
        k   = min(top_k, b["index"].ntotal)
        _, idxs = b["index"].search(arr, k)
        return [{"text":b["texts"][i],"metadata":b["metadatas"][i]} for i in idxs[0] if i>=0]

    def _cosine_search(self, b, vec, top_k):
        if not b["vectors"]: return []
        q = np.array(vec)
        scores = []
        for i,v in enumerate(b["vectors"]):
            va = np.array(v)
            s  = float(np.dot(q,va)/(np.linalg.norm(q)*np.linalg.norm(va)+1e-10))
            scores.append((s,i))
        scores.sort(reverse=True)
        return [{"text":b["texts"][i],"metadata":b["metadatas"][i]} for _,i in scores[:top_k]]

    def count(self) -> int:
        b = self._backend
        if b["type"] == "chroma": return b["col"].count()
        return len(b.get("texts",[]))