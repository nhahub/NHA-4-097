"""agents/rag_agent.py"""
from __future__ import annotations
from typing import Any
from tools.vector_store import VectorStore
from memory.memory_module import MemoryModule


class RAGAgent:
    def __init__(self, memory: MemoryModule):
        self.memory = memory
        self.vector_store = VectorStore()

    def add_document(self, name: str, text: str, chunk_size: int = 500, overlap: int = 50):
        chunks = self._chunk(text, chunk_size, overlap)
        ids    = [f"{name}::{i}" for i in range(len(chunks))]
        metas  = [{"source": name, "chunk": i} for i in range(len(chunks))]
        self.vector_store.add(ids, chunks, metas)

    def _chunk(self, text: str, size: int, overlap: int) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + size]))
            i += size - overlap
        return chunks or [text]

    def run(self, query: str, memory_context: str = "") -> dict[str, Any]:
        from tools.llm_client import chat_completion
        results = self.vector_store.search(query, top_k=5)
        if not results:
            return {"content": "No relevant documents found. Please upload related files.", "sources": []}

        context = "\n\n---\n\n".join(r["text"] for r in results)
        sources = list({r["metadata"].get("source","?") for r in results})

        system = (
            "You are a BI assistant. Answer using ONLY the context below.\n"
            f"Context:\n{context}\n\nMemory:\n{memory_context or 'None'}\n"
            "Answer in the same language as the question."
        )
        answer = chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": query}],
            max_tokens=800,
        )
        return {"content": answer, "sources": sources}