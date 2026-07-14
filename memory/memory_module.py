"""memory/memory_module.py"""
from __future__ import annotations
import time
from collections import deque
from tools.vector_store import VectorStore


class MemoryModule:
    SHORT_TERM_LIMIT = 20

    def __init__(self):
        self._short_term: deque = deque(maxlen=self.SHORT_TERM_LIMIT)
        self._long_term_store = VectorStore(collection="long_term_memory")
        self._lt_counter = 0
        self._entities: dict[str, str] = {}
        self._recent_insights: list[str] = []

    def add_short_term(self, role: str, content: str):
        self._short_term.append({"role": role, "content": content, "ts": time.time()})

    def get_short_term_messages(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self._short_term]

    def short_term_count(self) -> int:
        return len(self._short_term)

    def get_short_term_text(self) -> str:
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self._short_term)

    def store_insight(self, query: str, answer: str):
        if len(answer) < 30:
            return
        text = f"Q: {query[:200]}\nA: {answer[:500]}"
        self._long_term_store.add(
            ids=[f"lt_{self._lt_counter}"],
            texts=[text],
            metadatas=[{"type": "qa", "ts": time.time()}],
        )
        self._lt_counter += 1
        short = f"Previously: {query[:80]}..." if len(query) > 80 else f"Previously: {query}"
        self._recent_insights.append(short)
        if len(self._recent_insights) > 20:
            self._recent_insights.pop(0)

    def retrieve_relevant(self, query: str, top_k: int = 3) -> str:
        results = self._long_term_store.search(query, top_k=top_k)
        if not results:
            return ""
        return "Relevant past context:\n" + "\n---\n".join(r["text"] for r in results)

    def long_term_count(self) -> int:
        return self._lt_counter

    def get_recent_insights(self, n: int = 5) -> list[str]:
        return self._recent_insights[-n:]

    def add_entity(self, name: str, fact: str):
        self._entities[name] = fact

    def get_entity(self, name: str) -> str | None:
        return self._entities.get(name)

    def entity_count(self) -> int:
        return len(self._entities)

    def get_all_entities_text(self) -> str:
        if not self._entities:
            return ""
        return "Known entities:\n" + "\n".join(f"- {k}: {v}" for k,v in self._entities.items())

    def build_context(self, query: str) -> str:
        parts = []
        st = self.get_short_term_text()
        if st:  parts.append("=== Recent Conversation ===\n" + st)
        lt = self.retrieve_relevant(query)
        if lt:  parts.append("=== Long-term Memory ===\n" + lt)
        ent = self.get_all_entities_text()
        if ent: parts.append("=== Known Data Sources ===\n" + ent)
        return "\n\n".join(parts)

    def clear(self):
        self._short_term.clear()
        self._entities.clear()
        self._recent_insights.clear()