"""agents/web_search_agent.py — CAG web search with cache"""
from __future__ import annotations
import hashlib
from typing import Any
from memory.memory_module import MemoryModule
from tools.cache import Cache


class WebSearchAgent:
    def __init__(self, memory: MemoryModule, cache_ttl: int = 3600):
        self.memory = memory
        self.cache  = Cache(ttl=cache_ttl)

    def run(self, query: str, memory_context: str = "") -> dict[str, Any]:
        from tools.llm_client import chat_completion
        key    = "web:" + hashlib.md5(query.lower().strip().encode()).hexdigest()
        cached = self.cache.get(key)
        if cached:
            return {"content": cached, "from_cache": True}

        results = self._search(query)
        system  = (
            "You are a BI assistant with live search results.\n"
            f"Search results:\n{results}\n\nMemory:\n{memory_context or 'None'}\n"
            "Answer in the same language as the question."
        )
        answer = chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": query}],
            max_tokens=700,
        )
        self.cache.set(key, answer)
        return {"content": answer, "from_cache": False}

    def _search(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            return "\n\n".join(
                f"**{r.get('title','')}**\n{r.get('body','')}\nSource: {r.get('href','')}"
                for r in results
            )
        except ImportError:
            return "Web search unavailable. Run: pip install duckduckgo-search"
        except Exception as e:
            return f"Search error: {e}"