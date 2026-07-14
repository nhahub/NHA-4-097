"""agents/response_agent.py — format, filter, translate"""
from __future__ import annotations
import re
from typing import Any
from memory.memory_module import MemoryModule

_SENSITIVE = [
    re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
]

class ResponseAgent:
    def __init__(self, memory: MemoryModule):
        self.memory = memory

    def format(self, raw_result: dict, original_query: str,
               agent_used: str, target_lang: str = "en",
               memory_context: str = "") -> dict[str, Any]:
        from tools.llm_client import chat_completion

        content = self._filter(raw_result.get("content", ""))
        chart   = raw_result.get("chart")
        sources = raw_result.get("sources", [])

        lang_instruction = (
            "Translate and respond entirely in formal Arabic." if target_lang == "ar"
            else "Respond in English."
        )

        system = (
            "You are a professional BI report writer.\n"
            f"{lang_instruction}\n"
            "Format the raw result into a clear, structured response using markdown.\n"
            "Add a '💡 Recommendation' section if actionable insights exist.\n"
            "Never include raw code in the response.\n\n"
            f"Raw result:\n{content}\n\n"
            f"Agent used: {agent_used}\n"
            f"Original question: {original_query}"
        )
        try:
            formatted = chat_completion(
                [{"role": "user", "content": "Format and finalize this response."}],
                system=system, max_tokens=1000,
            )
        except Exception:
            formatted = content

        if sources:
            formatted += "\n\n---\n**Sources:** " + ", ".join(f"`{s}`" for s in sources)
        if raw_result.get("from_cache"):
            formatted += "\n\n*ℹ️ Served from cache.*"

        return {"answer": formatted, "chart": chart}

    def _filter(self, text: str) -> str:
        for pat in _SENSITIVE:
            text = pat.sub("[REDACTED]", text)
        return text