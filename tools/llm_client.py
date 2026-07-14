"""
tools/llm_client.py — supports Groq, OpenAI, Anthropic
Priority: GROQ_API_KEY → OPENAI_API_KEY → ANTHROPIC_API_KEY
"""
from __future__ import annotations
import os
from typing import Any


def chat_completion(
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.3,
    model: str | None = None,
) -> str:
    provider = _detect_provider()
    if provider == "groq":
        return _groq_call(messages, system, max_tokens, temperature, model)
    elif provider == "openai":
        return _openai_call(messages, system, max_tokens, temperature, model)
    elif provider == "anthropic":
        return _anthropic_call(messages, system, max_tokens, temperature, model)
    else:
        raise RuntimeError(
            "No API key found. Add GROQ_API_KEY=... to your .env file."
        )


def _detect_provider() -> str:
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def _groq_call(messages, system, max_tokens, temperature, model) -> str:
    from groq import Groq
    client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
    model    = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    all_msgs = []
    if system:
        all_msgs.append({"role": "system", "content": system})
    all_msgs.extend(messages)
    r = client.chat.completions.create(
        model=model, messages=all_msgs,
        max_tokens=max_tokens, temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def _openai_call(messages, system, max_tokens, temperature, model) -> str:
    import openai
    client   = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model    = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    all_msgs = []
    if system:
        all_msgs.append({"role": "system", "content": system})
    all_msgs.extend(messages)
    r = client.chat.completions.create(
        model=model, messages=all_msgs,
        max_tokens=max_tokens, temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def _anthropic_call(messages, system, max_tokens, temperature, model) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model  = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens, "messages": messages,
    }
    if system:
        kwargs["system"] = system
    r = client.messages.create(**kwargs)
    return r.content[0].text.strip()


# ── Backward-compat class ─────────────────────────────────────────────────────
class LLMClient:
    def chat(self, messages, system=None, max_tokens=1000, temperature=0.3, model=None) -> str:
        return chat_completion(messages, system, max_tokens, temperature, model)
    def chat_completion(self, *a, **kw) -> str:
        return self.chat(*a, **kw)