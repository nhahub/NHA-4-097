"""
agents/orchestrator.py
"""
from __future__ import annotations
import json
import pandas as pd
from typing import Any

from agents.rag_agent import RAGAgent
from agents.analytics_agent import AnalyticsAgent
from agents.web_search_agent import WebSearchAgent
from agents.response_agent import ResponseAgent
from memory.memory_module import MemoryModule
from utils.guardrails import check_injection


class OrchestratorAgent:
    def __init__(self, memory: MemoryModule):
        try:
            self.memory = memory
            self.rag = RAGAgent(memory)
            self.analytics = AnalyticsAgent(memory)
            self.web = WebSearchAgent(memory)
            self.responder = ResponseAgent(memory)
            self._dataframes: dict[str, pd.DataFrame] = {}
        except Exception as e:
            raise RuntimeError(f"خطأ في تهيئة Orchestrator: {e}")

    def ingest_file(self, file_obj) -> dict:
        name = file_obj.name
        ext  = name.rsplit(".", 1)[-1].lower()
        result: dict[str, Any] = {"name": name, "type": ext}

        if ext == "csv":
            df = pd.read_csv(file_obj)
            self._dataframes[name] = df
            self.analytics.register_dataframe(name, df)
            result["dataframe"] = df
            result["rows"] = len(df)
            self.memory.add_entity(name, f"File '{name}': columns={list(df.columns)}, rows={len(df)}")

        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_obj)
            self._dataframes[name] = df
            self.analytics.register_dataframe(name, df)
            result["dataframe"] = df
            result["rows"] = len(df)
            self.memory.add_entity(name, f"File '{name}': columns={list(df.columns)}, rows={len(df)}")

        elif ext in ("pdf", "txt", "docx"):
            text = self._extract_text(file_obj, ext)
            self.rag.add_document(name, text)
            result["text_preview"] = text[:300]

        return result

    def _extract_text(self, file_obj, ext: str) -> str:
        if ext == "txt":
            return file_obj.read().decode("utf-8", errors="ignore")
        if ext == "pdf":
            try:
                import pdfplumber
                with pdfplumber.open(file_obj) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except Exception:
                return ""
        if ext == "docx":
            try:
                from docx import Document
                return "\n".join(p.text for p in Document(file_obj).paragraphs)
            except Exception:
                return ""
        return ""

    def run(self, query: str, target_lang: str = "en",
            uploaded_files: list[str] | None = None) -> dict:

        safe, warning = check_injection(query)
        if not safe:
            return {"answer": f"⚠️ {warning}", "agent": "guardrails", "chart": None}

        self.memory.add_short_term("user", query)
        memory_context = self.memory.retrieve_relevant(query)
        agent_choice   = self._route(query, uploaded_files or [])

        if agent_choice == "rag":
            raw = self.rag.run(query, memory_context)
        elif agent_choice == "analytics":
            raw = self.analytics.run(query, memory_context)
        elif agent_choice == "web":
            raw = self.web.run(query, memory_context)
        elif agent_choice == "combined":
            r1  = self.rag.run(query, memory_context)
            r2  = self.analytics.run(query, memory_context)
            raw = {"content": r1.get("content","") + "\n\n" + r2.get("content",""),
                   "chart": r2.get("chart"), "sources": r1.get("sources",[])}
        else:
            raw = {"content": "No agent matched. Please try again."}

        final = self.responder.format(
            raw_result=raw, original_query=query,
            agent_used=agent_choice, target_lang=target_lang,
            memory_context=memory_context,
        )

        self.memory.add_short_term("assistant", final["answer"])
        self.memory.store_insight(query, final["answer"])
        return {**final, "agent": agent_choice}

    def _route(self, query: str, files: list[str]) -> str:
        has_tabular = any(f.endswith((".csv", ".xlsx", ".xls")) for f in files)
        has_docs    = any(f.endswith((".pdf", ".txt", ".docx")) for f in files)
        files_str   = ", ".join(files) if files else "none"

        try:
            from tools.llm_client import chat_completion
            prompt = (
                "You are a routing agent. Choose the best agent for this query.\n"
                "Agents: analytics (CSV/Excel data), rag (PDF/Word docs), "
                "web (live info/news), combined (both docs+data)\n"
                "Reply ONLY with valid JSON like: {\"agent\": \"analytics\", \"reason\": \"...\"}\n\n"
                f"Query: {query}\nFiles: {files_str}"
            )
            resp  = chat_completion([{"role": "user", "content": prompt}], max_tokens=60)
            resp  = resp.strip().strip("`").replace("json","",1).strip()
            agent = json.loads(resp).get("agent", "")
            if agent not in ("analytics", "rag", "web", "combined"):
                raise ValueError()
            return agent
        except Exception:
            ql = query.lower()
            if any(k in ql for k in ["رسم","حلل","مقارن","متوسط","مجموع","أعلى","أقل","تريند","إجمالي",
                                      "chart","plot","graph","average","sum","count","trend","top",
                                      "compare","highest","lowest","total","revenue","profit","show"]):
                return "analytics"
            if any(k in ql for k in ["سعر","اخبار","اليوم","الان",
                                      "price","news","today","latest","market","current"]):
                return "web"
            if has_tabular and has_docs: return "combined"
            if has_tabular:  return "analytics"
            if has_docs:     return "rag"
            return "analytics"