"""agents/analytics_agent.py"""
from __future__ import annotations
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any
from memory.memory_module import MemoryModule


class AnalyticsAgent:
    def __init__(self, memory: MemoryModule):
        self.memory = memory
        self._dataframes: dict[str, pd.DataFrame] = {}

    def register_dataframe(self, name: str, df: pd.DataFrame):
        self._dataframes[name] = df

    def run(self, query: str, memory_context: str = "") -> dict[str, Any]:
        from tools.llm_client import chat_completion
        if not self._dataframes:
            return {"content": "No tabular data loaded. Please upload a CSV or Excel file.", "chart": None}

        schemas = "\n".join(
            f"- {n}: columns={list(df.columns)}, rows={len(df)}"
            for n, df in self._dataframes.items()
        )
        df_sample = next(iter(self._dataframes.values())).head(3).to_string(index=False)

        system = (
            "You are a data analyst. Write Python/Pandas code to answer the user's question.\n"
            f"Available DataFrames (names below are labels only, NOT file paths):\n{schemas}\n\n"
            f"Sample data:\n{df_sample}\n\n"
            f"Memory:\n{memory_context or 'None'}\n\n"
            "STEP 1 — Relevance check:\n"
            "Look at the available columns and sample data above. If the user's question "
            "CANNOT be answered from this data (e.g. it's a general knowledge question, "
            "small talk, or about a topic/column that does not exist in the data), "
            "respond with ONLY this exact line and nothing else:\n"
            "OUT_OF_SCOPE: <one short sentence in the same language as the question, "
            "explaining the data does not cover this and listing the available columns>\n\n"
            "STEP 2 — If the question IS answerable from the data, write code:\n"
            "Rules:\n"
            "- The data is ALREADY loaded in memory as the variable `df`. Do NOT call "
            "pd.read_csv, pd.read_excel, open(), or any file-loading function — the file "
            "does not exist on disk, only `df` exists.\n"
            "- Use variable `df` for the main DataFrame\n"
            "- Store the answer string in `result`\n"
            "- If a chart helps, create a Plotly figure in `fig` using plotly.express as px\n"
            "- Do NOT use print()\n"
            "- Return ONLY Python code inside ```python ... ``` block"
        )
        raw = chat_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": query}],
            max_tokens=700,
        )
        if raw.strip().startswith("OUT_OF_SCOPE"):
            msg = raw.strip().split(":", 1)[-1].strip() or \
                  "This question is outside the scope of the uploaded data."
            return {"content": f"⚠️ {msg}", "chart": None}

        code = self._extract_code(raw)
        if not code:
            return {"content": raw, "chart": None}
        return self._execute(code)

    def _extract_code(self, text: str) -> str:
        m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        if "df" in text or "result" in text:
            return text.strip()
        return ""

    def _execute(self, code: str) -> dict[str, Any]:
        if not self._dataframes:
            return {"content": "No data available.", "chart": None}

        df = max(self._dataframes.values(), key=len)
        local_vars = {"df": df, "pd": pd, "px": px, "go": go, "result": None, "fig": None}

        forbidden = ["import os", "import sys", "open(", "__import__", "subprocess",
                     "read_csv(", "read_excel(", "pd.read_"]
        for f in forbidden:
            if f in code:
                return {"content": f"⚠️ Blocked: forbidden operation '{f}'. "
                                    "The data is already loaded in `df`; no file access is needed.",
                        "chart": None}
        try:
            exec(code, local_vars)
        except Exception as e:
            return {"content": f"Analysis error: {e}", "chart": None}

        result = local_vars.get("result")
        fig    = local_vars.get("fig")
        text   = str(result) if result is not None else ("Chart generated." if fig else "Done.")
        return {"content": text, "chart": fig}