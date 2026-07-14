"""
tools/sql_engine.py
Natural Language → SQL → Result using DuckDB on pandas DataFrames.
"""

from __future__ import annotations
import re
import pandas as pd
from typing import Any


NL_TO_SQL_PROMPT = """You are a SQL expert. Convert the user's question to a DuckDB SQL query.
The table name is always `df`.

Table schema:
{schema}

Sample data (first 3 rows):
{sample}

User question: {question}

Return ONLY the SQL query, no explanation, no markdown backticks.
"""


class SQLEngine:
    def __init__(self):
        self._dataframes: dict[str, pd.DataFrame] = {}

    def register(self, name: str, df: pd.DataFrame):
        self._dataframes[name] = df

    def nl_to_result(self, question: str, df_name: str | None = None) -> dict[str, Any]:
        """Convert natural language question to SQL and execute it."""
        try:
            import duckdb
        except ImportError:
            return {"error": "DuckDB not installed. Run: pip install duckdb"}

        # Pick dataframe
        if df_name and df_name in self._dataframes:
            df = self._dataframes[df_name]
        elif self._dataframes:
            df = next(iter(self._dataframes.values()))
        else:
            return {"error": "No dataframe registered"}

        schema = self._get_schema(df)
        sample = df.head(3).to_string(index=False)

        # Generate SQL
        sql = self._generate_sql(question, schema, sample)
        if not sql:
            return {"error": "Could not generate SQL"}

        # Execute
        try:
            conn = duckdb.connect()
            conn.register("df", df)
            result_df = conn.execute(sql).df()
            return {
                "sql": sql,
                "result": result_df,
                "rows": len(result_df),
            }
        except Exception as e:
            return {"error": f"SQL execution error: {e}", "sql": sql}

    def _generate_sql(self, question: str, schema: str, sample: str) -> str:
        from tools.llm_client import chat_completion
        prompt = NL_TO_SQL_PROMPT.format(
            schema=schema, sample=sample, question=question
        )
        sql = chat_completion([{"role": "user", "content": prompt}], max_tokens=200)
        # Clean up
        sql = re.sub(r"```sql|```", "", sql).strip()
        sql = sql.split(";")[0] + ";"   # single statement
        return sql

    def _get_schema(self, df: pd.DataFrame) -> str:
        lines = [f"  {col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)]
        return "df (\n" + "\n".join(lines) + "\n)"

    def execute_sql(self, sql: str, df_name: str | None = None) -> pd.DataFrame | str:
        """Execute raw SQL against a registered dataframe."""
        try:
            import duckdb
        except ImportError:
            return "DuckDB not installed."

        df = None
        if df_name and df_name in self._dataframes:
            df = self._dataframes[df_name]
        elif self._dataframes:
            df = next(iter(self._dataframes.values()))

        if df is None:
            return "No dataframe available."

        try:
            conn = duckdb.connect()
            conn.register("df", df)
            return conn.execute(sql).df()
        except Exception as e:
            return f"SQL error: {e}"