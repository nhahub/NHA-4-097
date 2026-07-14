"""
tools/auto_dashboard.py
Automatically generates a full dashboard from any uploaded DataFrame.
Detects column types and builds the most relevant charts automatically.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any


class AutoDashboard:
    """
    Given any DataFrame, auto-detects column types and generates
    the most relevant charts and KPI cards.
    """

    def __init__(self, df: pd.DataFrame, name: str = "Data"):
        self.df = df.copy()
        self.name = name
        self._clean()
        self._detect_columns()

    # ── Data cleaning ─────────────────────────────────────────────────────────
    def _clean(self):
        # Try to parse date-like columns
        for col in self.df.columns:
            if self.df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(self.df[col], infer_format=True, errors="coerce")
                    if parsed.notna().sum() > len(self.df) * 0.6:
                        self.df[col] = parsed
                except Exception:
                    pass
        # Strip whitespace from string columns
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = self.df[col].astype(str).str.strip()

    # ── Column type detection ─────────────────────────────────────────────────
    def _detect_columns(self):
        n_rows = len(self.df)

        # ID-like patterns to skip
        id_patterns = ("id", "order", "invoice", "code", "ref", "no", "num", "key", "uuid", "index")

        def is_id_col(col):
            cl = col.lower()
            if any(cl == p or cl.endswith("_" + p) or cl.startswith(p + "_") for p in id_patterns):
                return True
            # Nearly all unique → ID
            if self.df[col].nunique() > n_rows * 0.9:
                return True
            return False

        # Numeric: skip columns that look like IDs (all integers 1..N with no repeats)
        all_num = self.df.select_dtypes(include="number").columns.tolist()
        self.num_cols = [
            c for c in all_num
            if not is_id_col(c)
        ]

        # Categorical: low-cardinality, not ID-like
        self.cat_cols = [
            c for c in self.df.select_dtypes(include="object").columns
            if self.df[c].nunique() <= 50 and not is_id_col(c)
        ]

        self.date_cols = self.df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        # High-cardinality text (e.g. customer names, product names) — use for top-N charts
        self.high_card_cols = [
            c for c in self.df.select_dtypes(include="object").columns
            if 50 < self.df[c].nunique() <= n_rows * 0.8 and not is_id_col(c)
        ]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    def get_kpis(self) -> list[dict]:
        kpis = []
        kpis.append({"label": "Total Rows", "value": f"{len(self.df):,}", "delta": None})
        kpis.append({"label": "Columns",    "value": str(len(self.df.columns)), "delta": None})

        # Priority: revenue/sales/profit cols first, then others
        priority = ("revenue", "sales", "profit", "income", "amount", "total", "price", "margin")
        skip     = ("quantity", "qty", "count", "discount", "returned", "rating", "cost", "id")

        def score(c):
            cl = c.lower()
            if any(p in cl for p in priority): return 0
            if any(p in cl for p in skip):     return 2
            return 1

        sorted_cols = sorted(self.num_cols, key=score)

        for col in sorted_cols[:4]:
            total = self.df[col].sum()
            avg   = self.df[col].mean()
            label = col.replace("_", " ").title()
            # For margin/percentage cols show avg not sum
            if any(p in col.lower() for p in ("margin", "rate", "pct", "percent", "rating", "%")):
                kpis.append({
                    "label": f"Avg {label}",
                    "value": f"{avg:.1f}%",
                    "delta": None,
                })
            else:
                kpis.append({
                    "label": f"Total {label}",
                    "value": self._fmt(total),
                    "delta": f"avg {self._fmt(avg)}",
                })
        return kpis

    def _fmt(self, v: float) -> str:
        if abs(v) >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"{v/1_000:.1f}K"
        return f"{v:,.1f}"

    # ── Chart generation ──────────────────────────────────────────────────────
    def generate_all_charts(self) -> list[dict]:
        """Return list of {title, fig} dicts for every relevant chart."""
        charts = []

        # 1. Time series for each numeric col
        if self.date_cols and self.num_cols:
            charts += self._time_series_charts()

        # 2. Bar charts: category × numeric
        if self.cat_cols and self.num_cols:
            charts += self._bar_charts()

        # 3. Distribution histograms for numeric cols
        if self.num_cols:
            charts += self._histogram_charts()

        # 4. Correlation heatmap
        if len(self.num_cols) >= 3:
            charts.append(self._correlation_heatmap())

        # 5. Pie / donut for low-cardinality categoricals
        if self.cat_cols:
            charts += self._pie_charts()

        # 6. Box plots
        if self.cat_cols and self.num_cols:
            charts += self._box_plots()

        # 7. Scatter matrix if many numerics
        if len(self.num_cols) >= 3:
            charts.append(self._scatter_matrix())

        # 8. Top-N tables for high-cardinality cols
        if self.high_card_cols and self.num_cols:
            charts += self._top_n_charts()

        # 9. Missing values chart
        missing = self._missing_chart()
        if missing:
            charts.append(missing)

        return [c for c in charts if c is not None]

    # ─── Time series ──────────────────────────────────────────────────────────
    def _time_series_charts(self) -> list[dict]:
        charts = []
        date_col = self.date_cols[0]
        df_sorted = self.df.sort_values(date_col)

        for num_col in self.num_cols[:3]:
            # Try to group by month
            try:
                ts = (
                    df_sorted.set_index(date_col)[num_col]
                    .resample("ME").sum()
                    .reset_index()
                )
                ts.columns = [date_col, num_col]
            except Exception:
                ts = df_sorted[[date_col, num_col]].dropna()

            fig = px.line(
                ts, x=date_col, y=num_col,
                title=f"{num_col.replace('_',' ').title()} Over Time",
                markers=True,
                color_discrete_sequence=["#6366f1"],
            )
            fig.update_layout(**self._layout())
            fig.update_traces(line_width=2, marker_size=5)
            charts.append({"title": f"{num_col} over time", "fig": fig, "type": "timeseries"})

        return charts

    # ─── Bar charts ───────────────────────────────────────────────────────────
    def _bar_charts(self) -> list[dict]:
        charts = []
        colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

        for i, cat_col in enumerate(self.cat_cols[:2]):
            for j, num_col in enumerate(self.num_cols[:2]):
                agg = (
                    self.df.groupby(cat_col)[num_col]
                    .sum()
                    .reset_index()
                    .sort_values(num_col, ascending=False)
                    .head(15)
                )
                fig = px.bar(
                    agg, x=cat_col, y=num_col,
                    title=f"{num_col.replace('_',' ').title()} by {cat_col.replace('_',' ').title()}",
                    color_discrete_sequence=[colors[(i + j) % len(colors)]],
                    text_auto=".2s",
                )
                fig.update_layout(**self._layout())
                fig.update_traces(textposition="outside")
                charts.append({"title": f"{num_col} by {cat_col}", "fig": fig, "type": "bar"})

        return charts

    # ─── Histograms ───────────────────────────────────────────────────────────
    def _histogram_charts(self) -> list[dict]:
        charts = []
        for num_col in self.num_cols[:3]:
            fig = px.histogram(
                self.df, x=num_col,
                title=f"Distribution of {num_col.replace('_',' ').title()}",
                nbins=30,
                color_discrete_sequence=["#818cf8"],
                marginal="box",
            )
            fig.update_layout(**self._layout())
            charts.append({"title": f"{num_col} distribution", "fig": fig, "type": "histogram"})
        return charts

    # ─── Correlation heatmap ──────────────────────────────────────────────────
    def _correlation_heatmap(self) -> dict | None:
        try:
            corr = self.df[self.num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
                zmin=-1, zmax=1,
            )
            fig.update_layout(**self._layout())
            return {"title": "Correlation heatmap", "fig": fig, "type": "heatmap"}
        except Exception:
            return None

    # ─── Pie charts ───────────────────────────────────────────────────────────
    def _pie_charts(self) -> list[dict]:
        charts = []
        for cat_col in self.cat_cols[:2]:
            if self.num_cols:
                agg = self.df.groupby(cat_col)[self.num_cols[0]].sum().reset_index()
                values_col = self.num_cols[0]
            else:
                agg = self.df[cat_col].value_counts().reset_index()
                agg.columns = [cat_col, "count"]
                values_col = "count"

            agg = agg.sort_values(values_col, ascending=False).head(10)
            fig = px.pie(
                agg, names=cat_col, values=values_col,
                title=f"Share by {cat_col.replace('_',' ').title()}",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(**self._layout())
            charts.append({"title": f"Share by {cat_col}", "fig": fig, "type": "pie"})
        return charts

    # ─── Box plots ────────────────────────────────────────────────────────────
    def _box_plots(self) -> list[dict]:
        charts = []
        cat_col = self.cat_cols[0]
        for num_col in self.num_cols[:2]:
            top_cats = self.df[cat_col].value_counts().head(10).index
            df_filtered = self.df[self.df[cat_col].isin(top_cats)]
            fig = px.box(
                df_filtered, x=cat_col, y=num_col,
                title=f"{num_col.replace('_',' ').title()} Distribution by {cat_col.replace('_',' ').title()}",
                color=cat_col,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_layout(**self._layout(), showlegend=False)
            charts.append({"title": f"{num_col} boxplot by {cat_col}", "fig": fig, "type": "box"})
        return charts

    # ─── Scatter matrix ───────────────────────────────────────────────────────
    def _scatter_matrix(self) -> dict | None:
        try:
            cols = self.num_cols[:5]
            kwargs = dict(dimensions=cols, title="Scatter Matrix")
            if self.cat_cols:
                kwargs["color"] = self.cat_cols[0]
                kwargs["color_discrete_sequence"] = px.colors.qualitative.Set2
            fig = px.scatter_matrix(self.df.dropna(subset=cols), **kwargs)
            fig.update_traces(diagonal_visible=False, marker_size=3)
            fig.update_layout(**self._layout(height=600))
            return {"title": "Scatter matrix", "fig": fig, "type": "scatter_matrix"}
        except Exception:
            return None

    # ─── Top-N for high cardinality ───────────────────────────────────────────
    def _top_n_charts(self) -> list[dict]:
        charts = []
        for col in self.high_card_cols[:1]:
            for num_col in self.num_cols[:1]:
                top = (
                    self.df.groupby(col)[num_col]
                    .sum().reset_index()
                    .sort_values(num_col, ascending=True)
                    .tail(15)
                )
                fig = px.bar(
                    top, y=col, x=num_col, orientation="h",
                    title=f"Top 15 {col.replace('_',' ').title()} by {num_col.replace('_',' ').title()}",
                    color_discrete_sequence=["#6366f1"],
                    text_auto=".2s",
                )
                fig.update_layout(**self._layout())
                charts.append({"title": f"Top 15 {col}", "fig": fig, "type": "top_n"})
        return charts

    # ─── Missing values ───────────────────────────────────────────────────────
    def _missing_chart(self) -> dict | None:
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return None
        pct = (missing / len(self.df) * 100).round(1)
        fig = px.bar(
            x=pct.index, y=pct.values,
            title="Missing Values (%)",
            labels={"x": "Column", "y": "Missing %"},
            color=pct.values,
            color_continuous_scale="Reds",
            text_auto=True,
        )
        fig.update_layout(**self._layout())
        return {"title": "Missing values", "fig": fig, "type": "missing"}

    # ─── Layout helper ────────────────────────────────────────────────────────
    def _layout(self, height: int = 380) -> dict:
        return dict(
            height=height,
            margin=dict(l=40, r=20, t=50, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Inter, sans-serif",
            title_font_size=14,
            title_font_color="#6366f1",
            xaxis=dict(gridcolor="rgba(0,0,0,0.05)", showline=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0.05)", showline=False),
        )

    # ── Summary stats ─────────────────────────────────────────────────────────
    def get_summary_stats(self) -> pd.DataFrame:
        if not self.num_cols:
            return pd.DataFrame()
        stats = self.df[self.num_cols].describe().T.round(2)
        stats["missing"] = self.df[self.num_cols].isnull().sum()
        stats["missing_%"] = (stats["missing"] / len(self.df) * 100).round(1)
        return stats

    # ── AI narrative ──────────────────────────────────────────────────────────
    def get_ai_narrative(self) -> str:
        """Ask the LLM for a natural-language summary of the data."""
        try:
            from tools.llm_client import chat_completion
            stats = self.get_summary_stats().to_string() if not self.get_summary_stats().empty else "No numeric data"
            prompt = f"""You are a BI analyst. Given this dataset summary, write 3-4 bullet points
highlighting the most important insights, patterns, or anomalies. Be concise and business-focused.

Dataset: {self.name}
Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns
Numeric columns: {self.num_cols}
Categorical columns: {self.cat_cols}
Date columns: {self.date_cols}

Stats:
{stats}

Write bullet points starting with •. Max 4 bullets. Be specific with numbers."""
            return chat_completion([{"role": "user", "content": prompt}], max_tokens=300)
        except Exception:
            return "• Upload data and add your API key to get AI insights."