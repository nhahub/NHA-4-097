"""
app.py — BI Decision Support Agent
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import OrchestratorAgent
from memory.memory_module import MemoryModule
from tools.auto_dashboard import AutoDashboard
from utils.language_detector import detect_language

st.set_page_config(page_title="BI Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.kpi-grid{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}
.kpi-card{background:linear-gradient(135deg,#6366f115,#818cf808);border:1px solid #6366f130;
  border-radius:12px;padding:14px 18px;min-width:140px;flex:1}
.kpi-label{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.kpi-value{font-size:24px;font-weight:700;color:#1f2937}
.kpi-delta{font-size:11px;color:#6b7280;margin-top:2px}
.section-hdr{font-size:15px;font-weight:600;color:#6366f1;border-left:3px solid #6366f1;
  padding-left:10px;margin:20px 0 12px}
.agent-tag{display:inline-block;background:#ede9fe;color:#5b21b6;border-radius:10px;
  padding:2px 10px;font-size:11px;font-weight:600;margin-bottom:6px}
.insight-box{background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
  padding:12px 16px;margin-bottom:8px;font-size:13px;line-height:1.6}
</style>
""", unsafe_allow_html=True)


# ── Session init ──────────────────────────────────────────────────────────────
def init_state():
    if "messages"      not in st.session_state: st.session_state.messages      = []
    if "uploaded_data" not in st.session_state: st.session_state.uploaded_data = {}
    if "dashboards"    not in st.session_state: st.session_state.dashboards    = {}
    if "memory"        not in st.session_state: st.session_state.memory        = MemoryModule()
    if "orchestrator"  not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent(st.session_state.memory)

init_state()


def reset_app():
    for k in ["messages","uploaded_data","dashboards"]:
        st.session_state[k] = [] if k == "messages" else {}
    st.session_state.memory       = MemoryModule()
    st.session_state.orchestrator = OrchestratorAgent(st.session_state.memory)
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    if st.button("🔄 Reset Session", use_container_width=True):
        reset_app()

    # Show active LLM
    if os.getenv("GROQ_API_KEY"):
        st.success(f"✅ Groq · {os.getenv('GROQ_MODEL','llama-3.3-70b-versatile')}")
    elif os.getenv("OPENAI_API_KEY"):
        st.success(f"✅ OpenAI · {os.getenv('OPENAI_MODEL','gpt-4o-mini')}")
    elif os.getenv("ANTHROPIC_API_KEY"):
        st.success("✅ Anthropic · Claude")

    st.divider()
    st.markdown("## 📂 Upload Data")
    uploaded_files = st.file_uploader(
        "CSV, Excel, PDF, Word, TXT",
        accept_multiple_files=True,
        type=["csv","xlsx","xls","pdf","txt","docx"],
    )
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_data:
                with st.spinner(f"Processing {f.name}..."):
                    result = st.session_state.orchestrator.ingest_file(f)
                    st.session_state.uploaded_data[f.name] = result
                    df_uploaded = result.get("dataframe")
                    if df_uploaded is not None:
                        st.session_state.dashboards[f.name] = AutoDashboard(df_uploaded, f.name)
                st.success(f"✅ {f.name}")

    if st.session_state.uploaded_data:
        st.markdown("**Loaded:**")
        for fname, data in st.session_state.uploaded_data.items():
            rows = data.get("rows","")
            st.markdown(f"- 📄 `{fname}`" + (f" ({rows:,} rows)" if rows else ""))

    st.divider()
    st.markdown("## 🧠 Memory")
    mem = st.session_state.memory
    c1, c2 = st.columns(2)
    c1.metric("Turns",    mem.short_term_count())
    c2.metric("Insights", mem.long_term_count())

    lang_mode = st.selectbox("Language", ["Auto","English","العربية"])


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_chat, tab_stats = st.tabs(["📊 Dashboard","💬 Chat","📋 Stats"])


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    if not st.session_state.dashboards:
        st.info("📂 Upload a CSV or Excel file to auto-generate your dashboard.")
        st.stop()

    # Dataset selector
    fnames = list(st.session_state.dashboards.keys())
    sel    = st.selectbox("Dataset", fnames) if len(fnames) > 1 else fnames[0]
    dash: AutoDashboard = st.session_state.dashboards[sel]
    df = st.session_state.uploaded_data[sel]["dataframe"]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Key Metrics</div>', unsafe_allow_html=True)
    kpis     = dash.get_kpis()
    kpi_html = '<div class="kpi-grid">'
    for k in kpis:
        delta = f'<div class="kpi-delta">{k["delta"]}</div>' if k["delta"] else ""
        kpi_html += (
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{k["label"]}</div>'
            f'<div class="kpi-value">{k["value"]}</div>{delta}'
            f'</div>'
        )
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    has_key = any(os.getenv(k) for k in ("GROQ_API_KEY","OPENAI_API_KEY","ANTHROPIC_API_KEY"))
    if has_key:
        st.markdown('<div class="section-hdr">🤖 AI Insights</div>', unsafe_allow_html=True)
        with st.spinner("Generating insights..."):
            narrative = dash.get_ai_narrative()
        for line in [l.strip() for l in narrative.split("\n") if l.strip()]:
            st.markdown(f'<div class="insight-box">{line}</div>', unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔍 Filter Data", expanded=False):
        filters = {}
        ncols = min(len(dash.cat_cols[:3]) + (1 if dash.date_cols else 0), 4)
        fcols = st.columns(max(ncols, 1))

        for i, cat in enumerate(dash.cat_cols[:3]):
            with fcols[i]:
                opts = ["All"] + sorted(df[cat].dropna().unique().tolist())
                sel2 = st.selectbox(cat.replace("_"," ").title(), opts, key=f"flt_{cat}")
                if sel2 != "All":
                    filters[cat] = sel2

        if dash.date_cols:
            with fcols[min(len(dash.cat_cols[:3]), ncols - 1)]:
                dc = dash.date_cols[0]
                mn, mx = df[dc].min().date(), df[dc].max().date()
                dr = st.date_input("Date range", value=(mn, mx))
                if isinstance(dr, tuple) and len(dr) == 2:
                    filters["__date__"] = (dc, dr)

        df_f = df.copy()
        for col, val in filters.items():
            if col == "__date__":
                dc2, (d1, d2) = val
                df_f = df_f[(df_f[dc2].dt.date >= d1) & (df_f[dc2].dt.date <= d2)]
            else:
                df_f = df_f[df_f[col] == val]

        if filters:
            st.caption(f"Filtered: {len(df_f):,} / {len(df):,} rows")

    active_dash = AutoDashboard(df_f, sel) if filters else dash

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Charts</div>', unsafe_allow_html=True)

    with st.spinner("Building charts..."):
        charts = active_dash.generate_all_charts()

    if not charts:
        st.warning("No chartable columns found in this dataset.")
    else:
        def render_grid(lst, cols=2):
            for i in range(0, len(lst), cols):
                row  = lst[i: i + cols]
                grid = st.columns(len(row))
                for j, ch in enumerate(row):
                    with grid[j]:
                        st.plotly_chart(
                            ch["fig"], use_container_width=True,
                            key=f"d_{sel}_{ch['type']}_{i}_{j}"
                        )

        time_c = [c for c in charts if c["type"] == "timeseries"]
        bar_c  = [c for c in charts if c["type"] == "bar"]
        pie_c  = [c for c in charts if c["type"] == "pie"]
        rest_c = [c for c in charts if c["type"] not in ("timeseries","bar","pie")]

        if time_c:
            st.markdown("**📈 Trends**")
            render_grid(time_c, cols=1 if len(time_c) == 1 else 2)
        if bar_c:
            st.markdown("**📊 Breakdowns**")
            render_grid(bar_c)
        if pie_c:
            st.markdown("**🥧 Composition**")
            render_grid(pie_c)
        if rest_c:
            st.markdown("**🔍 Analysis**")
            render_grid(rest_c)

    # ── Raw data ──────────────────────────────────────────────────────────────
    with st.expander("📋 Raw Data", expanded=False):
        st.dataframe(df_f, use_container_width=True, height=300)
        st.download_button(
            "⬇ Download filtered CSV",
            df_f.to_csv(index=False).encode("utf-8"),
            file_name=f"{sel}_filtered.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Ask anything about your data")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("agent"):
                st.markdown(
                    f'<span class="agent-tag">via {msg["agent"]}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])
            if msg.get("chart"):
                st.plotly_chart(msg["chart"], use_container_width=True, key=f"c_{id(msg)}")

    user_input = st.chat_input("Ask about your data...")
    if user_input:
        lang = detect_language(user_input)
        if lang_mode == "English":   lang = "en"
        elif lang_mode == "العربية": lang = "ar"

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.orchestrator.run(
                        query=user_input,
                        target_lang=lang,
                        uploaded_files=list(st.session_state.uploaded_data.keys()),
                    )
                except Exception as e:
                    response = {"answer": f"❌ Error: {e}", "agent": "error", "chart": None}

            agent = response.get("agent", "orchestrator")
            st.markdown(f'<span class="agent-tag">via {agent}</span>', unsafe_allow_html=True)
            st.markdown(response["answer"])
            if response.get("chart"):
                st.plotly_chart(response["chart"], use_container_width=True)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": response["answer"],
            "agent":   response.get("agent"),
            "chart":   response.get("chart"),
        })


# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    if not st.session_state.dashboards:
        st.info("📂 Upload a file first.")
        st.stop()

    sel3  = st.selectbox("Dataset", list(st.session_state.dashboards.keys()), key="stats_sel")
    dash3 = st.session_state.dashboards[sel3]
    df3   = st.session_state.uploaded_data[sel3]["dataframe"]

    st.markdown('<div class="section-hdr">Numeric Summary</div>', unsafe_allow_html=True)
    stats = dash3.get_summary_stats()
    if not stats.empty:
        st.dataframe(stats, use_container_width=True)

    st.markdown('<div class="section-hdr">Categorical Summary</div>', unsafe_allow_html=True)
    cat_info = [
        {
            "Column":   c,
            "Unique":   df3[c].nunique(),
            "Top":      str(df3[c].value_counts().index[0]) if df3[c].notna().any() else "—",
            "Missing":  int(df3[c].isnull().sum()),
            "Missing%": round(df3[c].isnull().mean() * 100, 1),
        }
        for c in dash3.cat_cols
    ]
    if cat_info:
        st.dataframe(pd.DataFrame(cat_info), use_container_width=True)

    st.markdown('<div class="section-hdr">All Columns</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({
            "Column":   df3.columns,
            "Type":     df3.dtypes.values.astype(str),
            "Non-null": df3.notna().sum().values,
            "Null%":    (df3.isnull().mean() * 100).round(1).values,
            "Sample":   [
                str(df3[c].dropna().iloc[0]) if df3[c].notna().any() else "—"
                for c in df3.columns
            ],
        }),
        use_container_width=True,
        height=400,
    )