from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from filters import detect_columns, add_features, apply_filters
import charts

st.set_page_config(page_title="Reddit Depression India Dashboard", page_icon="🧠", layout="wide")

CSS = """
<style>
.block-container{padding-top:1.2rem;}
.hero{padding:28px;border-radius:26px;background:linear-gradient(135deg,#312E81 0%,#7C3AED 45%,#06B6D4 100%);color:white;box-shadow:0 20px 45px rgba(0,0,0,.25);}
.hero h1{font-size:2.35rem;margin-bottom:.25rem;}
.hero p{font-size:1rem;opacity:.95;}
.kpi{padding:18px;border-radius:20px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);}
.small-note{opacity:.75;font-size:.86rem;}
[data-testid="stMetricValue"]{font-size:1.85rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        if name.endswith(".json"):
            return pd.read_json(uploaded_file)
    data_dir = Path(__file__).parent / "data"
    files = [p for p in data_dir.glob("*") if p.suffix.lower() in [".csv", ".xlsx", ".xls", ".json"]]
    if not files:
        return None
    p = files[0]
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p)
    return pd.read_json(p)

st.markdown("""
<div class='hero'>
<h1>🧠 Reddit Depression in Indian Society — Premium EDA Dashboard</h1>
<p>Interactive mental-health discourse analytics with linked filters, KPI cards, NLP-style text features, and ten required professional visualizations.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload dataset file without renaming", type=["csv", "xlsx", "xls", "json"])
    st.caption("Or place the original Kaggle file inside the data/ folder.")

df_raw = load_data(uploaded)
if df_raw is None:
    st.warning("No dataset found. Put the Kaggle CSV/XLSX/JSON file in the data folder, or upload it from the sidebar.")
    st.code("python download_dataset.py\nstreamlit run app.py", language="bash")
    st.stop()

df_raw.columns = [str(c).strip() for c in df_raw.columns]
meta = detect_columns(df_raw)
with st.sidebar:
    st.header("Smart Column Mapping")
    date_col = st.selectbox("Date / time column", [None] + meta["date"], index=0)
    text_col = st.selectbox("Text column", [None] + meta["text"], index=1 if meta["text"] else 0)

df = add_features(df_raw, text_col=text_col, date_col=date_col)
meta = detect_columns(df)
category_options = [c for c in meta["category"] if c not in ["_clean_text"]] + ["_risk_band"]
numeric_options = list(dict.fromkeys(meta["numeric"] + ["_word_count", "_char_count", "_risk_score", "_negative_terms", "_positive_terms"]))

with st.sidebar:
    st.header("Linked Filters")
    date_range = None
    if df["_date"].notna().any():
        mn, mx = df["_date"].min().date(), df["_date"].max().date()
        date_range = st.date_input("Date range", value=(mn, mx), min_value=mn, max_value=mx)
        if not isinstance(date_range, tuple) or len(date_range) != 2:
            date_range = (mn, mx)
    category_col = st.selectbox("Category filter column", [None] + category_options, index=1 if category_options else 0)
    categories = None
    if category_col:
        vals = df[category_col].dropna().astype(str).value_counts().head(80).index.tolist()
        categories = st.multiselect("Select categories", vals, default=vals[: min(8, len(vals))])
    numeric_col = st.selectbox("Numerical range column", [None] + numeric_options, index=(numeric_options.index("_word_count") + 1 if "_word_count" in numeric_options else 0))
    num_range = None
    if numeric_col:
        s = pd.to_numeric(df[numeric_col], errors="coerce").dropna()
        if not s.empty:
            low, high = float(s.min()), float(s.max())
            if low < high:
                num_range = st.slider("Numerical range", low, high, (low, high))
    keyword = st.text_input("Search keyword", placeholder="e.g., anxiety, lonely, support")
    if st.button("Reset / Clear Filters"):
        st.rerun()

filtered = apply_filters(df, date_range, category_col, categories, numeric_col, num_range, keyword)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(filtered):,}", f"from {len(df):,}")
c2.metric("Avg Words", f"{filtered['_word_count'].mean() if len(filtered) else 0:.1f}")
c3.metric("Avg Risk Score", f"{filtered['_risk_score'].mean() if len(filtered) else 0:.2f}")
c4.metric("Critical Share", f"{(filtered['_risk_band'].astype(str).eq('Critical').mean()*100 if len(filtered) else 0):.1f}%")

st.markdown("---")
if filtered.empty:
    st.error("No rows match the current filters. Clear filters or widen the selected ranges.")
    st.stop()

main_cat = category_col if category_col else (category_options[0] if category_options else "_risk_band")
main_num = numeric_col if numeric_col else "_word_count"

tabs = st.tabs(["Executive Overview", "Required Charts", "Text Analytics", "Data Preview"])
with tabs[0]:
    left, right = st.columns([1.2, 1])
    with left:
        st.pyplot(charts.line_chart(filtered), use_container_width=True)
        st.pyplot(charts.area_chart(filtered), use_container_width=True)
    with right:
        st.pyplot(charts.count_plot(filtered), use_container_width=True)
        st.pyplot(charts.word_cloud(filtered), use_container_width=True)

with tabs[1]:
    st.subheader("All 10 Required Chart Types")
    a, b = st.columns(2)
    with a:
        st.pyplot(charts.pie_chart(filtered, main_cat), use_container_width=True)
        st.pyplot(charts.histogram(filtered, main_num), use_container_width=True)
        st.pyplot(charts.bar_chart(filtered, main_cat), use_container_width=True)
        st.pyplot(charts.box_plot(filtered, main_cat, main_num), use_container_width=True)
        st.pyplot(charts.heatmap(filtered), use_container_width=True)
    with b:
        st.pyplot(charts.line_chart(filtered), use_container_width=True)
        st.pyplot(charts.scatter_plot(filtered, main_num, "_risk_score"), use_container_width=True)
        st.pyplot(charts.area_chart(filtered), use_container_width=True)
        st.pyplot(charts.count_plot(filtered), use_container_width=True)
        st.pyplot(charts.violin_plot(filtered, main_cat, "_risk_score"), use_container_width=True)

with tabs[2]:
    st.subheader("Mental Health Text Signals")
    cols = ["_clean_text", "_word_count", "_negative_terms", "_positive_terms", "_risk_score", "_risk_band"]
    st.dataframe(filtered[cols].sort_values("_risk_score", ascending=False).head(100), use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "filtered_reddit_depression_dashboard.csv", "text/csv")

st.caption("Educational EDA only — this dashboard does not diagnose mental-health conditions.")
