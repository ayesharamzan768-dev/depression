from __future__ import annotations
import re
import pandas as pd
import numpy as np

DATE_HINTS = ["date", "created", "time", "timestamp", "posted", "utc"]
TEXT_HINTS = ["text", "body", "post", "title", "content", "selftext", "comment"]
CAT_HINTS = ["subreddit", "author", "label", "flair", "category", "sentiment", "class"]

NEGATIVE_WORDS = set("depressed depression sad lonely hopeless anxiety anxious stress stressed suicide suicidal tired worthless empty crying pain fear panic hurt broken lost help insomnia alone".split())
POSITIVE_WORDS = set("happy hope hopeful better good great calm support love loved strong improve recovered grateful peace peaceful smile okay fine".split())

def detect_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    lower = {c: str(c).lower() for c in cols}
    date_cols = [c for c in cols if any(h in lower[c] for h in DATE_HINTS)]
    text_cols = [c for c in cols if any(h in lower[c] for h in TEXT_HINTS) or df[c].dtype == "object"]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if (df[c].dtype == "object" or str(df[c].dtype) == "category") and df[c].nunique(dropna=True) <= 80]
    cat_cols = sorted(set(cat_cols + [c for c in cols if any(h in lower[c] for h in CAT_HINTS)]), key=lambda x: cols.index(x))
    return {"date": date_cols, "text": text_cols, "numeric": numeric_cols, "category": cat_cols}

def clean_text(s: object) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s'_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def add_features(df: pd.DataFrame, text_col: str | None = None, date_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        out["_date"] = out[date_col]
    else:
        out["_date"] = pd.NaT

    if text_col and text_col in out.columns:
        text = out[text_col].map(clean_text)
    else:
        obj_cols = out.select_dtypes(include="object").columns.tolist()
        text = out[obj_cols].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text) if obj_cols else pd.Series([""] * len(out), index=out.index)

    out["_clean_text"] = text
    out["_word_count"] = text.str.split().map(len)
    out["_char_count"] = text.str.len()
    out["_negative_terms"] = text.str.lower().str.split().map(lambda words: sum(w in NEGATIVE_WORDS for w in words))
    out["_positive_terms"] = text.str.lower().str.split().map(lambda words: sum(w in POSITIVE_WORDS for w in words))
    out["_risk_score"] = (out["_negative_terms"] - out["_positive_terms"] + np.log1p(out["_word_count"])).round(2)
    out["_risk_band"] = pd.cut(out["_risk_score"], bins=[-999, 2, 5, 9, 999], labels=["Low", "Moderate", "High", "Critical"])
    return out

def apply_filters(df: pd.DataFrame, date_range=None, category_col=None, categories=None, numeric_col=None, num_range=None, keyword="") -> pd.DataFrame:
    out = df.copy()
    if date_range and "_date" in out.columns and out["_date"].notna().any():
        start, end = date_range
        out = out[(out["_date"] >= pd.to_datetime(start)) & (out["_date"] <= pd.to_datetime(end) + pd.Timedelta(days=1))]
    if category_col and categories and category_col in out.columns:
        out = out[out[category_col].astype(str).isin([str(x) for x in categories])]
    if numeric_col and num_range and numeric_col in out.columns:
        out = out[out[numeric_col].between(num_range[0], num_range[1], inclusive="both")]
    if keyword:
        keyword = keyword.lower().strip()
        out = out[out["_clean_text"].str.lower().str.contains(keyword, na=False, regex=False)]
    return out
