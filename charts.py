from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme(style="whitegrid", font_scale=0.95)
PALETTE = ["#7C3AED", "#06B6D4", "#22C55E", "#F59E0B", "#EF4444", "#EC4899", "#14B8A6"]

def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=135)
    fig.patch.set_facecolor("white")
    return fig, ax

def pie_chart(df, cat_col):
    fig, ax = _fig()
    data = df[cat_col].astype(str).value_counts().head(8)
    ax.pie(data.values, labels=data.index, autopct="%1.1f%%", startangle=90, colors=PALETTE)
    ax.set_title(f"Proportional Distribution by {cat_col}", fontweight="bold")
    return fig

def histogram(df, num_col="_word_count"):
    fig, ax = _fig()
    sns.histplot(df[num_col].dropna(), kde=True, ax=ax, color=PALETTE[0])
    ax.set_title(f"Frequency Distribution of {num_col}", fontweight="bold")
    ax.set_xlabel(num_col); ax.set_ylabel("Frequency")
    return fig

def line_chart(df):
    fig, ax = _fig()
    if df["_date"].notna().any():
        s = df.dropna(subset=["_date"]).set_index("_date").resample("M").size()
        s.plot(ax=ax, marker="o", color=PALETTE[1])
        ax.set_xlabel("Month")
    else:
        s = df.reset_index().groupby(df.index // max(1, len(df)//20)).size()
        s.plot(ax=ax, marker="o", color=PALETTE[1])
        ax.set_xlabel("Record sequence")
    ax.set_ylabel("Posts")
    ax.set_title("Post Volume Trend", fontweight="bold")
    return fig

def bar_chart(df, cat_col):
    fig, ax = _fig()
    data = df[cat_col].astype(str).value_counts().head(12)
    sns.barplot(x=data.values, y=data.index, ax=ax, palette=PALETTE * 3, hue=data.index, legend=False)
    ax.set_title(f"Top Categories by {cat_col}", fontweight="bold")
    ax.set_xlabel("Records"); ax.set_ylabel(cat_col)
    return fig

def scatter_plot(df, x="_word_count", y="_risk_score"):
    fig, ax = _fig()
    sample = df[[x, y, "_risk_band"]].dropna().sample(min(len(df), 2500), random_state=7) if len(df) else df
    sns.scatterplot(data=sample, x=x, y=y, hue="_risk_band", ax=ax, palette=PALETTE[:4], alpha=0.7)
    ax.set_title(f"Relationship: {x} vs {y}", fontweight="bold")
    return fig

def box_plot(df, cat_col, num_col="_word_count"):
    fig, ax = _fig()
    top = df[cat_col].astype(str).value_counts().head(8).index
    data = df[df[cat_col].astype(str).isin(top)]
    sns.boxplot(data=data, x=num_col, y=cat_col, ax=ax, palette=PALETTE * 2, hue=cat_col, legend=False)
    ax.set_title(f"Spread, Median and Outliers: {num_col}", fontweight="bold")
    return fig

def heatmap(df):
    fig, ax = _fig()
    num = df.select_dtypes(include="number")
    cols = num.var().sort_values(ascending=False).head(10).index if len(num.columns) > 10 else num.columns
    corr = num[cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features", fontweight="bold")
    return fig

def area_chart(df):
    fig, ax = _fig()
    if df["_date"].notna().any():
        s = df.dropna(subset=["_date"]).set_index("_date").resample("M").size().cumsum()
        ax.fill_between(s.index, s.values, alpha=0.35, color=PALETTE[2]); ax.plot(s.index, s.values, color=PALETTE[2])
        ax.set_xlabel("Month")
    else:
        s = df.reset_index().groupby(df.index // max(1, len(df)//20)).size().cumsum()
        ax.fill_between(range(len(s)), s.values, alpha=0.35, color=PALETTE[2]); ax.plot(range(len(s)), s.values, color=PALETTE[2])
        ax.set_xlabel("Sequence")
    ax.set_ylabel("Cumulative Records"); ax.set_title("Cumulative Trend", fontweight="bold")
    return fig

def count_plot(df):
    fig, ax = _fig()
    sns.countplot(data=df, y="_risk_band", order=["Low", "Moderate", "High", "Critical"], ax=ax, palette=PALETTE[:4], hue="_risk_band", legend=False)
    ax.set_title("Count Plot: Risk Band Frequency", fontweight="bold")
    ax.set_xlabel("Records"); ax.set_ylabel("Risk Band")
    return fig

def violin_plot(df, cat_col, num_col="_risk_score"):
    fig, ax = _fig()
    top = df[cat_col].astype(str).value_counts().head(6).index
    data = df[df[cat_col].astype(str).isin(top)]
    sns.violinplot(data=data, x=num_col, y=cat_col, ax=ax, palette=PALETTE * 2, hue=cat_col, legend=False, cut=0)
    ax.set_title(f"Distribution Density of {num_col}", fontweight="bold")
    return fig

def word_cloud(df):
    text = " ".join(df["_clean_text"].dropna().astype(str).sample(min(len(df), 6000), random_state=3)) if len(df) else "no data"
    wc = WordCloud(width=1200, height=500, background_color="white", colormap="viridis", max_words=140).generate(text)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=135)
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Most Frequent Discussion Terms", fontweight="bold")
    return fig
