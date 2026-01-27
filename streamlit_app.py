# ==========================================================
# FULL DEG ANALYSIS PLATFORM (BASE + EXTENSIONS, SAFE)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
import os
import json
from datetime import datetime
from gprofiler import GProfiler

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Full DEG Analysis", layout="wide")
st.title("🧬 Full DEG Analysis Platform")

# ---------------- FILE UPLOAD (BASE) ----------------
uploaded = st.file_uploader(
    "Upload DEG results (CSV / TSV / XLSX, ≤ 1 GB)",
    type=["csv", "tsv", "xlsx"]
)

if uploaded is None:
    st.stop()

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    return pd.read_excel(file)

df = load_data(uploaded)
st.success(f"Dataset loaded: {df.shape[0]} genes")

# ---------------- COLUMN MAPPING (BASE) ----------------
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

# ---------------- FILTERING (BASE) ----------------
st.sidebar.header("Filtering")
neg_fc = st.sidebar.slider("Negative logFC (≤)", -10, -1, -1)
pos_fc = st.sidebar.slider("Positive logFC (≥)", 1, 10, 1)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
genes = deg[gene_col].astype(str).unique().tolist()

# ======================================================
# 🔹 NEW SECTION 1: EXECUTIVE SUMMARY (RULE-BASED)
# ======================================================
st.header("📌 Executive Summary")

up = (deg["Regulation"] == "Up").sum()
down = (deg["Regulation"] == "Down").sum()

exec_summary = f"""
Differential expression analysis identified **{len(deg)} genes**
with statistically significant expression changes
(p ≤ {p_cut}).
Among these, **{up} genes were upregulated**
and **{down} genes were downregulated**,
indicating condition-specific transcriptional regulation.
"""

st.markdown(exec_summary)

# ======================================================
# 🔹 BASE FUNCTIONALITY (UNCHANGED BELOW)
# ======================================================

# ---------------- VOLCANO (BASE) ----------------
st.subheader("Volcano Plot")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c="grey", s=10)
ax.scatter(
    deg[deg["Regulation"] == "Up"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Up"][pval_col]),
    c="red", label="Up", s=20
)
ax.scatter(
    deg[deg["Regulation"] == "Down"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Down"][pval_col]),
    c="blue", label="Down", s=20
)
ax.legend()
st.pyplot(fig)

st.info(
    "Volcano plots highlight genes with both large expression changes "
    "and strong statistical significance."
)

# ---------------- HUB OPTIONS (BASE) ----------------
st.subheader("Hub Gene Selection")
hub_n = st.selectbox("Number of hub genes", [10, 20, 30], index=0)

# ---------------- STRING PPI (BASE) ----------------
@st.cache_data
def fetch_string_ppi(glist):
    if not glist:
        return pd.DataFrame()
    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0d".join(glist[:200]),
        "species": 9606,
        "required_score": 700
    }
    r = requests.post(url, data=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text), sep="\t")

ppi = fetch_string_ppi(genes)

hub_genes = []
if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:hub_n]
    hub_genes = [h[0] for h in hubs]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G.subgraph(hub_genes), seed=42)
    nx.draw_networkx(G.subgraph(hub_genes), pos, ax=ax)
    ax.axis("off")
    st.pyplot(fig)

    st.info(
        "Highly connected hub genes may represent key regulators "
        "in the underlying biological condition."
    )

# ---------------- HEATMAP (BASE) ----------------
st.subheader("Heatmap (Hub Genes)")

expr_cols = [
    c for c in df.columns
    if c not in [gene_col, logfc_col, pval_col, "Regulation"]
    and pd.api.types.is_numeric_dtype(df[c])
]

if expr_cols and hub_genes:
    heat_df = df[df[gene_col].isin(hub_genes)].set_index(gene_col)[expr_cols]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(heat_df, cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)

    st.info(
        "Heatmaps show relative expression patterns across samples, "
        "revealing similarity and clustering trends."
    )

# ---------------- FUNCTIONAL ENRICHMENT (BASE) ----------------
st.subheader("Functional Enrichment")

gp = GProfiler(return_dataframe=True)
enrich = gp.profile(organism="hsapiens", query=genes)

if not enrich.empty:
    st.dataframe(enrich)

# ======================================================
# 🔹 NEW SECTION 2: AI-ASSISTED MANUSCRIPT SUMMARY
# ======================================================

st.header("📝 AI-Assisted Manuscript Summary")

use_ai = st.checkbox(
    "Enable AI-generated manuscript summary (optional)",
    help="Uses an external language model if API key is available."
)

def generate_rule_summary():
    return f"""
RNA-seq differential expression analysis identified {len(deg)} genes
with significant expression changes (p ≤ {p_cut}).
Upregulated genes suggest activation of condition-associated pathways,
while downregulated genes indicate suppression of specific biological processes.
Functional enrichment and network analyses highlight key regulatory genes.
"""

def generate_ai_summary(context):
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            return None

        prompt = f"""
You are a bioinformatics scientist.
Write a manuscript-ready Results paragraph
based on the following analysis summary:

{context}
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception:
        return None

context = {
    "DEGs": len(deg),
    "Upregulated": up,
    "Downregulated": down,
    "Hub_genes": hub_genes[:5]
}

ai_text = None
if use_ai:
    ai_text = generate_ai_summary(json.dumps(context, indent=2))

final_summary = ai_text if ai_text else generate_rule_summary()
st.text_area("Manuscript-Ready Summary", final_summary, height=220)

# ======================================================
# 🔹 NEW SECTION 3: REPRODUCIBILITY METADATA
# ======================================================
st.header("🔁 Reproducibility")

metadata = {
    "timestamp": datetime.utcnow().isoformat(),
    "p_value_cutoff": p_cut,
    "logFC_positive": pos_fc,
    "logFC_negative": neg_fc,
    "total_genes": len(df),
    "DEGs": len(deg)
}

st.json(metadata)

st.download_button(
    "Download Run Metadata",
    json.dumps(metadata, indent=2),
    file_name="run_metadata.json"
)

st.success("✅ Analysis completed successfully.")
