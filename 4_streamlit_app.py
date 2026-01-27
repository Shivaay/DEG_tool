# ==========================================================
# FULL DEG ANALYSIS PLATFORM (STABLE + UPGRADED)
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
from gprofiler import GProfiler

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Full DEG Analysis", layout="wide")
st.title("🧬 Full DEG Analysis Platform")

# ---------------- FILE UPLOAD ----------------
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

# ---------------- COLUMN MAPPING ----------------
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

# ---------------- FILTERING ----------------
st.sidebar.header("Filtering")
neg_fc = st.sidebar.slider("Negative logFC (≤)", -10, -1, -1)
pos_fc = st.sidebar.slider("Positive logFC (≥)", 1, 10, 1)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
if deg.empty:
    st.warning("No genes passed filters.")
    st.stop()

genes = deg[gene_col].astype(str).unique().tolist()

# ---------------- VOLCANO ----------------
st.subheader("Volcano Plot")

up_color = st.color_picker("Upregulated color", "#d62728")
down_color = st.color_picker("Downregulated color", "#1f77b4")
neutral_color = st.color_picker("Non-significant color", "#bdbdbd")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c=neutral_color, s=10)
ax.scatter(
    deg[deg["Regulation"] == "Up"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Up"][pval_col]),
    c=up_color, label="Up", s=20
)
ax.scatter(
    deg[deg["Regulation"] == "Down"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Down"][pval_col]),
    c=down_color, label="Down", s=20
)
ax.set_xlabel("logFC")
ax.set_ylabel("-log10(p-value)")
ax.legend()
st.pyplot(fig)

# ---------------- HUB OPTIONS ----------------
st.subheader("Hub Gene Selection")
hub_n = st.selectbox("Number of hub genes", [10, 20, 30], index=0)
hub_method = st.radio("Hub scoring method", ["Degree", "MCC"])

# ---------------- STRING PPI ----------------
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
    try:
        r = requests.post(url, data=params, timeout=30)
        if r.status_code != 200:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(r.text), sep="\t")
    except Exception:
        return pd.DataFrame()

ppi = fetch_string_ppi(genes)

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")

    if hub_method == "Degree":
        hub_scores = dict(G.degree())
    else:  # MCC approximation
        hub_scores = {
            n: nx.clustering(G, n) * G.degree(n)
            for n in G.nodes()
        }

    hub_df = (
        pd.DataFrame(hub_scores.items(), columns=["Gene", "Score"])
        .sort_values("Score", ascending=False)
        .head(hub_n)
    )

    hub_list = hub_df["Gene"].tolist()

    st.subheader("Hub Genes")
    st.dataframe(hub_df)

    # Clean rectangular PPI
    H = G.subgraph(hub_list)
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(H, seed=42)

    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.6)
    nx.draw_networkx_nodes(
        H, pos,
        node_shape="s",
        node_color="#ffcc80",
        node_size=2500,
        ax=ax
    )
    nx.draw_networkx_labels(H, pos, font_size=10, ax=ax)
    ax.set_title("PPI Network (Hub Genes)")
    ax.axis("off")
    st.pyplot(fig)

# ---------------- HEATMAP ----------------
st.subheader("Heatmap (Hub Genes)")

expr_cols = [
    c for c in df.columns
    if c not in [gene_col, logfc_col, pval_col, "Regulation"]
    and pd.api.types.is_numeric_dtype(df[c])
]

if len(expr_cols) >= 2:
    heat_df = df[df[gene_col].isin(hub_list)].set_index(gene_col)[expr_cols]
    if not heat_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(heat_df, cmap="RdBu_r", center=0, ax=ax)
        st.pyplot(fig)
else:
    st.info("Expression matrix not detected — heatmap skipped.")

# ---------------- TRRUST ----------------
st.subheader("TF–Gene Network (TRRUST)")

if os.path.exists("trrust_human.tsv"):
    trrust = pd.read_csv("trrust_human.tsv", sep="\t", header=None)
    trrust.columns = ["TF", "Target", "Mode", "PMID"]
    tf_edges = trrust[trrust["Target"].isin(hub_list)][["TF", "Target"]]

    if not tf_edges.empty:
        G_tf = nx.DiGraph(tf_edges.values.tolist())
        TFs = tf_edges["TF"].unique()
        targets = tf_edges["Target"].unique()

        pos = {}
        pos.update((tf, (0, i)) for i, tf in enumerate(TFs))
        pos.update((tg, (1, i)) for i, tg in enumerate(targets))

        fig, ax = plt.subplots(figsize=(9, 6))
        nx.draw_networkx_edges(G_tf, pos, ax=ax)
        nx.draw_networkx_nodes(G_tf, pos, TFs, node_shape="s", node_color="#90caf9", node_size=2500)
        nx.draw_networkx_nodes(G_tf, pos, targets, node_shape="s", node_color="#ffab91", node_size=2500)
        nx.draw_networkx_labels(G_tf, pos, font_size=9)
        ax.axis("off")
        st.pyplot(fig)
else:
    st.info("TRRUST file not found — TF network disabled.")

# ---------------- FUNCTIONAL ENRICHMENT ----------------
st.subheader("Functional Enrichment")

gp = GProfiler(return_dataframe=True)
enrich = gp.profile(organism="hsapiens", query=genes)

if not enrich.empty:
    st.subheader("All Terms")
    st.dataframe(enrich)

    st.subheader("KEGG")
    st.dataframe(enrich[enrich["source"] == "KEGG"])

    st.subheader("GO:BP")
    st.dataframe(enrich[enrich["source"] == "GO:BP"])

st.success("✅ Analysis completed successfully.")
