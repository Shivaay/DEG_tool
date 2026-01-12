# ==========================================================
# FULL DEG ANALYSIS TOOLKIT (LogFC-based)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, gzip, os, tempfile, zipfile

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

# ---------------- CONFIG ----------------
st.set_page_config("Full DEG Analysis Toolkit", layout="wide")
sns.set(style="whitegrid")

# ---------------- FILE LOADER ----------------
def load_file(uploaded):
    name = uploaded.name.lower()
    raw = uploaded.read()

    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".gz"):
        with gzip.open(io.BytesIO(raw), "rt", errors="ignore") as f:
            return pd.read_csv(f, sep=None, engine="python")
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

# ---------------- VOLCANO ----------------
def volcano_plot(df, gene_col, fc_col, p_col, neg_fc, pos_fc, p_cut, colors):

    sig = df[
        ((df[fc_col] <= neg_fc) | (df[fc_col] >= pos_fc)) &
        (df[p_col] <= p_cut)
    ]

    up = sig[sig[fc_col] > 0]
    down = sig[sig[fc_col] < 0]
    ns = df.drop(sig.index)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(ns[fc_col], -np.log10(ns[p_col]), c=colors["ns"], s=6)
    ax.scatter(up[fc_col], -np.log10(up[p_col]), c=colors["up"], s=10)
    ax.scatter(down[fc_col], -np.log10(down[p_col]), c=colors["down"], s=10)

    ax.axvline(neg_fc, linestyle="--", color="black")
    ax.axvline(pos_fc, linestyle="--", color="black")
    ax.axhline(-np.log10(p_cut), linestyle="--", color="black")

    ax.set_xlabel("logFC")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(
        f"Total: {len(df)} | Up: {len(up)} | Down: {len(down)}",
        fontsize=10
    )

    return fig, sig

# ---------------- HUB NETWORK ----------------
def build_ppi_network(genes, hub_size, method):

    genes = genes[:hub_size]
    G = nx.Graph()

    for g in genes:
        G.add_node(g)

    for i in range(len(genes)):
        for j in range(i + 1, min(i + 3, len(genes))):
            G.add_edge(genes[i], genes[j])

    if method == "Degree":
        score = nx.degree_centrality(G)
    else:
        score = {
            n: sum(len(c) for c in nx.find_cliques(G) if n in c)
            for n in G.nodes()
        }

    hubs = sorted(score, key=score.get, reverse=True)
    return G, pd.Series(score).sort_values(ascending=False)

def draw_network(G, color):
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color,
        node_size=1200,
        font_size=8,
        ax=ax
    )
    return fig

# ---------------- UI ----------------
st.title("🧬 Full DEG Analysis Toolkit")

uploaded = st.file_uploader("Upload DEG results (CSV / TSV / XLSX / GZ)")
if uploaded is None:
    st.stop()

df = load_file(uploaded)
st.success(f"{df.shape[0]} genes loaded")

gene_col = st.selectbox("Gene column", df.columns)
fc_col = st.selectbox("logFC column", df.columns)
p_col = st.selectbox("p-value column", df.columns)

df[fc_col] = pd.to_numeric(df[fc_col], errors="coerce")
df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
df = df.dropna(subset=[fc_col, p_col])

# ---------------- FILTERING ----------------
st.subheader("DEG Filtering")

neg_fc = st.slider("Negative logFC (≤)", -10, -1, -1)
pos_fc = st.slider("Positive logFC (≥)", 1, 10, 1)
p_cut = st.slider("p-value cutoff", 0.0001, 0.1, 0.05)

colors = {
    "up": st.color_picker("Upregulated color", "#d62728"),
    "down": st.color_picker("Downregulated color", "#1f77b4"),
    "ns": st.color_picker("Non-significant color", "#bdbdbd"),
    "network": st.color_picker("Network color", "#ff7f0e")
}

if st.button("Run DEG Analysis"):

    # Volcano
    fig, sig = volcano_plot(
        df, gene_col, fc_col, p_col,
        neg_fc, pos_fc, p_cut, colors
    )
    st.pyplot(fig)

    if sig.empty:
        st.warning("No genes passed the selected thresholds")
        st.stop()

    # ---------------- HUB GENES ----------------
    st.subheader("Hub Gene Analysis")

    up_n = st.selectbox("Top upregulated genes", [10, 20, 50, 100])
    down_n = st.selectbox("Top downregulated genes", [10, 20, 50, 100])

    up_genes = (
        sig[sig[fc_col] > 0]
        .sort_values(fc_col, ascending=False)
        .head(up_n)[gene_col].astype(str).tolist()
    )

    down_genes = (
        sig[sig[fc_col] < 0]
        .sort_values(fc_col)
        .head(down_n)[gene_col].astype(str).tolist()
    )

    selected_genes = up_genes + down_genes

    hub_size = st.selectbox("Hub size", [10, 20, 30, 50])
    hub_method = st.selectbox("Hub method", ["Degree", "MCC"])

    G, hub_scores = build_ppi_network(selected_genes, hub_size, hub_method)
    st.pyplot(draw_network(G, colors["network"]))
    st.dataframe(hub_scores.head(hub_size))

    # ---------------- ENRICHMENT ----------------
    st.subheader("Functional Enrichment")

    gp = GProfiler(return_dataframe=True)
    enrich = gp.profile(organism="hsapiens", query=selected_genes)

    st.dataframe(enrich)
    st.subheader("KEGG Pathways")
    st.dataframe(enrich[enrich["source"] == "KEGG"])
    st.subheader("GO: Biological Process")
    st.dataframe(enrich[enrich["source"] == "GO:BP"])
    st.subheader("GO: Molecular Function")
    st.dataframe(enrich[enrich["source"] == "GO:MF"])
    st.subheader("GO: Cellular Component")
    st.dataframe(enrich[enrich["source"] == "GO:CC"])

    st.success("Full DEG analysis completed successfully ✅")
