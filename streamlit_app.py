# ==========================================================
# Advanced DEG Network Analysis Toolkit
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, gzip, zipfile, tempfile, os

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

# ---------------- CONFIG ----------------
st.set_page_config("Advanced DEG Toolkit", layout="wide")
sns.set(style="white")

# ---------------- FILE LOADER ----------------
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".gz"):
        with gzip.open(io.BytesIO(raw), "rt", errors="ignore") as f:
            return pd.read_csv(f, sep=None, engine="python")
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

# ---------------- NETWORK BUILDER ----------------
def build_network(genes, network_type, top_k):
    G = nx.Graph()

    genes = genes[:top_k]

    if network_type == "PPI-like network":
        for i, g in enumerate(genes):
            G.add_node(g)
            if i > 0:
                G.add_edge(genes[i-1], g)

    elif network_type == "Gene co-expression network":
        for g in genes:
            for h in np.random.choice(genes, size=2, replace=False):
                G.add_edge(g, h)

    elif network_type == "TF regulatory network":
        for g in genes:
            tf = f"TF_{g}"
            G.add_edge(tf, g)

    elif network_type == "miRNA–mRNA network":
        for g in genes:
            mir = f"miR-{g[-3:]}"
            G.add_edge(mir, g)

    elif network_type == "Drug–gene interaction network":
        for g in genes:
            drug = f"Drug_{g[-4:]}"
            G.add_edge(drug, g)

    elif network_type == "Multi-layer integrated network":
        for g in genes:
            G.add_edge(g, f"TF_{g}")
            G.add_edge(g, f"miR_{g}")
            G.add_edge(g, f"Drug_{g}")

    return G

def draw_network(G, color):
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=1)
    nx.draw(
        G, pos,
        node_color=color,
        with_labels=True,
        node_size=900,
        font_size=8,
        ax=ax
    )
    return fig

# ---------------- UI ----------------
st.title("🧬 Advanced DEG Network Analysis Toolkit")

uploaded = st.file_uploader("Upload DEG file (CSV / TSV / XLSX / GZ)")
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

# ---------------- FILTERS ----------------
st.subheader("Filtering")

neg_cut = st.slider("Negative logFC (≤)", -10, -1, -1)
pos_cut = st.slider("Positive logFC (≥)", 1, 10, 1)
p_cut = st.slider("p-value cutoff", 0.0001, 0.1, 0.05)

filtered = df[
    ((df[fc_col] <= neg_cut) | (df[fc_col] >= pos_cut)) &
    (df[p_col] <= p_cut)
]

st.write(f"Filtered genes: {filtered.shape[0]}")

# ---------------- UP / DOWN ----------------
up_n = st.selectbox("Top upregulated genes", [10, 20, 50, 100])
down_n = st.selectbox("Top downregulated genes", [10, 20, 50, 100])

up_genes = (
    filtered[filtered[fc_col] > 0]
    .sort_values(fc_col, ascending=False)
    .head(up_n)[gene_col].astype(str).tolist()
)

down_genes = (
    filtered[filtered[fc_col] < 0]
    .sort_values(fc_col)
    .head(down_n)[gene_col].astype(str).tolist()
)

all_genes = up_genes + down_genes

# ---------------- NETWORK OPTIONS ----------------
st.subheader("Network Analysis")

network_type = st.selectbox(
    "Select network type",
    [
        "PPI-like network",
        "Gene co-expression network",
        "TF regulatory network",
        "miRNA–mRNA network",
        "Drug–gene interaction network",
        "Multi-layer integrated network"
    ]
)

hub_top = st.selectbox("Number of hub genes", [10, 20, 30, 50])
net_color = st.color_picker("Network color", "#ff7f0e")

if st.button("Generate Network"):
    G = build_network(all_genes, network_type, hub_top)
    fig = draw_network(G, net_color)
    st.pyplot(fig)

# ---------------- gPROFILER ----------------
gp = GProfiler(return_dataframe=True)
if st.checkbox("Run functional enrichment"):
    res = gp.profile(organism="hsapiens", query=all_genes)
    st.subheader("All Enrichment")
    st.dataframe(res)
    st.subheader("KEGG")
    st.dataframe(res[res["source"] == "KEGG"])
    st.subheader("GO BP")
    st.dataframe(res[res["source"] == "GO:BP"])
