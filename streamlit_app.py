# ==========================================================
# DEG Analysis Toolkit — LogFC & P-value Based
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, os, gzip, zipfile, tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

# ---------------- CONFIG ----------------

st.set_page_config("DEG Professional Toolkit", layout="wide")
sns.set(style="white")

# ---------------- FILE LOADER ----------------

def load_uploaded(file):
    raw = file.read()
    name = file.name.lower()

    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".gz"):
        with gzip.open(io.BytesIO(raw), "rt", errors="ignore") as f:
            return pd.read_csv(f, sep=None, engine="python")
    return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

# ---------------- PLOTS ----------------

def plot_volcano(df, gene_col, fc_col, p_col, fc_cut, p_cut, colors):

    sig = df[
        ((df[fc_col] >= fc_cut) | (df[fc_col] <= -fc_cut))
        & (df[p_col] <= p_cut)
    ]

    up = sig[sig[fc_col] > 0]
    down = sig[sig[fc_col] < 0]
    ns = df.drop(sig.index)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(ns[fc_col], -np.log10(ns[p_col]), c="lightgrey", s=6)
    ax.scatter(up[fc_col], -np.log10(up[p_col]), c=colors["up"], s=10)
    ax.scatter(down[fc_col], -np.log10(down[p_col]), c=colors["down"], s=10)

    ax.set_title(
        f"Total: {len(df)} | Up: {len(up)} | Down: {len(down)}",
        fontsize=10
    )

    ax.set_xlabel("logFC")
    ax.set_ylabel("-log10(p-value)")
    return fig, sig

def plot_heatmap(expr, genes):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(expr.loc[genes], cmap="vlag", ax=ax)
    ax.set_ylabel("Gene")
    return fig

def plot_hub_network(genes, method, color):

    if len(genes) < 5:
        return None, None

    G = nx.barabasi_albert_graph(len(genes), 2, seed=1)
    G = nx.relabel_nodes(G, dict(enumerate(genes)))

    if method == "Degree":
        score = nx.degree_centrality(G)
    else:
        score = {
            n: sum(len(c) for c in nx.find_cliques(G) if n in c)
            for n in G.nodes()
        }

    hub = sorted(score, key=score.get, reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=2)
    nx.draw(G, pos, nodelist=hub, node_color=color, with_labels=True, ax=ax)
    return fig, pd.Series(score).sort_values(ascending=False).head(10)

# ---------------- UI ----------------

st.title("DEG Analysis Toolkit (logFC & p-value based)")

uploaded = st.file_uploader("Upload DEG results (CSV / TSV / XLSX / GZ)")

if uploaded is None:
    st.info("Upload a DEG result file to begin")
    st.stop()

df = load_uploaded(uploaded)
st.success(f"Loaded {df.shape[0]} genes")

st.dataframe(df.head())

# ---------------- COLUMN SELECTION ----------------

gene_col = st.selectbox("Gene column", df.columns)
fc_col = st.selectbox("logFC column", df.columns)
p_col = st.selectbox("p-value column", df.columns)

# Force numeric
df[fc_col] = pd.to_numeric(df[fc_col], errors="coerce")
df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
df = df.dropna(subset=[fc_col, p_col])

# ---------------- FILTERS ----------------

fc_cut = st.selectbox(
    "logFC cutoff",
    list(range(-9, 0)) + list(range(1, 10)),
    index=8
)

p_cut = st.slider("p-value cutoff", 0.0001, 0.1, 0.05)

top_n = st.selectbox("Top genes for downstream analysis", [50, 100, 200, 500])

colors = {
    "up": st.color_picker("Upregulated color", "#d62728"),
    "down": st.color_picker("Downregulated color", "#1f77b4"),
    "network": st.color_picker("Network color", "#2ca02c")
}

hub_method = st.selectbox("Hub gene method", ["Degree", "MCC"])

# ---------------- RUN ----------------

if st.button("Run Analysis"):

    fig, sig = plot_volcano(
        df, gene_col, fc_col, p_col, abs(fc_cut), p_cut, colors
    )

    st.pyplot(fig)

    if sig.empty:
        st.warning("No genes passed the selected filters")
        st.stop()

    genes = sig.sort_values(p_col).head(top_n)[gene_col].astype(str).tolist()

    # Heatmap requires expression values → optional
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        expr = df.set_index(gene_col)[numeric_cols]
        st.pyplot(plot_heatmap(expr, genes))

    net_fig, hub_df = plot_hub_network(genes[:50], hub_method, colors["network"])
    if net_fig:
        st.pyplot(net_fig)
        st.dataframe(hub_df)

    gp = GProfiler(return_dataframe=True).profile(
        organism="hsapiens",
        query=genes
    )
    st.dataframe(gp)

    # ---------------- DOWNLOAD ----------------

    tmp = tempfile.mkdtemp()
    sig.to_excel(os.path.join(tmp, "Filtered_DEG.xlsx"), index=False)
    hub_df.to_excel(os.path.join(tmp, "Hub_genes.xlsx"))
    gp.to_excel(os.path.join(tmp, "gProfiler.xlsx"))

    zip_path = os.path.join(tmp, "Results.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(tmp):
            z.write(os.path.join(tmp, f), f)

    st.download_button(
        "Download all results",
        open(zip_path, "rb"),
        "DEG_results.zip"
    )

    st.success("Analysis completed successfully")
