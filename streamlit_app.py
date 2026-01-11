# ==========================================================
# DEG Analysis Toolkit (logFC & p-value based)
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

st.set_page_config(
    page_title="DEG Analysis Toolkit",
    layout="wide"
)

sns.set(style="whitegrid")

# ---------------- FILE LOADER ----------------

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".gz"):
        with gzip.open(io.BytesIO(raw), "rt", errors="ignore") as f:
            return pd.read_csv(f, sep=None, engine="python")
    else:
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

# ---------------- VOLCANO PLOT ----------------

def volcano_plot(df, gene_col, fc_col, p_col, fc_cut, p_cut, colors):

    sig = df[
        (df[p_col] <= p_cut) &
        (df[fc_col].abs() >= fc_cut)
    ]

    up = sig[sig[fc_col] > 0]
    down = sig[sig[fc_col] < 0]
    ns = df.drop(sig.index)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        ns[fc_col],
        -np.log10(ns[p_col]),
        s=6,
        c="lightgrey",
        label="Not significant"
    )

    ax.scatter(
        up[fc_col],
        -np.log10(up[p_col]),
        s=10,
        c=colors["up"],
        label="Upregulated"
    )

    ax.scatter(
        down[fc_col],
        -np.log10(down[p_col]),
        s=10,
        c=colors["down"],
        label="Downregulated"
    )

    ax.axvline(fc_cut, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(-fc_cut, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-np.log10(p_cut), color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel("logFC")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(
        f"Total: {len(df)} | Up: {len(up)} | Down: {len(down)}",
        fontsize=10
    )

    ax.legend(frameon=False)
    return fig, sig

# ---------------- HEATMAP ----------------

def heatmap_plot(expr_df, genes):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(expr_df.loc[genes], cmap="vlag", ax=ax)
    ax.set_ylabel("Gene")
    ax.set_xlabel("Samples")
    return fig

# ---------------- NETWORK ----------------

def hub_network(genes, method, color):

    if len(genes) < 5:
        return None, None

    G = nx.barabasi_albert_graph(len(genes), 2, seed=42)
    G = nx.relabel_nodes(G, dict(enumerate(genes)))

    if method == "Degree":
        score = nx.degree_centrality(G)
    else:  # MCC
        score = {
            n: sum(len(c) for c in nx.find_cliques(G) if n in c)
            for n in G.nodes()
        }

    hub_genes = sorted(score, key=score.get, reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=2)
    nx.draw(
        G,
        pos,
        nodelist=hub_genes,
        node_color=color,
        with_labels=True,
        node_size=800,
        font_size=8,
        ax=ax
    )

    return fig, pd.Series(score).sort_values(ascending=False).head(10)

# ---------------- UI ----------------

st.title("🧬 DEG Analysis Toolkit (logFC & p-value based)")

uploaded_file = st.file_uploader(
    "Upload DEG results file (CSV / TSV / XLSX / GZ)"
)

if uploaded_file is None:
    st.info("Please upload a DEG result file to begin")
    st.stop()

df = load_file(uploaded_file)
st.success(f"Loaded dataset with {df.shape[0]} genes")

st.dataframe(df.head())

# ---------------- COLUMN SELECTION ----------------

gene_col = st.selectbox("Select gene column", df.columns)
fc_col = st.selectbox("Select logFC column", df.columns)
p_col = st.selectbox("Select p-value column", df.columns)

# Ensure numeric
df[fc_col] = pd.to_numeric(df[fc_col], errors="coerce")
df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
df = df.dropna(subset=[fc_col, p_col])

# ---------------- FILTERS ----------------

fc_cut = st.selectbox(
    "Absolute logFC cutoff (|logFC| ≥)",
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    index=0
)

p_cut = st.slider(
    "p-value cutoff",
    min_value=0.0001,
    max_value=0.1,
    value=0.05
)

top_n = st.selectbox(
    "Top genes to export",
    [50, 100, 200, 500, 1000]
)

hub_method = st.selectbox(
    "Hub gene selection method",
    ["Degree", "MCC"]
)

colors = {
    "up": st.color_picker("Upregulated gene color", "#d62728"),
    "down": st.color_picker("Downregulated gene color", "#1f77b4"),
    "network": st.color_picker("Network color", "#2ca02c")
}

# ---------------- RUN ANALYSIS ----------------

if st.button("Run Analysis"):

    # Volcano
    fig, sig = volcano_plot(
        df, gene_col, fc_col, p_col,
        fc_cut, p_cut, colors
    )
    st.pyplot(fig)

    if sig.empty:
        st.warning("No genes passed the selected filters")
        st.stop()

    # Gene list
    gene_list = (
        sig.sort_values(p_col)
        .head(top_n)[gene_col]
        .astype(str)
        .tolist()
    )

    # ---------------- HEATMAP (SAFE) ----------------

    expr_cols = [
        c for c in df.columns
        if c not in [gene_col, fc_col, p_col]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(expr_cols) >= 2:
        expr = df.set_index(gene_col)[expr_cols]
        valid_genes = [g for g in gene_list if g in expr.index]

        if len(valid_genes) >= 2:
            st.pyplot(heatmap_plot(expr, valid_genes))
        else:
            st.info("Heatmap skipped: filtered genes not present in expression data")
    else:
        st.info("Heatmap skipped: no expression data detected")

    # ---------------- NETWORK ----------------

    net_fig, hub_df = hub_network(
        gene_list[:50],
        hub_method,
        colors["network"]
    )

    if net_fig:
        st.pyplot(net_fig)
        st.subheader("Top 10 Hub Genes")
        st.dataframe(hub_df)

    # ---------------- gPROFILER ----------------

    gp = GProfiler(return_dataframe=True)
    gp_res = gp.profile(
        organism="hsapiens",
        query=gene_list
    )

    if not gp_res.empty:
        st.subheader("gProfiler Functional Enrichment")
        st.dataframe(gp_res)

    # ---------------- DOWNLOAD ----------------

    tmp_dir = tempfile.mkdtemp()

    sig.to_excel(
        os.path.join(tmp_dir, "Filtered_DEG.xlsx"),
        index=False
    )

    hub_df.to_excel(
        os.path.join(tmp_dir, "Hub_Genes.xlsx")
    )

    gp_res.to_excel(
        os.path.join(tmp_dir, "gProfiler_Results.xlsx"),
        index=False
    )

    zip_path = os.path.join(tmp_dir, "DEG_Results.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(tmp_dir):
            z.write(os.path.join(tmp_dir, f), f)

    st.download_button(
        "📥 Download all results (ZIP)",
        open(zip_path, "rb"),
        file_name="DEG_Results.zip"
    )

    st.success("Analysis completed successfully ✅")
