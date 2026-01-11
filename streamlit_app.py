# ==========================================================
# DEG Analysis Toolkit – Fully Error-Proof Version
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, os, gzip, zipfile, tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from gprofiler import GProfiler

# ---------------- CONFIG ----------------

st.set_page_config("DEG Professional Toolkit", layout="wide")
sns.set(style="white")

# ---------------- DEMO DATA ----------------

def load_demo_dataset(n_genes=20000):
    np.random.seed(1)
    genes = [f"Gene_{i}" for i in range(1, n_genes + 1)]
    ctrl = np.random.poisson(50, (n_genes, 3))
    trt = np.random.poisson(80, (n_genes, 3))

    df = pd.DataFrame(
        np.hstack([ctrl, trt]),
        columns=["Ctrl_1", "Ctrl_2", "Ctrl_3", "Treat_1", "Treat_2", "Treat_3"]
    )
    df.insert(0, "Gene", genes)
    return df

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

# ---------------- DATA PREP ----------------

def prepare_expression(df):
    gene_col = df.columns[0]
    expr = df.drop(columns=[gene_col])
    expr = expr.apply(pd.to_numeric, errors="coerce").fillna(0)
    expr.index = df[gene_col].astype(str)
    return expr

# ---------------- DEG ----------------

def compute_deg(expr, ctrl, treat):

    # HARD VALIDATION (prevents zero-size error)
    if len(ctrl) < 2 or len(treat) < 2:
        raise ValueError("Each group must contain at least 2 samples")

    logmat = np.log2(expr + 1)

    logFC = logmat[treat].mean(axis=1) - logmat[ctrl].mean(axis=1)

    pvals = ttest_ind(
        logmat[treat].values,
        logmat[ctrl].values,
        axis=1,
        equal_var=False
    ).pvalue

    padj = multipletests(pvals, method="fdr_bh")[1]

    return pd.DataFrame(
        {"log2FC": logFC, "pvalue": pvals, "padj": padj},
        index=expr.index
    )

# ---------------- VOLCANO ----------------

def plot_volcano(de, fc_cut, padj_cut, colors):

    sig = de[(abs(de.log2FC) >= fc_cut) & (de.padj <= padj_cut)]
    up = sig[sig.log2FC > 0]
    down = sig[sig.log2FC < 0]
    ns = de.drop(sig.index)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ns.log2FC, -np.log10(ns.pvalue), c="lightgrey", s=6)
    ax.scatter(up.log2FC, -np.log10(up.pvalue), c=colors["up"], s=10)
    ax.scatter(down.log2FC, -np.log10(down.pvalue), c=colors["down"], s=10)

    ax.set_title(
        f"Total genes: {len(de)} | Up: {len(up)} | Down: {len(down)}",
        fontsize=10
    )

    ax.set_xlabel("log2FC")
    ax.set_ylabel("-log10(p-value)")
    return fig, sig

# ---------------- HEATMAP ----------------

def plot_heatmap(expr, genes):

    if len(genes) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(expr.loc[genes], cmap="vlag", ax=ax)
    ax.set_ylabel("Gene")
    return fig

# ---------------- HUB NETWORK ----------------

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

# ---------------- GPROFILER ----------------

def run_gprofiler(genes):

    if len(genes) < 5:
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)
    return gp.profile(organism="hsapiens", query=genes)

# ---------------- UI ----------------

st.title("Professional DEG Analysis Toolkit")

uploaded = st.file_uploader("Upload Expression Matrix")

if st.button("Load Demo Dataset"):
    df = load_demo_dataset()
elif uploaded:
    df = load_uploaded(uploaded)
else:
    st.info("Upload a file or load demo dataset")
    st.stop()

expr = prepare_expression(df)

st.success(f"{expr.shape[0]} genes | {expr.shape[1]} samples loaded")

ctrl = st.multiselect("Control samples (≥2)", expr.columns, expr.columns[:3])
treat = st.multiselect("Treatment samples (≥2)", expr.columns, expr.columns[3:6])

fc_cut = st.selectbox("log2FC cutoff", [1, 2, 3, 4, 5])
padj_cut = st.slider("Adjusted p-value cutoff", 0.001, 0.1, 0.05)
top_n = st.selectbox("Top genes to export", [50, 100, 200, 500])

colors = {
    "up": st.color_picker("Upregulated color", "#d62728"),
    "down": st.color_picker("Downregulated color", "#1f77b4"),
    "network": st.color_picker("Network color", "#2ca02c")
}

hub_method = st.selectbox("Hub gene method", ["Degree", "MCC"])

# ---------------- RUN ----------------

if st.button("Run Analysis"):

    try:
        de = compute_deg(expr, ctrl, treat)
    except Exception as e:
        st.error(str(e))
        st.stop()

    fig, sig = plot_volcano(de, fc_cut, padj_cut, colors)
    st.pyplot(fig)

    if sig.empty:
        st.warning("No significant genes found with current filters")
        st.stop()

    genes = sig.sort_values("padj").head(top_n).index.tolist()

    heat = plot_heatmap(expr, genes)
    if heat:
        st.pyplot(heat)

    net_fig, hub_df = plot_hub_network(genes[:50], hub_method, colors["network"])
    if net_fig:
        st.pyplot(net_fig)
        st.dataframe(hub_df)

    gp = run_gprofiler(genes)
    if not gp.empty:
        st.dataframe(gp)

    tmp = tempfile.mkdtemp()
    sig.to_excel(os.path.join(tmp, "DEG_results.xlsx"))
    gp.to_excel(os.path.join(tmp, "gProfiler.xlsx"))

    zip_path = os.path.join(tmp, "Results.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(tmp):
            z.write(os.path.join(tmp, f), f)

    st.download_button("Download all results", open(zip_path, "rb"), "DEG_results.zip")

    st.success("Analysis completed successfully")
