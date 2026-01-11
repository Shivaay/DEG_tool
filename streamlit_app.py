# ==========================================================
# DEG Analysis Toolkit — Cloud Safe & Type Safe
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, os, gzip, zipfile, tempfile

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from scipy import stats
import statsmodels.stats.multitest as smm
from gprofiler import GProfiler

# ---------------- CONFIG ----------------

st.set_page_config(page_title="DEG Professional Toolkit", layout="wide")
sns.set(style="white")

MAX_FILE_SIZE_MB = 1024

# ---------------- DEMO DATA ----------------

def load_demo_dataset():
    np.random.seed(1)
    genes = [f"Gene_{i}" for i in range(1, 20001)]
    data = np.random.poisson(lam=50, size=(20000, 6))
    df = pd.DataFrame(
        data,
        columns=["Ctrl1", "Ctrl2", "Ctrl3", "Treat1", "Treat2", "Treat3"]
    )
    df.insert(0, "Gene", genes)
    return df

# ---------------- FILE LOADER ----------------

def load_uploaded(uploaded_file):

    if int(uploaded_file.size) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large (max {MAX_FILE_SIZE_MB} MB)")
        st.stop()

    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    try:
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(raw))

        if name.endswith(".gz"):
            with gzip.open(io.BytesIO(raw), "rt", encoding="utf-8", errors="ignore") as f:
                return pd.read_csv(f, sep=None, engine="python")

        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    except Exception as e:
        raise ValueError("File parsing failed: " + str(e))

# ---------------- DATA SANITIZATION ----------------

def sanitize_expression(df):

    gene_col = df.columns[0]
    genes = df[gene_col].astype(str)

    expr = df.drop(columns=[gene_col])

    # FORCE numeric conversion
    expr = expr.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are completely non-numeric
    expr = expr.dropna(axis=1, how="all")

    if expr.shape[1] < 2:
        raise ValueError("Expression matrix must contain numeric sample columns")

    # Replace remaining NaNs with 0 (RNA-seq safe)
    expr = expr.fillna(0)

    expr.index = genes
    return expr

# ---------------- DEG ----------------

def compute_deg(expr, groupA, groupB):

    logmat = np.log2(expr.astype(float) + 1)

    results = []

    for gene in logmat.index:
        a = logmat.loc[gene, groupA]
        b = logmat.loc[gene, groupB]

        logfc = float(a.mean() - b.mean())
        _, p = stats.ttest_ind(a, b, equal_var=False)

        results.append((gene, logfc, float(p)))

    de = pd.DataFrame(results, columns=["Gene", "log2FC", "pvalue"])
    de.set_index("Gene", inplace=True)
    de["padj"] = smm.multipletests(de["pvalue"], method="fdr_bh")[1]

    return de

# ---------------- PLOTS ----------------

def plot_volcano(de, path, fc, padj, upc, downc):

    plt.figure(figsize=(7, 6))

    sig_up = de[(de.padj <= padj) & (de.log2FC >= fc)]
    sig_down = de[(de.padj <= padj) & (de.log2FC <= -fc)]
    nonsig = de.drop(sig_up.index.union(sig_down.index))

    plt.scatter(nonsig.log2FC, -np.log10(nonsig.pvalue), s=6, c="lightgrey")
    plt.scatter(sig_up.log2FC, -np.log10(sig_up.pvalue), s=10, c=upc)
    plt.scatter(sig_down.log2FC, -np.log10(sig_down.pvalue), s=10, c=downc)

    for g in pd.concat([sig_up.head(10), sig_down.head(10)]).itertuples():
        plt.text(float(g.log2FC), float(-np.log10(g.pvalue)), str(g.Index), fontsize=7)

    plt.axvline(fc, ls="--", c="black")
    plt.axvline(-fc, ls="--", c="black")
    plt.axhline(-np.log10(padj), ls=":", c="black")

    plt.xlabel("log2FC")
    plt.ylabel("-log10 p-value")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------------- HUB NETWORK ----------------

def hub_network(genes, path):

    if len(genes) < 10:
        return

    G = nx.barabasi_albert_graph(len(genes), 2, seed=1)
    G = nx.relabel_nodes(G, dict(enumerate(genes)))

    plt.figure(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=2)

    nx.draw(G, pos, node_size=300, node_color="#2ca02c", with_labels=True, font_size=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------------- UI ----------------

st.title("Professional DEG Analysis Toolkit")

uploaded = st.file_uploader("Upload expression matrix")

if st.button("Load Demo Dataset"):
    df = load_demo_dataset()
elif uploaded:
    df = load_uploaded(uploaded)
else:
    df = None

if df is not None:

    try:
        expr = sanitize_expression(df)
        st.success("Data loaded and validated")

    except Exception as e:
        st.error(str(e))
        st.stop()

    st.dataframe(expr.head())

    fc_cut = st.slider("Absolute log2FC cutoff", 0.5, 3.0, 1.0)
    padj_cut = st.slider("Adjusted p-value cutoff", 0.001, 0.1, 0.05)

    if st.button("Run DEG Analysis"):

        samples = expr.columns
        mid = len(samples) // 2
        groupA = samples[:mid]
        groupB = samples[mid:]

        de = compute_deg(expr, groupA, groupB)
        sig = de[(de.padj <= padj_cut) & (abs(de.log2FC) >= fc_cut)]

        tmp = tempfile.mkdtemp()
        volcano = os.path.join(tmp, "volcano.png")
        hub = os.path.join(tmp, "hub.png")

        plot_volcano(de, volcano, fc_cut, padj_cut, "#d62728", "#1f77b4")
        hub_network(list(sig.head(50).index), hub)

        st.image(volcano)
        st.image(hub)

        st.success("Analysis completed successfully")
