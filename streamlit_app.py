# ==========================================================
# Professional DEG Analysis Web Tool
# Streamlit Cloud–Safe Version
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

# ---------------------- CONFIG ----------------------

st.set_page_config(page_title="DEG Professional Toolkit", layout="wide")
sns.set(style="white")

MAX_FILE_SIZE_MB = 1024  # 1 GB

# ---------------------- DEMO DATA ----------------------

def load_demo_dataset():
    np.random.seed(1)
    genes = [f"Gene_{i}" for i in range(1, 20001)]
    sham = np.random.poisson(50, (20000, 3))
    treat = np.random.poisson(80, (20000, 3))

    df = pd.DataFrame(
        np.hstack([sham, treat]),
        columns=["Sham1", "Sham2", "Sham3", "Treat1", "Treat2", "Treat3"]
    )
    df.insert(0, "Gene", genes)
    return df

# ---------------------- FILE LOADER ----------------------

def load_uploaded(uploaded_file):

    if int(uploaded_file.size) > int(MAX_FILE_SIZE_MB) * 1024 * 1024:
        st.error(f"File too large. Maximum {MAX_FILE_SIZE_MB} MB allowed.")
        st.stop()

    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    try:
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(data))

        if name.endswith(".gz"):
            with gzip.open(io.BytesIO(data), "rt", encoding="utf-8", errors="ignore") as fh:
                return pd.read_csv(fh, sep=None, engine="python")

        return pd.read_csv(io.BytesIO(data), sep=None, engine="python", low_memory=False)

    except Exception as e:
        raise ValueError("Could not parse file: " + str(e))

# ---------------------- DEG COMPUTATION ----------------------

def compute_deg(expr, groupA, groupB):

    logmat = np.log2(expr + 1)
    results = []

    for gene, row in logmat.iterrows():
        a = row[groupA]
        b = row[groupB]

        logfc = float(a.mean() - b.mean())
        _, p = stats.ttest_ind(a, b, equal_var=False)

        results.append((str(gene), logfc, float(p)))

    de = pd.DataFrame(results, columns=["Gene", "log2FC", "pvalue"])
    de.set_index("Gene", inplace=True)

    de["padj"] = smm.multipletests(de["pvalue"], method="fdr_bh")[1]
    return de

# ---------------------- VOLCANO PLOT ----------------------

def plot_volcano(de, outpng, fc_cut, padj_cut, up_color, down_color):

    plt.figure(figsize=(7, 6))

    sig_up = de[(de.padj <= padj_cut) & (de.log2FC >= fc_cut)]
    sig_down = de[(de.padj <= padj_cut) & (de.log2FC <= -fc_cut)]
    nonsig = de.drop(sig_up.index.union(sig_down.index))

    plt.scatter(nonsig.log2FC, -np.log10(nonsig.pvalue), s=6, c="lightgrey")
    plt.scatter(sig_up.log2FC, -np.log10(sig_up.pvalue), s=10, c=up_color)
    plt.scatter(sig_down.log2FC, -np.log10(sig_down.pvalue), s=10, c=down_color)

    for g in pd.concat([sig_up.head(15), sig_down.head(15)]).itertuples():
        plt.text(
            float(g.log2FC),
            float(-np.log10(g.pvalue)),
            str(g.Index),
            fontsize=7
        )

    plt.axvline(fc_cut, linestyle="--", color="black")
    plt.axvline(-fc_cut, linestyle="--", color="black")
    plt.axhline(-np.log10(padj_cut), linestyle=":", color="black")

    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10 p-value")

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

# ---------------------- HEATMAP ----------------------

def plot_heatmap(expr, genes, outpng):

    if len(genes) == 0:
        return

    sub = expr.loc[genes]
    z = (sub.sub(sub.mean(axis=1), axis=0)).div(sub.std(axis=1), axis=0)

    plt.figure(figsize=(9, 7))
    sns.heatmap(z, cmap="vlag", yticklabels=True)

    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

# ---------------------- HUB GENES ----------------------

def compute_mcc(G):
    cliques = list(nx.find_cliques(G))
    return {n: sum(len(c) for c in cliques if n in c) for n in G.nodes()}

def hub_network(gene_list, method, node_color, outpng):

    if len(gene_list) < 5:
        return pd.DataFrame()

    G = nx.barabasi_albert_graph(len(gene_list), 2, seed=1)
    mapping = {i: str(gene_list[i]) for i in range(len(gene_list))}
    G = nx.relabel_nodes(G, mapping)

    scores = nx.degree_centrality(G) if method == "Degree" else compute_mcc(G)
    hub = pd.Series(scores).sort_values(ascending=False).head(10)

    plt.figure(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=2)

    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, nodelist=hub.index, node_color=node_color, node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    plt.close()

    return hub.to_frame("score")

# ---------------------- GPROFILER ----------------------

def run_gprofiler(genes):

    if len(genes) < 5:
        return pd.DataFrame()

    gp = GProfiler(return_dataframe=True)
    return gp.profile(organism="hsapiens", query=[str(g) for g in genes])

# ---------------------- UI ----------------------

st.title("Professional DEG Analysis Toolkit")

uploaded = st.file_uploader("Upload expression matrix (CSV/TSV/XLSX/GZ)")

if st.button("Load Demo Dataset"):
    df = load_demo_dataset()
    st.success("Demo dataset loaded successfully")

elif uploaded:
    try:
        df = load_uploaded(uploaded)
        st.success("File loaded successfully")
    except Exception as e:
        st.error(str(e))
        df = None
else:
    df = None

if df is not None:
    st.dataframe(df.head())

fc_cut = st.slider("Absolute log2FC cutoff", 0.5, 3.0, 1.0, 0.1)
padj_cut = st.slider("Adjusted p-value cutoff", 0.001, 0.1, 0.05, 0.001)

up_color = st.color_picker("Upregulated color", "#d62728")
down_color = st.color_picker("Downregulated color", "#1f77b4")

hub_method = st.selectbox("Hub gene method", ["Degree", "MCC"])
hub_color = st.color_picker("Hub node color", "#2ca02c")

run = st.button("Run DEG Analysis")

# ---------------------- RUN PIPELINE ----------------------

if run and df is not None:

    expr = df.set_index(df.columns[0])

    samples = expr.columns
    groupA = [s for s in samples if "sham" in s.lower() or "ctrl" in s.lower()]
    groupB = [s for s in samples if s not in groupA]

    if len(groupA) == 0:
        mid = len(samples) // 2
        groupA = samples[:mid]
        groupB = samples[mid:]

    de = compute_deg(expr, groupB, groupA)

    sig = de[(de.padj <= padj_cut) & (abs(de.log2FC) >= fc_cut)]
    up = sig[sig.log2FC > 0]
    down = sig[sig.log2FC < 0]

    tmpdir = tempfile.mkdtemp()

    volcano_png = os.path.join(tmpdir, "volcano.png")
    heatmap_png = os.path.join(tmpdir, "heatmap.png")
    hub_png = os.path.join(tmpdir, "hub.png")

    plot_volcano(de, volcano_png, fc_cut, padj_cut, up_color, down_color)
    plot_heatmap(expr, list(sig.head(100).index), heatmap_png)

    hub_df = hub_network(list(sig.head(200).index), hub_method, hub_color, hub_png)

    gprof = run_gprofiler(list(sig.index))
    gprof.to_csv(os.path.join(tmpdir, "gprofiler.csv"), index=False)

    up.to_csv(os.path.join(tmpdir, "Upregulated.csv"))
    down.to_csv(os.path.join(tmpdir, "Downregulated.csv"))

    zip_path = os.path.join(tmpdir, "Results.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for f in os.listdir(tmpdir):
            if f.endswith((".png", ".csv")):
                z.write(os.path.join(tmpdir, f), f)

    st.image(volcano_png)
    st.image(heatmap_png)
    st.image(hub_png)

    st.download_button("Download All Results (ZIP)", open(zip_path, "rb"), "DEG_results.zip")

    st.success("Analysis completed successfully")
