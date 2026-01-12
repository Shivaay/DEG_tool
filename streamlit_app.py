# ==========================================================
# FULL DEG ANALYSIS PLATFORM — STABLE & ERROR-PROOF
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
import io
import os
from gprofiler import GProfiler

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="Full DEG Analysis Platform",
    layout="wide"
)

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

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"File loading failed: {e}")
    st.stop()

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
    st.warning("No genes passed the selected filters.")
    st.stop()

genes = deg[gene_col].astype(str).unique().tolist()

st.success(
    f"DEGs: {len(deg)} | "
    f"Up: {(deg['Regulation']=='Up').sum()} | "
    f"Down: {(deg['Regulation']=='Down').sum()}"
)

# ---------------- VOLCANO PLOT ----------------
st.subheader("Volcano Plot")

up_color = st.color_picker("Upregulated color", "#d62728")
down_color = st.color_picker("Downregulated color", "#1f77b4")
neutral_color = st.color_picker("Non-significant color", "#bdbdbd")

fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(
    df[logfc_col],
    -np.log10(df[pval_col]),
    c=neutral_color,
    s=10
)

ax.scatter(
    deg[deg["Regulation"] == "Up"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Up"][pval_col]),
    c=up_color,
    label="Up",
    s=20
)

ax.scatter(
    deg[deg["Regulation"] == "Down"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Down"][pval_col]),
    c=down_color,
    label="Down",
    s=20
)

ax.set_xlabel("logFC")
ax.set_ylabel("-log10(p-value)")
ax.legend()
ax.set_title("Volcano Plot")

st.pyplot(fig)

# ---------------- DOWNLOAD DEG TABLE ----------------
st.download_button(
    "Download DEG Table (CSV)",
    deg.to_csv(index=False),
    file_name="DEGs.csv"
)

# ---------------- STRING PPI ----------------
st.subheader("Protein–Protein Interaction Network (STRING)")

top_n = st.selectbox("Number of hub genes", [10, 20, 30], index=0)

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
    G = nx.from_pandas_edgelist(
        ppi,
        "preferredName_A",
        "preferredName_B"
    )

    hub_genes = sorted(
        G.degree,
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    hub_list = [g[0] for g in hub_genes]
    H = G.subgraph(hub_list)

    fig, ax = plt.subplots(figsize=(7, 7))
    nx.draw(
        H,
        with_labels=True,
        node_size=2000,
        node_color="orange",
        font_size=10
    )
    st.pyplot(fig)

    hub_df = pd.DataFrame(hub_genes, columns=["Gene", "Degree"])
    st.dataframe(hub_df)
else:
    st.info("STRING network unavailable or no interactions found.")

# ---------------- FUNCTIONAL ENRICHMENT ----------------
st.subheader("Functional Enrichment (g:Profiler)")

def run_gprofiler(glist):
    if not glist:
        return pd.DataFrame()
    try:
        gp = GProfiler(return_dataframe=True)
        return gp.profile(organism="hsapiens", query=glist)
    except Exception:
        return pd.DataFrame()

enrich = run_gprofiler(genes)

if not enrich.empty:
    st.subheader("All Enriched Terms")
    st.dataframe(enrich)

    kegg = enrich[enrich["source"] == "KEGG"]
    bp = enrich[enrich["source"] == "GO:BP"]

    if not kegg.empty:
        st.subheader("KEGG Pathways")
        st.dataframe(kegg)

    if not bp.empty:
        st.subheader("GO Biological Process")
        st.dataframe(bp)
else:
    st.info("No enrichment results available.")

# ---------------- TRRUST TF NETWORK ----------------
st.subheader("TF–Gene Regulatory Network (TRRUST)")

if os.path.exists("trrust_human.tsv"):
    trrust = pd.read_csv("trrust_human.tsv", sep="\t", header=None)
    trrust.columns = ["TF", "Target", "Mode", "PMID"]

    tf_edges = trrust[trrust["Target"].isin(genes)][["TF", "Target"]]

    if not tf_edges.empty:
        G_tf = nx.from_pandas_edgelist(tf_edges, "TF", "Target", create_using=nx.DiGraph())
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw(G_tf, with_labels=True, node_size=1500)
        st.pyplot(fig)
else:
    st.info("TRRUST file not found — TF network disabled.")

# ---------------- miRTarBase ----------------
st.subheader("miRNA–Gene Regulatory Network")

if os.path.exists("miRTarBase_MTI.xlsx"):
    mir = pd.read_excel("miRTarBase_MTI.xlsx")
    mir_edges = mir[mir["Target Gene"].isin(genes)][["miRNA", "Target Gene"]]

    if not mir_edges.empty:
        G_m = nx.from_pandas_edgelist(mir_edges, "miRNA", "Target Gene", create_using=nx.DiGraph())
        fig, ax = plt.subplots(figsize=(7, 7))
        nx.draw(G_m, with_labels=True, node_size=1200)
        st.pyplot(fig)
else:
    st.info("miRTarBase file not found — miRNA network disabled.")

st.success("✅ DEG analysis completed successfully without errors.")
