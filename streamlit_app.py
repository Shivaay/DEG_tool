import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import networkx as nx
import matplotlib.pyplot as plt
from gprofiler import GProfiler

st.set_page_config(layout="wide")
st.title("🧬 Full DEG Analysis Platform")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("Upload DEG File")
uploaded = st.sidebar.file_uploader(
    "CSV / TSV / XLSX (≤ 1GB)",
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

# ---------------- COLUMN SELECTION ----------------
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene Column", df.columns)
logfc_col = st.sidebar.selectbox("logFC Column", df.columns)
pval_col = st.sidebar.selectbox("p-value Column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

# ---------------- FILTERING ----------------
st.sidebar.header("Filtering")
logfc_thresh = st.sidebar.select_slider(
    "logFC Threshold",
    options=list(range(-9, 0)) + list(range(1, 10)),
    value=(-1, 1)
)
pval_thresh = st.sidebar.slider("p-value cutoff", 0.0, 1.0, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= logfc_thresh[1]) & (df[pval_col] <= pval_thresh), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= logfc_thresh[0]) & (df[pval_col] <= pval_thresh), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]

st.success(f"DEGs identified: {len(deg)}")

# ---------------- VOLCANO ----------------
st.subheader("Volcano Plot")

col_up = st.color_picker("Upregulated color", "#d62728")
col_down = st.color_picker("Downregulated color", "#1f77b4")
col_neutral = st.color_picker("Neutral color", "#bdbdbd")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c=col_neutral, s=10)
ax.scatter(
    deg[deg["Regulation"]=="Up"][logfc_col],
    -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),
    c=col_up, s=20, label="Up"
)
ax.scatter(
    deg[deg["Regulation"]=="Down"][logfc_col],
    -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),
    c=col_down, s=20, label="Down"
)
ax.legend()
ax.set_xlabel("logFC")
ax.set_ylabel("-log10(p-value)")
ax.set_title(
    f"Total: {len(df)} | Up: {(deg['Regulation']=='Up').sum()} | Down: {(deg['Regulation']=='Down').sum()}"
)
st.pyplot(fig)

# ---------------- DOWNLOAD ----------------
st.download_button(
    "Download DEG Table",
    deg.to_csv(index=False),
    "DEGs.csv"
)

# ---------------- STRING PPI ----------------
st.subheader("PPI Network (STRING)")
genes = deg[gene_col].astype(str).unique().tolist()

@st.cache_data
def string_ppi(glist):
    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0d".join(glist[:200]),
        "species": 9606
    }
    r = requests.post(url, data=params)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(pd.compat.StringIO(r.text), sep="\t")

ppi = string_ppi(genes)

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    hub_genes = [h[0] for h in hubs]

    st.write("Top 10 Hub Genes:", hub_genes)

    H = G.subgraph(hub_genes)
    fig, ax = plt.subplots(figsize=(6,6))
    nx.draw(
        H,
        with_labels=True,
        node_color="orange",
        node_size=2000,
        font_size=10
    )
    st.pyplot(fig)

# ---------------- FUNCTIONAL ENRICHMENT ----------------
st.subheader("Functional Enrichment (g:Profiler)")
gp = GProfiler(return_dataframe=True)
enrich = gp.profile(
    organism="hsapiens",
    query=genes
)

if not enrich.empty:
    st.dataframe(enrich)

    st.subheader("KEGG Pathways")
    st.dataframe(enrich[enrich["source"]=="KEGG"])

    st.subheader("GO Biological Process")
    st.dataframe(enrich[enrich["source"]=="GO:BP"])

# ---------------- TRRUST ----------------
st.subheader("TF–Gene Network (TRRUST)")
if os.path.exists("trrust_human.tsv"):
    trrust = pd.read_csv("trrust_human.tsv", sep="\t", header=None)
    trrust.columns = ["TF","Target","Mode","PMID"]
    tf_edges = trrust[trrust["Target"].isin(genes)][["TF","Target"]]

    if not tf_edges.empty:
        G_tf = nx.from_pandas_edgelist(tf_edges, "TF", "Target")
        fig, ax = plt.subplots(figsize=(6,6))
        nx.draw(G_tf, with_labels=True, node_size=1500)
        st.pyplot(fig)
else:
    st.info("TRRUST file not found — feature disabled safely")

# ---------------- miRTarBase ----------------
st.subheader("miRNA–Gene Network")
if os.path.exists("miRTarBase_MTI.xlsx"):
    mir = pd.read_excel("miRTarBase_MTI.xlsx")
    mir_edges = mir[mir["Target Gene"].isin(genes)][["miRNA","Target Gene"]]
    if not mir_edges.empty:
        G_m = nx.from_pandas_edgelist(mir_edges, "miRNA", "Target Gene")
        fig, ax = plt.subplots(figsize=(6,6))
        nx.draw(G_m, with_labels=True, node_size=1200)
        st.pyplot(fig)
else:
    st.info("miRTarBase file not found — feature disabled safely")
