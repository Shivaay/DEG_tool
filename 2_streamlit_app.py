# ==========================================================
# FULL DEG ANALYSIS TOOLKIT WITH REAL BIOLOGICAL NETWORKS
# STRING + TRRUST + miRTarBase
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, gzip, os, tempfile, zipfile
import requests

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

# ---------------- STRING API ----------------
def fetch_string_ppi(genes, score_cutoff=700, species=9606):
    interactions = []
    genes_str = "%0d".join(genes)

    url = (
        "https://string-db.org/api/tsv/network"
        f"?identifiers={genes_str}"
        f"&species={species}"
        f"&required_score={score_cutoff}"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        for line in r.text.split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) > 5:
                interactions.append((parts[2], parts[3]))
    except Exception:
        return []

    return interactions

# ---------------- TRRUST LOADER ----------------
@st.cache_data
def load_trrust():
    url = "https://raw.githubusercontent.com/SlowhandedBio/TRRUST/master/data/trrust_rawdata.human.tsv"
    df = pd.read_csv(url, sep="\t", header=None)
    df.columns = ["TF", "Target", "Mode", "PMID"]
    return df

# ---------------- miRTarBase (USER UPLOAD) ----------------
def load_mirtarbase(upload):
    if upload is None:
        return None
    return load_file(upload)

# ---------------- NETWORK DRAW ----------------
def draw_network(G, title, color):
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color,
        node_size=900,
        font_size=8,
        ax=ax
    )
    ax.set_title(title)
    return fig

# ---------------- UI ----------------
st.title("🧬 Full DEG Analysis Toolkit (Real Biological Networks)")

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
neg_fc = st.slider("Negative logFC (≤)", -10, -1, -1)
pos_fc = st.slider("Positive logFC (≥)", 1, 10, 1)
p_cut = st.slider("p-value cutoff", 0.0001, 0.1, 0.05)

filtered = df[
    ((df[fc_col] <= neg_fc) | (df[fc_col] >= pos_fc)) &
    (df[p_col] <= p_cut)
]

if filtered.empty:
    st.warning("No genes passed filters")
    st.stop()

genes = filtered[gene_col].astype(str).tolist()

# ---------------- STRING PPI ----------------
st.subheader("STRING Protein–Protein Interaction Network")

score = st.slider("STRING confidence score", 400, 900, 700)
ppi_edges = fetch_string_ppi(genes[:100], score)

if ppi_edges:
    G_ppi = nx.Graph()
    G_ppi.add_edges_from(ppi_edges)
    st.pyplot(draw_network(G_ppi, "STRING PPI Network", "#ff7f0e"))
else:
    st.info("No STRING interactions found or API unavailable")

# ---------------- TRRUST TF NETWORK ----------------
st.subheader("TF–Gene Regulatory Network (TRRUST)")

trrust = load_trrust()
tf_edges = trrust[trrust["Target"].isin(genes)][["TF", "Target"]].values.tolist()

if tf_edges:
    G_tf = nx.DiGraph()
    G_tf.add_edges_from(tf_edges)
    st.pyplot(draw_network(G_tf, "TRRUST TF–Gene Network", "#1f77b4"))
else:
    st.info("No TF interactions found in TRRUST")

# ---------------- miRTarBase ----------------
st.subheader("miRNA–mRNA Regulatory Network")

mir_upload = st.file_uploader(
    "Upload miRTarBase file (optional)", key="mir"
)

mir_df = load_mirtarbase(mir_upload)
if mir_df is not None:
    cols = mir_df.columns.tolist()
    mir = mir_df[mir_df[cols[1]].isin(genes)]
    edges = mir[[cols[0], cols[1]]].values.tolist()
    G_mir = nx.DiGraph()
    G_mir.add_edges_from(edges)
    st.pyplot(draw_network(G_mir, "miRNA–mRNA Network", "#2ca02c"))
else:
    st.info("Upload miRTarBase file to enable miRNA network")

# ---------------- FUNCTIONAL ENRICHMENT ----------------
st.subheader("Functional Enrichment (gProfiler)")

gp = GProfiler(return_dataframe=True)
enrich = gp.profile(organism="hsapiens", query=genes)

st.dataframe(enrich)
st.subheader("KEGG")
st.dataframe(enrich[enrich["source"] == "KEGG"])
st.subheader("GO:BP")
st.dataframe(enrich[enrich["source"] == "GO:BP"])
st.subheader("GO:MF")
st.dataframe(enrich[enrich["source"] == "GO:MF"])
st.subheader("GO:CC")
st.dataframe(enrich[enrich["source"] == "GO:CC"])

st.success("Full DEG analysis with real biological networks completed ✅")
