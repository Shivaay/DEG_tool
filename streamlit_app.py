# ==========================================================
# FULL DEG ANALYSIS PLATFORM
# BASE PRESERVED + PROFESSIONAL INTERPRETATION EXTENSIONS
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
import os
import json
from datetime import datetime
from gprofiler import GProfiler

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="Full DEG Analysis", layout="wide")
st.title("🧬 Phoenix BioInfoSys — DEG Analysis Platform")

# ---------------- FILE UPLOAD (BASE) ----------------
uploaded = st.file_uploader(
    "Upload DEG results (CSV / TSV / XLSX)",
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
st.success(f"Dataset loaded: {df.shape[0]} genes")

# ---------------- COLUMN MAPPING (BASE) ----------------
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

# ---------------- FILTERING (BASE) ----------------
st.sidebar.header("Filtering")
neg_fc = st.sidebar.slider("Negative logFC (≤)", -10.0, -0.5, -1.0)
pos_fc = st.sidebar.slider("Positive logFC (≥)", 0.5, 10.0, 1.0)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
genes = deg[gene_col].astype(str).unique().tolist()

up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()

# ======================================================
# EXECUTIVE SUMMARY
# ======================================================
st.header("📌 Executive Summary")

st.markdown(
    f"""
A total of **{len(deg)} genes** were identified as differentially expressed
(p ≤ {p_cut}), including **{len(up_genes)} upregulated** and
**{len(down_genes)} downregulated genes**.
These results indicate distinct transcriptional programs associated
with the experimental condition.
"""
)

# ======================================================
# BASE VISUALS (UNCHANGED)
# ======================================================
st.subheader("Volcano Plot")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c="lightgrey", s=10)
ax.scatter(deg[deg["Regulation"] == "Up"][logfc_col],
           -np.log10(deg[deg["Regulation"] == "Up"][pval_col]),
           c="red", s=20, label="Up")
ax.scatter(deg[deg["Regulation"] == "Down"][logfc_col],
           -np.log10(deg[deg["Regulation"] == "Down"][pval_col]),
           c="blue", s=20, label="Down")
ax.legend()
st.pyplot(fig)

# ======================================================
# ENRICHMENT (FULL, SEPARATED — RESTORED)
# ======================================================
st.header("🧠 Functional Enrichment Analysis")

gp = GProfiler(return_dataframe=True)

@st.cache_data
def run_enrichment(gene_list):
    if not gene_list:
        return pd.DataFrame()
    return gp.profile(organism="hsapiens", query=gene_list)

enrich_all = run_enrichment(genes)
enrich_up = run_enrichment(up_genes)
enrich_down = run_enrichment(down_genes)

def show_enrich(title, df, source):
    st.subheader(title)
    subset = df[df["source"] == source]
    if subset.empty:
        st.info("No significant terms detected.")
    else:
        st.dataframe(subset[["name", "p_value", "intersection_size"]])

# ---- GLOBAL ----
show_enrich("GO: Biological Process (All DEGs)", enrich_all, "GO:BP")
show_enrich("GO: Molecular Function (All DEGs)", enrich_all, "GO:MF")
show_enrich("GO: Cellular Component (All DEGs)", enrich_all, "GO:CC")
show_enrich("KEGG Pathways (All DEGs)", enrich_all, "KEGG")

# ---- DIRECTIONAL ----
show_enrich("GO:BP — Upregulated Genes", enrich_up, "GO:BP")
show_enrich("GO:BP — Downregulated Genes", enrich_down, "GO:BP")

# ======================================================
# HUB GENES (BASE LOGIC PRESERVED)
# ======================================================
st.header("🔗 Hub Gene Analysis")

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
    r = requests.post(url, data=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text), sep="\t")

ppi = fetch_string_ppi(genes)
hub_genes = []

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")
    hub_genes = [x[0] for x in sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]]
    st.write("**Top hub genes:**", ", ".join(hub_genes))

# ======================================================
# PROFESSIONAL BIOLOGICAL INTERPRETATION SUMMARY
# ======================================================
st.header("📝 Integrated Biological Interpretation (Manuscript-Ready)")

def summarize_terms(df):
    if df.empty:
        return "No dominant functional categories were detected."
    return ", ".join(df["name"].head(3).tolist())

up_bp = summarize_terms(enrich_up[enrich_up["source"] == "GO:BP"])
down_bp = summarize_terms(enrich_down[enrich_down["source"] == "GO:BP"])

hub_function_text = (
    f"The identified hub genes ({', '.join(hub_genes[:5])}) "
    "are highly connected within the protein–protein interaction network, "
    "suggesting potential roles as key regulatory or signaling molecules "
    "within the observed biological response."
    if hub_genes else
    "No dominant hub genes were identified under the selected thresholds."
)

final_summary = f"""
**Upregulated genes** were predominantly associated with biological processes
such as {up_bp}. This suggests activation of pathways relevant to the
experimental condition.

In contrast, **downregulated genes** were mainly enriched in processes
including {down_bp}, indicating potential suppression of these biological
functions.

{hub_function_text}

**Overall conclusion:**  
The combined differential expression, functional enrichment, and network
analyses indicate coordinated transcriptional reprogramming, involving
activation and repression of distinct biological pathways, consistent with
a condition-specific molecular response.
"""

st.text_area("Manuscript-Ready Summary", final_summary, height=320)

# ======================================================
# REPRODUCIBILITY
# ======================================================
st.header("🔁 Reproducibility Metadata")

metadata = {
    "timestamp": datetime.utcnow().isoformat(),
    "p_value_cutoff": p_cut,
    "logFC_positive": pos_fc,
    "logFC_negative": neg_fc,
    "total_genes": len(df),
    "DEGs": len(deg),
    "hub_genes": hub_genes
}

st.json(metadata)

st.download_button(
    "Download Run Metadata (JSON)",
    json.dumps(metadata, indent=2),
    file_name="run_metadata.json"
)

st.success("✅ Analysis completed successfully.")
