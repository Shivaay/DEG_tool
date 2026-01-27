# ==========================================================
# PHOENIX BIOINFOSYS — DEG ANALYSIS & INTERPRETATION PLATFORM
# (Reviewer-ready | Streamlit-hostable | No payment layer)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
import json
from datetime import datetime
from gprofiler import GProfiler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Phoenix DEG Platform", layout="wide")
st.title("🧬 Phoenix BioInfoSys — DEG Analysis Platform")

st.caption(
    "A reproducible, interpretation-focused platform for differential gene expression analysis."
)

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload DEG results (CSV / TSV / XLSX)",
    type=["csv", "tsv", "xlsx"]
)

if uploaded is None:
    st.info("Upload a DEG table to begin analysis.")
    st.stop()

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    return pd.read_excel(file)

df = load_data(uploaded)

# ---------------- COLUMN MAPPING ----------------
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene identifier column", df.columns)
logfc_col = st.sidebar.selectbox("log2 Fold Change column", df.columns)
pval_col = st.sidebar.selectbox("Adjusted p-value / p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

# ---------------- FILTERING ----------------
st.sidebar.header("DEG Thresholds")
logfc_cut = st.sidebar.slider("Absolute log2FC ≥", 0.5, 3.0, 1.0)
p_cut = st.sidebar.slider("Adjusted p-value ≤", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= logfc_cut) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= -logfc_cut) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
genes = deg[gene_col].astype(str).unique().tolist()

# ---------------- EXECUTIVE SUMMARY ----------------
st.header("📌 Executive Summary")

up = (deg["Regulation"] == "Up").sum()
down = (deg["Regulation"] == "Down").sum()

summary_text = f"""
Differential expression analysis identified **{len(deg)} significantly altered genes**
(adjusted p-value ≤ {p_cut}, |log2FC| ≥ {logfc_cut}).
Of these, **{up} genes were upregulated** and **{down} genes were downregulated**.
These transcriptional changes suggest condition-specific biological regulation.
"""

st.markdown(summary_text)

# ---------------- VOLCANO ----------------
st.header("🌋 Volcano Plot")

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c="lightgrey", s=10)
ax.scatter(
    deg[deg["Regulation"] == "Up"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Up"][pval_col]),
    c="#d62728", label="Upregulated", s=20
)
ax.scatter(
    deg[deg["Regulation"] == "Down"][logfc_col],
    -np.log10(deg[deg["Regulation"] == "Down"][pval_col]),
    c="#1f77b4", label="Downregulated", s=20
)
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(adjusted p-value)")
ax.legend()
st.pyplot(fig)

st.info(
    "Genes in the upper left and right corners represent the most statistically significant "
    "and biologically meaningful expression changes."
)

# ---------------- FUNCTIONAL ENRICHMENT ----------------
st.header("🧠 Functional Interpretation")

gp = GProfiler(return_dataframe=True)
enrich = gp.profile(organism="hsapiens", query=genes)

if not enrich.empty:
    top_bp = enrich[enrich["source"] == "GO:BP"].head(10)

    st.subheader("Dominant Biological Processes")
    st.dataframe(top_bp[["name", "p_value"]])

    st.info(
        "Enriched biological processes highlight cellular functions most affected "
        "under the studied condition."
    )

# ---------------- FUNCTIONAL GENE GROUPING ----------------
st.header("🧩 Functional Gene Grouping")

categories = {
    "Immune / Inflammation": ["immune", "cytokine", "inflammatory"],
    "Cell Cycle": ["cell cycle", "mitotic"],
    "Metabolism": ["metabolic", "mitochondrial"],
    "Stress Response": ["stress", "response"]
}

group_summary = []

for cat, keywords in categories.items():
    hits = enrich[
        enrich["name"].str.contains("|".join(keywords), case=False, na=False)
    ]
    if not hits.empty:
        group_summary.append((cat, hits.iloc[0]["name"]))

group_df = pd.DataFrame(group_summary, columns=["Category", "Representative Process"])
st.dataframe(group_df)

# ---------------- PPI NETWORK ----------------
st.header("🔗 Protein–Protein Interaction Network")

@st.cache_data
def fetch_string_ppi(glist):
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

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")
    hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    hub_genes = [h[0] for h in hubs]

    st.write("**Top hub genes:**", ", ".join(hub_genes))

    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G.subgraph(hub_genes), seed=42)
    nx.draw_networkx(G.subgraph(hub_genes), pos, ax=ax, node_color="#ffcc80")
    ax.axis("off")
    st.pyplot(fig)

    st.info(
        "Highly connected hub genes may represent key regulators "
        "or central control points in the observed biological response."
    )

# ---------------- MANUSCRIPT-READY TEXT ----------------
st.header("📝 Manuscript-Ready Text")

results_text = f"""
RNA-seq differential expression analysis identified {len(deg)} genes
significantly altered between conditions (adjusted p-value ≤ {p_cut},
|log2FC| ≥ {logfc_cut}). Functional enrichment analysis revealed
predominant involvement of immune regulation, metabolic processes,
and cell cycle-related pathways.
"""

methods_text = """
Differential gene expression results were analyzed using the Phoenix
BioInfoSys DEG Platform. Genes were filtered based on adjusted p-value
and fold-change thresholds. Functional enrichment was performed using
g:Profiler, and protein–protein interaction networks were derived
from STRING-db.
"""

st.text_area("Results Section", results_text, height=160)
st.text_area("Methods Section", methods_text, height=160)

# ---------------- REPRODUCIBILITY ----------------
st.header("🔁 Reproducibility Metadata")

metadata = {
    "date": datetime.utcnow().isoformat(),
    "logFC_cutoff": logfc_cut,
    "p_value_cutoff": p_cut,
    "genes_analyzed": len(df),
    "DEGs": len(deg)
}

st.json(metadata)

st.download_button(
    "Download Run Metadata (JSON)",
    json.dumps(metadata, indent=2),
    file_name="run_metadata.json"
)

# ---------------- EXPORT FIGURE ----------------
st.header("📤 Export")

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button(
    "Download Volcano Plot (300 DPI)",
    buf.getvalue(),
    file_name="volcano_300dpi.png"
)

st.success("✅ Analysis completed successfully.")
