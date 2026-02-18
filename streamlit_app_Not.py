import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
from gprofiler import GProfiler
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys")

st.title("PhoenixBioInfoSys – DEG, Enrichment & Network Analysis")

# ============================================================
# GLOBAL SAFETY SETTINGS
# ============================================================
MAX_STRING_GENES = 200
MAX_MCC_GENES = 100
STRING_TIMEOUT = 25

# ============================================================
# FILE UPLOAD
# ============================================================
file = st.file_uploader(
    "Upload DEG / Expression file (CSV / TSV / XLSX)",
    type=["csv", "tsv", "xlsx"]
)

if not file:
    st.stop()

# ============================================================
# READ FILE (SAFE)
# ============================================================
if file.name.endswith(".csv"):
    df = pd.read_csv(file)
elif file.name.endswith(".tsv"):
    df = pd.read_csv(file, sep="\t")
else:
    df = pd.read_excel(file)

st.subheader("Data Preview")
st.dataframe(df.head())

# ============================================================
# COLUMN MAPPING
# ============================================================
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("log2FC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df = df.rename(columns={
    gene_col: "gene",
    logfc_col: "logFC",
    pval_col: "pvalue"
})

df["gene"] = df["gene"].astype(str)
df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
df = df.dropna(subset=["gene", "logFC", "pvalue"])

# ============================================================
# FILTERS
# ============================================================
st.sidebar.header("Thresholds")
pos_fc = st.sidebar.slider("Positive logFC", 1, 10, 2)
neg_fc = st.sidebar.slider("Negative logFC", -10, -1, -2)
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

df["Regulation"] = "NS"
df.loc[(df["logFC"] >= pos_fc) & (df["pvalue"] <= p_cut), "Regulation"] = "Up"
df.loc[(df["logFC"] <= neg_fc) & (df["pvalue"] <= p_cut), "Regulation"] = "Down"

up = df[df["Regulation"] == "Up"]
down = df[df["Regulation"] == "Down"]

st.subheader("DEG Summary")
st.json({
    "Up": len(up),
    "Down": len(down),
    "NS": len(df[df["Regulation"] == "NS"])
})

# ============================================================
# VOLCANO (LIGHTWEIGHT – AUTO SAFE)
# ============================================================
st.subheader("Volcano Plot")

fig_v, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="logFC",
    y=-np.log10(df["pvalue"]),
    hue="Regulation",
    palette={"Up":"red","Down":"blue","NS":"grey"},
    ax=ax
)
ax.axvline(pos_fc, linestyle="--")
ax.axvline(neg_fc, linestyle="--")
ax.axhline(-np.log10(p_cut), linestyle="--")
st.pyplot(fig_v)

# ============================================================
# HEATMAP (LAZY + SAFE)
# ============================================================
st.subheader("Heatmap")

expr_cols = df.select_dtypes(include=[np.number]).columns.drop(
    ["logFC","pvalue"], errors="ignore"
)

if st.button("Generate Heatmap"):
    if len(expr_cols) < 2:
        st.info("Heatmap requires ≥2 expression columns.")
    else:
        with st.spinner("Generating heatmap..."):
            cg = sns.clustermap(
                df.set_index("gene")[expr_cols],
                z_score=0,
                cmap="vlag",
                figsize=(8,8)
            )
            st.pyplot(cg.fig)

# ============================================================
# ENRICHMENT (LAZY + CACHED)
# ============================================================
st.subheader("Functional Enrichment")

gp = GProfiler(return_dataframe=True)

@st.cache_data
def run_enrichment_cached(glist):
    return gp.profile(organism="hsapiens", query=glist)

if st.button("Run Enrichment"):
    with st.spinner("Running enrichment..."):
        for label, genes in [("Upregulated", up), ("Downregulated", down)]:
            if len(genes) == 0:
                continue
            res = run_enrichment_cached(genes["gene"].tolist())
            if res is not None and not res.empty:
                st.markdown(f"### {label}")
                for src in ["GO:BP","GO:MF","GO:CC","KEGG"]:
                    tab = res[res["source"]==src]
                    if not tab.empty:
                        st.markdown(f"**{src}**")
                        st.dataframe(tab)

# ============================================================
# STRING PPI (FULLY SAFE)
# ============================================================
st.subheader("PPI Network")

hub_method = st.selectbox("Hub method", ["Degree","MCC"])
top_n = st.selectbox("Top hubs", [10,20,50])

genes = list(set(up["gene"].tolist()+down["gene"].tolist()))
genes = genes[:MAX_STRING_GENES]

@st.cache_data
def fetch_string_ppi_safe(glist):
    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers":"%0d".join(glist),
        "species":9606,
        "required_score":700
    }
    r = requests.post(url, data=params, timeout=STRING_TIMEOUT)
    return r.json() if r.status_code==200 else []

if st.button("Generate PPI Network"):
    with st.spinner("Fetching STRING interactions..."):
        interactions = fetch_string_ppi_safe(genes)

    if not interactions:
        st.warning("No STRING interactions returned.")
    else:
        G = nx.Graph()
        for i in interactions:
            G.add_edge(i["preferredName_A"], i["preferredName_B"])

        # MCC SAFETY
        if hub_method=="MCC" and G.number_of_nodes()>MAX_MCC_GENES:
            st.warning("MCC auto-switched to Degree (network too large).")
            hub_method="Degree"

        if hub_method=="Degree":
            scores = dict(G.degree())
        else:
            scores = {}
            cliques = list(nx.find_cliques(G))
            for n in G.nodes():
                scores[n] = sum(len(c) for c in cliques if n in c)

        hubs = sorted(scores, key=scores.get, reverse=True)[:top_n]

        pos = nx.spring_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(8,8))
        nx.draw(G, pos, node_size=50, with_labels=False, ax=ax)
        st.pyplot(fig)

# ============================================================
# PDF REPORT (LAZY)
# ============================================================
st.subheader("PDF Report")

if st.button("Generate PDF"):
    with st.spinner("Generating PDF..."):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        pdf.cell(0,10,"DEG Summary",ln=True)
        pdf.cell(0,8,f"Up: {len(up)}",ln=True)
        pdf.cell(0,8,f"Down: {len(down)}",ln=True)
        buf = io.BytesIO()
        pdf.output(buf)
        st.download_button("Download PDF", buf.getvalue(), "report.pdf")
