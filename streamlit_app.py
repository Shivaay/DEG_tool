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

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="PhoenixBioInfoSys – DEG & Network Analysis"
)

st.title("PhoenixBioInfoSys – Differential Expression & Network Biology")

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
file = st.file_uploader(
    "Upload DEG / Expression file (CSV / TSV / XLSX, ≤1GB)",
    type=["csv", "tsv", "xlsx"]
)

if not file:
    st.stop()

# -------------------------------------------------------
# FILE READ
# -------------------------------------------------------
if file.name.endswith(".csv"):
    df = pd.read_csv(file)
elif file.name.endswith(".tsv"):
    df = pd.read_csv(file, sep="\t")
else:
    df = pd.read_excel(file)

st.subheader("Data Preview")
st.dataframe(df.head())

# -------------------------------------------------------
# COLUMN MAPPING (FIXES COLUMN ERRORS)
# -------------------------------------------------------
st.sidebar.header("Column Mapping")

gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("log2FC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df = df.rename(columns={
    gene_col: "gene",
    logfc_col: "logFC",
    pval_col: "pvalue"
})

# -------------------------------------------------------
# SAFE TYPE COERCION (CRASH FIX)
# -------------------------------------------------------
df["gene"] = df["gene"].astype(str)
df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
df = df.dropna(subset=["gene", "logFC", "pvalue"])

# -------------------------------------------------------
# FILTER CONTROLS
# -------------------------------------------------------
st.sidebar.header("Filtering Thresholds")

pos_fc = st.sidebar.slider("Positive logFC", 1, 10, 2)
neg_fc = st.sidebar.slider("Negative logFC", -10, -1, -2)
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

# -------------------------------------------------------
# COLOR PALETTES
# -------------------------------------------------------
st.sidebar.header("Color Palettes")

volcano_palette = st.sidebar.selectbox(
    "Volcano plot palette",
    ["Set1", "Set2", "Dark2", "coolwarm"]
)

heatmap_palette = st.sidebar.selectbox(
    "Heatmap palette",
    ["vlag", "coolwarm", "RdBu_r", "viridis"]
)

ppi_palette = st.sidebar.selectbox(
    "PPI palette",
    ["Reds", "Oranges", "Spectral", "plasma"]
)

# -------------------------------------------------------
# DEG CLASSIFICATION
# -------------------------------------------------------
df["Regulation"] = "NS"
df.loc[(df["logFC"] >= pos_fc) & (df["pvalue"] <= p_cut), "Regulation"] = "Up"
df.loc[(df["logFC"] <= neg_fc) & (df["pvalue"] <= p_cut), "Regulation"] = "Down"

up = df[df["Regulation"] == "Up"]
down = df[df["Regulation"] == "Down"]

st.subheader("DEG Summary")
st.json({
    "Upregulated": len(up),
    "Downregulated": len(down),
    "Non-significant": len(df[df["Regulation"] == "NS"])
})

# -------------------------------------------------------
# VOLCANO PLOT (300 DPI)
# -------------------------------------------------------
st.subheader("Volcano Plot")

fig_v, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(
    x="logFC", y=-np.log10(df["pvalue"]),
    hue="Regulation",
    data=df,
    palette=volcano_palette,
    ax=ax,
    edgecolor=None
)

ax.axvline(pos_fc, linestyle="--")
ax.axvline(neg_fc, linestyle="--")
ax.axhline(-np.log10(p_cut), linestyle="--")
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(p-value)")

st.pyplot(fig_v)

buf = io.BytesIO()
fig_v.savefig(buf, dpi=300, bbox_inches="tight", facecolor="white")
st.download_button("Download Volcano Plot (300 DPI)", buf.getvalue(), "volcano.png")

# -------------------------------------------------------
# HEATMAP (CONDITIONAL, FIXED)
# -------------------------------------------------------
st.subheader("Heatmap")

expr_cols = df.select_dtypes(include=[np.number]).columns.drop(
    ["logFC", "pvalue"], errors="ignore"
)

if len(expr_cols) >= 2:
    heat_data = df.set_index("gene")[expr_cols]

    fig_h = sns.clustermap(
        heat_data,
        z_score=0,
        cmap=heatmap_palette,
        figsize=(8, 8)
    )

    st.pyplot(fig_h.fig)

    buf = io.BytesIO()
    fig_h.fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button("Download Heatmap (300 DPI)", buf.getvalue(), "heatmap.png")
else:
    st.info("Heatmap requires ≥2 numeric expression columns.")

# -------------------------------------------------------
# FUNCTIONAL ENRICHMENT (UP & DOWN)
# -------------------------------------------------------
st.subheader("Functional Enrichment Analysis")

gp = GProfiler(return_dataframe=True)

def run_enrichment(glist, title):
    if len(glist) == 0:
        st.info(f"No genes for {title}")
        return

    res = gp.profile(organism="hsapiens", query=glist)
    if res is None or res.empty:
        st.info(f"No enrichment for {title}")
        return

    st.markdown(f"### {title}")
    for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
        tab = res[res["source"] == src]
        if not tab.empty:
            st.markdown(f"**{src}**")
            st.dataframe(tab)

run_enrichment(up["gene"].unique().tolist(), "Upregulated Genes")
run_enrichment(down["gene"].unique().tolist(), "Downregulated Genes")

# -------------------------------------------------------
# STRING PPI (REAL-TIME, STABLE)
# -------------------------------------------------------
st.subheader("Protein–Protein Interaction Network")

hub_method = st.selectbox("Hub scoring method", ["Degree", "MCC"])
top_n = st.selectbox("Top hub genes", [10, 20, 50])

genes = list(set(up["gene"].tolist() + down["gene"].tolist()))[:200]
G = nx.Graph()

def fetch_string_ppi(glist):
    url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "%0d".join(glist),
        "species": 9606,
        "required_score": 700
    }
    r = requests.post(url, data=params, timeout=30)
    return r.json() if r.status_code == 200 else []

try:
    interactions = fetch_string_ppi(genes)
    for i in interactions:
        G.add_edge(
            i["preferredName_A"],
            i["preferredName_B"],
            weight=i["score"]
        )
except Exception as e:
    st.warning("STRING API unavailable or rate-limited.")
    st.stop()

# -------------------------------------------------------
# HUB SCORING
# -------------------------------------------------------
if hub_method == "Degree":
    scores = dict(G.degree())
else:
    scores = {}
    cliques = list(nx.find_cliques(G))
    for n in G.nodes():
        scores[n] = sum(len(c) for c in cliques if n in c)

hubs = sorted(scores, key=scores.get, reverse=True)[:top_n]

pos = nx.spring_layout(G, seed=42, k=1.2)

fig_p, ax = plt.subplots(figsize=(8, 8))

node_colors = []
for n in G.nodes():
    if n in hubs[:3]:
        node_colors.append("darkred")
    elif n in hubs:
        node_colors.append("orange")
    else:
        node_colors.append("yellow")

nx.draw(
    G, pos,
    node_color=node_colors,
    node_size=[scores.get(n, 1) * 40 for n in G.nodes()],
    with_labels=False,
    ax=ax
)

st.pyplot(fig_p)

buf = io.BytesIO()
fig_p.savefig(buf, dpi=300, bbox_inches="tight")
st.download_button("Download PPI Network (300 DPI)", buf.getvalue(), "ppi.png")

hub_df = pd.DataFrame({
    "Gene": hubs,
    "Score": [scores[h] for h in hubs]
})
st.dataframe(hub_df)

# -------------------------------------------------------
# AUTOMATED PDF REPORT
# -------------------------------------------------------
st.subheader("Automated Manuscript PDF")

if st.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)

    pdf.cell(0, 10, "Differential Expression Summary", ln=True)
    pdf.cell(0, 8, f"Upregulated genes: {len(up)}", ln=True)
    pdf.cell(0, 8, f"Downregulated genes: {len(down)}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Top Hub Genes", ln=True)
    for g in hubs[:10]:
        pdf.cell(0, 7, g, ln=True)

    buf = io.BytesIO()
    pdf.output(buf)
    st.download_button("Download PDF Report", buf.getvalue(), "DEG_Report.pdf")
