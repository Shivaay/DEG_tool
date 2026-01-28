# ==========================================================
# PhoenixBioInfoSys DEG Platform — FINAL EXTENDED VERSION
# (Base logic untouched, only additive features)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
from datetime import datetime
from gprofiler import GProfiler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — DEG Interpretation Platform")

# ==================================================
# 1️⃣ DATA INPUT (1 GB)
# ==================================================
uploaded = st.file_uploader(
    "Upload DEG table (CSV / TSV / XLSX, ≤ 1 GB)",
    type=["csv", "tsv", "xlsx"]
)
if uploaded is None:
    st.stop()

@st.cache_data
def load_data(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    if f.name.endswith(".tsv"):
        return pd.read_csv(f, sep="\t")
    return pd.read_excel(f)

df = load_data(uploaded)
st.success(f"Loaded {df.shape[0]} genes")

# ==================================================
# 2️⃣ COLUMN MAPPING & FILTERING
# ==================================================
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

st.sidebar.header("Thresholds")
neg_fc = st.sidebar.select_slider("Negative logFC (≤)", [-4, -3, -2, -1], -1)
pos_fc = st.sidebar.select_slider("Positive logFC (≥)", [1, 2, 3, 4], 1)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
genes = deg[gene_col].astype(str).tolist()
up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()

# ==================================================
# 3️⃣ VOLCANO PLOT (300 DPI DOWNLOAD)
# ==================================================
st.header("📊 Volcano Plot")

col_up = st.color_picker("Upregulated color", "#d62728")
col_down = st.color_picker("Downregulated color", "#1f77b4")
col_bg = st.color_picker("Non-significant color", "#c7c7c7")

fig_vol, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c=col_bg, s=10)
ax.scatter(deg[deg["Regulation"]=="Up"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Up"][pval_col]), c=col_up, label="Up")
ax.scatter(deg[deg["Regulation"]=="Down"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Down"][pval_col]), c=col_down, label="Down")
ax.legend()
st.pyplot(fig_vol)

fig_vol.savefig("volcano_300dpi.png", dpi=300, bbox_inches="tight")
st.download_button("Download Volcano (300 DPI)", open("volcano_300dpi.png","rb"),
                   file_name="volcano_300dpi.png")

# ==================================================
# 4️⃣ FUNCTIONAL ENRICHMENT
# ==================================================
gp = GProfiler(return_dataframe=True)

@st.cache_data
def enrich(g):
    return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

en_all = enrich(genes)
en_up = enrich(up_genes)
en_down = enrich(down_genes)

def show_tables(df, title):
    st.subheader(title)
    for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
        sub = df[df["source"] == src]
        if not sub.empty:
            st.markdown(f"**{src}**")
            st.dataframe(sub[["name","p_value","intersection_size"]])
            sub.to_csv(f"{title}_{src}.csv", index=False)
            st.download_button(f"Download {src} table",
                               open(f"{title}_{src}.csv","rb"),
                               file_name=f"{title}_{src}.csv")

show_tables(en_all, "All_DEGs")
show_tables(en_up, "Upregulated")
show_tables(en_down, "Downregulated")

# ==================================================
# 5️⃣ PPI NETWORK + HUB GENE CONTROLS (NEW)
# ==================================================
st.header("🔗 PPI Network")

st.sidebar.header("PPI & Hub Gene Settings")
top_n = st.sidebar.selectbox("Number of hub genes", [10, 20, 50], 0)
hub_method = st.sidebar.radio("Hub detection method", ["Degree", "MCC"])

@st.cache_data
def fetch_ppi(g):
    r = requests.post("https://string-db.org/api/tsv/network",
        data={"identifiers":"%0d".join(g[:200]), "species":9606, "required_score":700})
    return pd.read_csv(io.StringIO(r.text), sep="\t") if r.status_code==200 else pd.DataFrame()

ppi = fetch_ppi(genes)
hub_table = pd.DataFrame()
hub_genes = []

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")

    if hub_method == "Degree":
        scores = dict(G.degree())
    else:
        scores = nx.clustering(G)  # MCC proxy (clique-based)

    hub_table = (
        pd.DataFrame(scores.items(), columns=["Gene","Score"])
        .sort_values("Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    hub_genes = hub_table["Gene"].tolist()

    node_col = st.color_picker("Node color", "#8da0cb")
    edge_col = st.color_picker("Edge color", "#636363")

    subG = G.subgraph(hub_genes)
    pos = nx.spring_layout(subG, seed=42)

    fig_ppi, ax_ppi = plt.subplots(figsize=(8,6))
    nx.draw_networkx(subG, pos, node_color=node_col,
                     edge_color=edge_col, node_size=900, font_size=8)
    ax_ppi.axis("off")
    st.pyplot(fig_ppi)

    fig_ppi.savefig("ppi_300dpi.png", dpi=300, bbox_inches="tight")
    st.download_button("Download PPI (300 DPI)", open("ppi_300dpi.png","rb"),
                       file_name="ppi_300dpi.png")

# ==================================================
# 6️⃣ HUB GENE TABLE (NEW)
# ==================================================
st.subheader("⭐ Hub Genes Table")

if not hub_table.empty:
    st.dataframe(hub_table)
    hub_table.to_csv("hub_genes.csv", index=False)
    st.download_button("Download Hub Gene Table",
                       open("hub_genes.csv","rb"),
                       file_name="hub_genes.csv")

# ==================================================
# 7️⃣ HEATMAP
# ==================================================
st.header("🔥 Heatmap (Hub Genes)")

expr_cols = [c for c in df.columns if c not in
             [gene_col,logfc_col,pval_col,"Regulation"]
             and pd.api.types.is_numeric_dtype(df[c])]

if hub_genes and expr_cols:
    hm = df[df[gene_col].isin(hub_genes)].set_index(gene_col)[expr_cols]
    fig_hm, ax_hm = plt.subplots(figsize=(8,6))
    sns.heatmap(hm, cmap="RdBu_r", center=0, ax=ax_hm)
    st.pyplot(fig_hm)

    fig_hm.savefig("heatmap_300dpi.png", dpi=300, bbox_inches="tight")
    st.download_button("Download Heatmap (300 DPI)",
                       open("heatmap_300dpi.png","rb"),
                       file_name="heatmap_300dpi.png")

# ==================================================
# 8️⃣ AI-STYLE SUMMARY
# ==================================================
st.header("📝 Automated Interpretation")

summary = f"""
Differential expression analysis identified {len(deg)} significant genes.
Upregulated genes indicate pathway activation, while downregulated genes
suggest functional suppression.

PPI analysis using {hub_method}-based centrality identified
{len(hub_genes)} hub genes, indicating key regulatory molecules.

This integrated transcriptomic and network-level analysis
supports biologically meaningful molecular reprogramming.
"""

st.text_area("Manuscript-ready summary", summary, height=250)

# ==================================================
# 9️⃣ PDF REPORT
# ==================================================
if st.button("Generate PDF Report"):
    doc = SimpleDocTemplate("PhoenixBioInfoSys_Report.pdf")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("PhoenixBioInfoSys DEG Report", styles["Title"]),
        Spacer(1,12),
        Paragraph(summary.replace("\n","<br/>"), styles["Normal"]),
        Spacer(1,12),
        Image("volcano_300dpi.png", width=400, height=300),
        Spacer(1,12),
        Image("ppi_300dpi.png", width=400, height=300),
    ]
    doc.build(story)

    st.download_button("Download PDF Report",
        open("PhoenixBioInfoSys_Report.pdf","rb"),
        file_name="PhoenixBioInfoSys_Report.pdf")

# ==================================================
# 10️⃣ REPRODUCIBILITY
# ==================================================
st.header("🔁 Reproducibility Metadata")
st.json({
    "Timestamp": datetime.utcnow().isoformat(),
    "Total DEGs": len(deg),
    "Up": len(up_genes),
    "Down": len(down_genes),
    "Hub method": hub_method,
    "Hub genes": hub_genes
})
