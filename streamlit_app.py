# ==========================================================
# PhoenixBioInfoSys DEG Platform (Extended – No Base Changes)
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

# ---------- OPTIONAL IMPORTS ----------
try:
    import mygene
except ImportError:
    mygene = None

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — Differential Expression Analysis")

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
up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()
genes = deg[gene_col].astype(str).tolist()

# ==================================================
# 3️⃣ VOLCANO PLOT + DOWNLOAD
# ==================================================
st.header("📊 Volcano Plot")

up_color = st.color_picker("Upregulated color", "#d62728")
down_color = st.color_picker("Downregulated color", "#1f77b4")
bg_color = st.color_picker("Background color", "#cccccc")

fig_vol, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c=bg_color, s=10)
ax.scatter(deg[deg["Regulation"]=="Up"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),
           c=up_color, label="Up")
ax.scatter(deg[deg["Regulation"]=="Down"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),
           c=down_color, label="Down")
ax.legend()
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(p-value)")
st.pyplot(fig_vol)

fig_vol.savefig("volcano_300dpi.png", dpi=300, bbox_inches="tight")
st.download_button("Download Volcano (300 DPI)", open("volcano_300dpi.png","rb"),
                   file_name="volcano_300dpi.png")

# ==================================================
# 4️⃣ ENRICHMENT (GO & KEGG)
# ==================================================
st.header("🧠 Functional Enrichment")

gp = GProfiler(return_dataframe=True)

@st.cache_data
def enrich(g):
    return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

en_all = enrich(genes)
en_up = enrich(up_genes)
en_down = enrich(down_genes)

def show_enrich(df, title):
    st.subheader(title)
    for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
        sub = df[df["source"] == src]
        if not sub.empty:
            st.markdown(f"**{src}**")
            st.dataframe(sub)
            st.download_button(
                f"Download {title} {src}",
                sub.to_csv(index=False).encode(),
                file_name=f"{title}_{src}.csv"
            )

show_enrich(en_all, "All_DEGs")
show_enrich(en_up, "Upregulated")
show_enrich(en_down, "Downregulated")

# ==================================================
# 5️⃣ PPI NETWORK + HUB GENE SELECTION
# ==================================================
st.header("🔗 PPI Network")

st.sidebar.header("PPI Settings")
top_n_hubs = st.sidebar.selectbox("Top hub genes", [10, 20, 50], 0)
node_color = st.color_picker("Node color", "#8da0cb")
edge_color = st.color_picker("Edge color", "#555555")

@st.cache_data
def fetch_ppi(glist):
    r = requests.post(
        "https://string-db.org/api/tsv/network",
        data={"identifiers":"%0d".join(glist[:200]),
              "species":9606,
              "required_score":700}
    )
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text), sep="\t")

ppi = fetch_ppi(genes)
hub_table = pd.DataFrame()
hub_genes = []

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")
    deg_df = pd.DataFrame(G.degree(), columns=["Gene", "Degree"])
    hub_table = deg_df.sort_values("Degree", ascending=False).head(top_n_hubs)
    hub_genes = hub_table["Gene"].tolist()

    subG = G.subgraph(hub_genes)
    pos = nx.spring_layout(subG, seed=42)

    fig_ppi, ax_ppi = plt.subplots(figsize=(8, 6))
    nx.draw_networkx(
        subG, pos,
        node_color=node_color,
        edge_color=edge_color,
        node_size=900,
        font_size=8,
        ax=ax_ppi
    )
    ax_ppi.axis("off")
    st.pyplot(fig_ppi)

    fig_ppi.savefig("ppi_300dpi.png", dpi=300, bbox_inches="tight")
    st.download_button("Download PPI (300 DPI)", open("ppi_300dpi.png","rb"),
                       file_name="ppi_300dpi.png")

# ==================================================
# 6️⃣ HUB GENE TABLE + DOWNLOAD
# ==================================================
st.header("⭐ Hub Genes")

if not hub_table.empty:
    st.dataframe(hub_table)
    st.download_button(
        "Download Hub Gene Table",
        hub_table.to_csv(index=False).encode(),
        file_name="hub_genes.csv"
    )

# ==================================================
# 7️⃣ HEATMAP + DOWNLOAD
# ==================================================
st.header("🔥 Heatmap")

expr_cols = [c for c in df.columns if c not in
             [gene_col, logfc_col, pval_col, "Regulation"]
             and pd.api.types.is_numeric_dtype(df[c])]

if hub_genes and expr_cols:
    mat = df[df[gene_col].isin(hub_genes)].set_index(gene_col)[expr_cols]
    fig_hm, ax_hm = plt.subplots(figsize=(8, 6))
    sns.heatmap(mat, cmap="RdBu_r", center=0, ax=ax_hm)
    st.pyplot(fig_hm)

    fig_hm.savefig("heatmap_300dpi.png", dpi=300, bbox_inches="tight")
    st.download_button("Download Heatmap (300 DPI)", open("heatmap_300dpi.png","rb"),
                       file_name="heatmap_300dpi.png")

# ==================================================
# 8️⃣ REPRODUCIBILITY
# ==================================================
st.header("🔁 Reproducibility")

st.json({
    "timestamp": datetime.utcnow().isoformat(),
    "DEGs": len(deg),
    "Up": len(up_genes),
    "Down": len(down_genes),
    "Top hub genes": hub_genes
})
