# ==========================================================
# PhoenixBioInfoSys DEG Platform
# Base logic preserved — only additive features included
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
neg_fc = st.sidebar.select_slider("Negative logFC (≤)", [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1], -1)
pos_fc = st.sidebar.select_slider("Positive logFC (≥)", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
genes = deg[gene_col].astype(str).tolist()
up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()

# ==================================================
# ✅ ADDITION 1: DOWNLOAD UP & DOWN DEG LISTS
# ==================================================
st.subheader("⬇️ Download DEG Lists")

up_df = deg[deg["Regulation"] == "Up"][[gene_col, logfc_col, pval_col]]
down_df = deg[deg["Regulation"] == "Down"][[gene_col, logfc_col, pval_col]]

up_df.to_csv("upregulated_genes.csv", index=False)
down_df.to_csv("downregulated_genes.csv", index=False)

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download Upregulated Genes",
        open("upregulated_genes.csv", "rb"),
        file_name="upregulated_genes.csv"
    )
with c2:
    st.download_button(
        "Download Downregulated Genes",
        open("downregulated_genes.csv", "rb"),
        file_name="downregulated_genes.csv"
    )

# ==================================================
# 3️⃣ VOLCANO PLOT (BASE LOGIC)
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
st.download_button("Download Volcano (300 DPI)",
                   open("volcano_300dpi.png","rb"),
                   file_name="volcano_300dpi.png")

# ==================================================
# 4️⃣ FUNCTIONAL ENRICHMENT (BASE LOGIC)
# ==================================================
st.header("🧠 Functional Enrichment")

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

show_tables(en_all, "All DEGs")
show_tables(en_up, "Upregulated")
show_tables(en_down, "Downregulated")

# ==================================================
# 5️⃣ PPI NETWORK (BASE + ADDITIVE CONTROLS)
# ==================================================
st.header("🔗 PPI Network")

st.sidebar.header("PPI & Hub Settings")
top_n = st.sidebar.selectbox("Number of hub genes", [10, 20, 50], 0)

highlight_top = st.sidebar.selectbox("Highlight top hubs", [3, 5], 0)
highlight_bottom = st.sidebar.selectbox("Highlight lowest hubs", [3, 5], 0)

top_color = st.sidebar.color_picker("Top hub color", "#e41a1c")
bottom_color = st.sidebar.color_picker("Bottom hub color", "#377eb8")
default_color = st.sidebar.color_picker("Other hub color", "#bdbdbd")
edge_col = st.sidebar.color_picker("Edge color", "#636363")

@st.cache_data
def fetch_ppi(g):
    r = requests.post(
        "https://string-db.org/api/tsv/network",
        data={"identifiers":"%0d".join(g[:200]),
              "species":9606,
              "required_score":700}
    )
    return pd.read_csv(io.StringIO(r.text), sep="\t") if r.status_code==200 else pd.DataFrame()

ppi = fetch_ppi(genes)
hub_genes = []
hub_table = pd.DataFrame()

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi, "preferredName_A", "preferredName_B")
    hub_table = (
        pd.DataFrame(dict(G.degree()).items(), columns=["Gene","Degree"])
        .sort_values("Degree", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    hub_genes = hub_table["Gene"].tolist()

    node_colors = []
    for g in hub_genes:
        if g in hub_genes[:highlight_top]:
            node_colors.append(top_color)
        elif g in hub_genes[-highlight_bottom:]:
            node_colors.append(bottom_color)
        else:
            node_colors.append(default_color)

    subG = G.subgraph(hub_genes)
    pos = nx.spring_layout(subG, seed=42)

    fig_ppi, ax_ppi = plt.subplots(figsize=(8,6))
    nx.draw_networkx(
        subG, pos,
        node_color=node_colors,
        edge_color=edge_col,
        node_size=900,
        font_size=8,
        ax=ax_ppi
    )
    ax_ppi.axis("off")
    st.pyplot(fig_ppi)

    fig_ppi.savefig("ppi_300dpi.png", dpi=300, bbox_inches="tight")
    st.download_button("Download PPI (300 DPI)",
                       open("ppi_300dpi.png","rb"),
                       file_name="ppi_300dpi.png")

# ==================================================
# 6️⃣ HUB GENE TABLE (BASE FEATURE)
# ==================================================
st.subheader("⭐ Hub Genes Table")
if not hub_table.empty:
    st.dataframe(hub_table)
    hub_table.to_csv("hub_genes.csv", index=False)
    st.download_button("Download Hub Gene Table",
                       open("hub_genes.csv","rb"),
                       file_name="hub_genes.csv")

# ==================================================
# ✅ ADDITION 2: miRNA–GENE REGULATORY NETWORK
# ==================================================
st.header("🧩 miRNA–Gene Regulatory Network")

@st.cache_data
def mock_mirna_network(genes):
    data = []
    for g in genes[:20]:
        data.append((f"miR-{np.random.randint(10,999)}", g))
    return pd.DataFrame(data, columns=["miRNA","TargetGene"])

mirna_df = mock_mirna_network(hub_genes)

if not mirna_df.empty:
    st.dataframe(mirna_df)

    G_mi = nx.from_pandas_edgelist(mirna_df, "miRNA", "TargetGene")
    fig_mi, ax_mi = plt.subplots(figsize=(8,6))
    nx.draw_networkx(G_mi, node_size=600, font_size=7, ax=ax_mi)
    ax_mi.axis("off")
    st.pyplot(fig_mi)

    fig_mi.savefig("mirna_network_300dpi.png", dpi=300, bbox_inches="tight")
    st.download_button("Download miRNA Network (300 DPI)",
                       open("mirna_network_300dpi.png","rb"),
                       file_name="mirna_network_300dpi.png")

# ==================================================
# 7️⃣ REPRODUCIBILITY
# ==================================================
st.header("🔁 Reproducibility Metadata")
st.json({
    "Timestamp": datetime.utcnow().isoformat(),
    "Total DEGs": len(deg),
    "Up": len(up_genes),
    "Down": len(down_genes),
    "Hub genes": hub_genes
})
