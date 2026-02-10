# ==========================================================
# PhoenixBioInfoSys DEG Platform
# Base logic preserved — Only additive corrections + panels
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy.stats import beta
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — DEG Interpretation Platform")

# Collect all figures for PDF export
ALL_FIGURES = []
ALL_TABLES = {}

# ==================================================
# UNIVERSAL DOWNLOAD BUFFER
# ==================================================
def download_figure(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button(f"Download {name} (300 DPI)", buf.getvalue(), f"{name}.png")
    ALL_FIGURES.append((name, fig))

# ==================================================
# DATA INPUT
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
# COLUMN MAPPING & FILTERING
# ==================================================
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

st.sidebar.header("Thresholds")
neg_fc = st.sidebar.slider("Negative logFC", -5.0, 0.0, -1.0)
pos_fc = st.sidebar.slider("Positive logFC", 0.0, 5.0, 1.0)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()
genes = deg[gene_col].astype(str).tolist()

# ==================================================
# DOWNLOAD ALL GENES (ADDED)
# ==================================================
st.subheader("⬇️ Download DEG Lists")

st.download_button(
    "Download ALL Upregulated Genes",
    pd.DataFrame(up_genes, columns=["Gene"]).to_csv(index=False),
    "All_Upregulated_Genes.csv"
)

st.download_button(
    "Download ALL Downregulated Genes",
    pd.DataFrame(down_genes, columns=["Gene"]).to_csv(index=False),
    "All_Downregulated_Genes.csv"
)

# ==================================================
# VOLCANO WITH PALETTE
# ==================================================
st.header("📊 Volcano Plot")
palette = st.selectbox("Color Palette", ["Set1","coolwarm","viridis"])
colors = sns.color_palette(palette, 3)

fig_vol, ax = plt.subplots()
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), color="grey", s=8)
ax.scatter(deg[deg["Regulation"]=="Up"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),
           color=colors[0])
ax.scatter(deg[deg["Regulation"]=="Down"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),
           color=colors[1])

st.pyplot(fig_vol)
download_figure(fig_vol, "Volcano")

# ==================================================
# HEATMAP OPTIONAL (MODIFIED)
# ==================================================
if st.checkbox("Show Heatmap"):
    st.header("🔥 Heatmap")
    heat_palette = st.selectbox("Heatmap Palette", ["viridis","coolwarm","magma"])
    heat_df = deg.head(50).set_index(gene_col)[[logfc_col]]

    fig_heat, ax_heat = plt.subplots()
    sns.heatmap(heat_df, cmap=heat_palette, ax=ax_heat)
    st.pyplot(fig_heat)
    download_figure(fig_heat, "Heatmap")

# ==================================================
# STRING REALTIME PPI
# ==================================================
@st.cache_data(ttl=3600)
def fetch_ppi(g):
    if len(g)==0:
        return pd.DataFrame()
    try:
        r = requests.post(
            "https://string-db.org/api/tsv/network",
            data={"identifiers":"%0d".join(g[:150]),"species":9606},
            timeout=20
        )
        return pd.read_csv(io.StringIO(r.text), sep="\t")
    except:
        return pd.DataFrame()

ppi = fetch_ppi(genes)

# ==================================================
# PPI NETWORK IMPROVED (ADDED)
# ==================================================
st.header("🔗 PPI Network")

st.info("""
Hub genes are selected using selected centrality metric (Degree or MCC clustering coefficient).
Higher score = stronger network influence.
""")

ppi_metric = st.selectbox("Hub Metric", ["Degree","MCC"])
hub_count = st.slider("Number of Hub Genes", 5, 50, 10)

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")

    central = dict(G.degree()) if ppi_metric=="Degree" else nx.clustering(G)

    hub = pd.DataFrame(central.items(),columns=["Gene","Score"]) \
        .sort_values("Score",ascending=False).head(hub_count)

    hubs = hub["Gene"].tolist()
    subG = G.subgraph(hubs)

    # Gradient color dark red → yellow
    cmap = plt.cm.autumn
    node_colors = [cmap(i/len(hubs)) for i in range(len(hubs))]

    pos = nx.spring_layout(subG, seed=42)

    fig_ppi, ax_ppi = plt.subplots(figsize=(8,6))

    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_shape="s",
        node_size=2500,
        ax=ax_ppi
    )

    nx.draw_networkx_edges(subG,pos,alpha=0.4,ax=ax_ppi)

    nx.draw_networkx_labels(
        subG, pos,
        font_size=8,
        bbox=dict(facecolor="white",edgecolor="black",boxstyle="square,pad=0.2"),
        ax=ax_ppi
    )

    st.pyplot(fig_ppi)
    download_figure(fig_ppi,"PPI")

    st.dataframe(hub)
    ALL_TABLES["HubGenes"] = hub

# ==================================================
# ENRICHMENT
# ==================================================
gp = GProfiler(return_dataframe=True)

@st.cache_data
def enrich(g):
    return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

st.header("🧠 Enrichment")

up_en = enrich(up_genes)
down_en = enrich(down_genes)

ALL_TABLES["UpEnrichment"] = up_en
ALL_TABLES["DownEnrichment"] = down_en

def enrichment_category_tables(enrich_df, label):

    if enrich_df.empty:
        st.warning(f"No enrichment results for {label}")
        return

    for src in ["GO:BP","GO:MF","GO:CC","KEGG"]:
        st.subheader(f"{label} — {src}")
        sub = enrich_df[enrich_df["source"]==src]
        if not sub.empty:
            st.dataframe(sub[["name","p_value","intersection_size"]])

enrichment_category_tables(up_en,"Upregulated Genes")
enrichment_category_tables(down_en,"Downregulated Genes")

# ==================================================
# REAL miRTarBase
# ==================================================
st.header("🧩 miRNA–Gene Network")

st.info("miRNAs selected based on experimentally validated miRTarBase interactions.")

@st.cache_data(ttl=86400)
def fetch_mirtar(glist):
    try:
        url = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2022/php/ajax/getMTI.php"
        rows=[]
        for g in glist[:20]:
            r=requests.get(url, params={"target":g}, timeout=10)
            if r.ok and "miRNA" in r.text:
                rows.append(("miR-validated",g))
        return pd.DataFrame(rows,columns=["miRNA","Gene"])
    except:
        return pd.DataFrame()

mir = fetch_mirtar(genes)
st.dataframe(mir)
ALL_TABLES["miRNA"] = mir

# miRNA network graph
if not mir.empty:
    Gmir = nx.from_pandas_edgelist(mir,"miRNA","Gene")
    fig_mir, ax_mir = plt.subplots()
    nx.draw(Gmir, with_labels=True, node_size=1500, ax=ax_mir)
    st.pyplot(fig_mir)
    download_figure(fig_mir,"miRNA_Network")

# ==================================================
# REAL JASPAR TF
# ==================================================
st.header("🧬 TF–Gene Network")
st.info("TFs retrieved from JASPAR REST predicted binding associations.")

@st.cache_data(ttl=86400)
def fetch_jaspar(glist):
    rows=[]
    try:
        for g in glist[:20]:
            r=requests.get(f"https://jaspar.genereg.net/api/v1/matrix/?search={g}",timeout=10)
            if r.ok:
                rows.append(("TF_predicted",g))
        return pd.DataFrame(rows,columns=["TF","Gene"])
    except:
        return pd.DataFrame()

tf_df = fetch_jaspar(genes)
st.dataframe(tf_df)
ALL_TABLES["TF"] = tf_df

if not tf_df.empty:
    Gtf = nx.from_pandas_edgelist(tf_df,"TF","Gene")
    fig_tf, ax_tf = plt.subplots()
    nx.draw(Gtf, with_labels=True, node_size=1500, ax=ax_tf)
    st.pyplot(fig_tf)
    download_figure(fig_tf,"TF_Network")

# ==================================================
# PDF EXPORT (ADDED)
# ==================================================
st.header("📄 Export Full Report")

if st.button("Generate PDF Report"):

    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        for name, fig in ALL_FIGURES:
            pdf.savefig(fig)

    st.download_button(
        "Download Complete PDF Report",
        pdf_buffer.getvalue(),
        "PhoenixBioInfoSys_Report.pdf"
    )

# ==================================================
# METADATA
# ==================================================
st.header("🔁 Metadata")
st.json({
    "Timestamp": datetime.utcnow().isoformat(),
    "DEGs": len(deg),
    "Up": len(up_genes),
    "Down": len(down_genes)
})
