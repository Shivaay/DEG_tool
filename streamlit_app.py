# ==========================================================
# ================= PHOENIX BIOINFORMATICS DASHBOARD =========
# =================== UI ENHANCED VERSION ===================
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
from gprofiler import GProfiler
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import beta
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Phoenix Bioinformatics Dashboard",
    layout="wide",
    page_icon="🧬"
)

# ==========================================================
# GLOBAL STORAGE
# ==========================================================
ALL_FIGURES = []
ALL_TABLES = {}

# ==========================================================
# STYLING
# ==========================================================
st.markdown("""
<style>
.main {
    background-color:#0e1117;
}
.block-container {
    padding-top:1rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# UNIVERSAL DOWNLOAD BUFFER
# ==========================================================
def download_figure(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button(f"Download {name}", buf.getvalue(), f"{name}.png")
    ALL_FIGURES.append((name, fig))

# ==========================================================
# HEADER
# ==========================================================
st.title("🧬 Phoenix Multi-Omics DEG Dashboard")
st.caption("Advanced Bioinformatics + Clinical Interpretation Platform")

# ==========================================================
# DATA INPUT PANEL
# ==========================================================
with st.expander("📂 Upload Dataset", expanded=True):

    uploaded = st.file_uploader(
        "Upload DEG table (CSV / TSV / XLSX)",
        type=["csv","tsv","xlsx"]
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

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================
st.sidebar.header("⚙️ DEG Controls")

gene_col = st.sidebar.selectbox("Gene Column", df.columns)
logfc_col = st.sidebar.selectbox("logFC Column", df.columns)
pval_col = st.sidebar.selectbox("p-value Column", df.columns)

neg_fc = st.sidebar.slider("Negative logFC", -5.0,0.0,-1.0)
pos_fc = st.sidebar.slider("Positive logFC", 0.0,5.0,1.0)
p_cut = st.sidebar.slider("p-value Cutoff",0.0001,0.1,0.05)

# ==========================================================
# DEG FILTERING
# ==========================================================
df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col]>=pos_fc)&(df[pval_col]<=p_cut),"Regulation"]="Up"
df.loc[(df[logfc_col]<=neg_fc)&(df[pval_col]<=p_cut),"Regulation"]="Down"

deg = df[df["Regulation"]!="Neutral"]

up_genes = deg[deg["Regulation"]=="Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"]=="Down"][gene_col].astype(str).tolist()
genes = deg[gene_col].astype(str).tolist()

# ==========================================================
# DASHBOARD METRICS
# ==========================================================
colA, colB, colC = st.columns(3)

colA.metric("Total DEGs", len(deg))
colB.metric("Upregulated", len(up_genes))
colC.metric("Downregulated", len(down_genes))

# ==========================================================
# DOWNLOAD SECTION
# ==========================================================
st.subheader("⬇️ DEG Downloads")

top_n = st.selectbox("Select Top Genes",[10,20,50,100])

up_df = deg[deg["Regulation"]=="Up"].sort_values(logfc_col,ascending=False).head(top_n)
down_df = deg[deg["Regulation"]=="Down"].sort_values(logfc_col).head(top_n)

c1,c2,c3,c4 = st.columns(4)

c1.download_button("Top Up", up_df.to_csv(index=False),"UpGenes.csv")
c2.download_button("Top Down", down_df.to_csv(index=False),"DownGenes.csv")
c3.download_button("All Up", pd.DataFrame(up_genes).to_csv(index=False),"All_Up.csv")
c4.download_button("All Down", pd.DataFrame(down_genes).to_csv(index=False),"All_Down.csv")

# ==========================================================
# VISUALIZATION PANEL
# ==========================================================
st.header("📊 Expression Visualizations")

col1, col2 = st.columns(2)

# ---------- VOLCANO ----------
with col1:
    st.subheader("Volcano Plot")

    palette = st.selectbox("Palette",["Set1","coolwarm","viridis"])

    colors = sns.color_palette(palette,3)

    fig_vol, ax = plt.subplots()
    ax.scatter(df[logfc_col], -np.log10(df[pval_col]), color="grey", s=8)
    ax.scatter(up_df[logfc_col], -np.log10(up_df[pval_col]), color=colors[0])
    ax.scatter(down_df[logfc_col], -np.log10(down_df[pval_col]), color=colors[1])

    st.pyplot(fig_vol)
    download_figure(fig_vol,"Volcano")

# ---------- HEATMAP ----------
with col2:
    st.subheader("Top DEG Heatmap")

    heat_df = deg.head(50).set_index(gene_col)[[logfc_col]]

    fig_heat, ax_heat = plt.subplots()
    sns.heatmap(heat_df, cmap="coolwarm", ax=ax_heat)

    st.pyplot(fig_heat)
    download_figure(fig_heat,"Heatmap")

# ==========================================================
# STRING PPI
# ==========================================================
@st.cache_data(ttl=3600)
def fetch_ppi(g):
    if len(g)==0:
        return pd.DataFrame()
    try:
        r=requests.post(
            "https://string-db.org/api/tsv/network",
            data={"identifiers":"%0d".join(g[:150]),"species":9606},
            timeout=20
        )
        return pd.read_csv(io.StringIO(r.text), sep="\t")
    except:
        return pd.DataFrame()

ppi = fetch_ppi(genes)

# ==========================================================
# PPI PANEL
# ==========================================================
st.header("🔗 Protein Interaction Network")

if not ppi.empty:

    G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")

    hub = pd.DataFrame(dict(G.degree()).items(),columns=["Gene","Score"])
    hub = hub.sort_values("Score",ascending=False).head(10)

    subG = G.subgraph(hub["Gene"].tolist())

    fig_ppi, ax_ppi = plt.subplots(figsize=(8,6))
    pos = nx.spring_layout(subG,seed=42)
    nx.draw(subG,pos,with_labels=True,node_size=2500,ax=ax_ppi)

    st.pyplot(fig_ppi)
    download_figure(fig_ppi,"PPI")

    st.dataframe(hub)

    ALL_TABLES["HubGenes"] = hub

# ==========================================================
# ENRICHMENT
# ==========================================================
gp = GProfiler(return_dataframe=True)

@st.cache_data
def enrich(g):
    return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

up_en = enrich(up_genes)
down_en = enrich(down_genes)

ALL_TABLES["UpEnrichment"] = up_en
ALL_TABLES["DownEnrichment"] = down_en

# ==========================================================
# ENRICHMENT TABLES (UNCHANGED)
# ==========================================================
st.header("🧠 Functional Enrichment")

def enrichment_category_tables(enrich_df,label):
    if enrich_df.empty:
        return
    for src in ["GO:BP","GO:MF","GO:CC","KEGG"]:
        st.subheader(f"{label} — {src}")
        sub=enrich_df[enrich_df["source"]==src]
        if not sub.empty:
            st.dataframe(sub[["name","p_value","intersection_size"]])

enrichment_category_tables(up_en,"Upregulated Genes")
enrichment_category_tables(down_en,"Downregulated Genes")

# ==========================================================
# ⭐ NEW GENE ONTOLOGY VISUALIZATION
# ==========================================================
st.header("📈 Gene Ontology Visualization")

def go_visualization(enrich_df,title):

    go_df = enrich_df[enrich_df["source"].str.contains("GO")].head(10)

    if go_df.empty:
        return

    col1,col2 = st.columns(2)

    # BAR PLOT
    with col1:
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(
            x=-np.log10(go_df["p_value"]),
            y=go_df["name"],
            ax=ax_bar
        )
        ax_bar.set_title(f"{title} GO Bar Plot")
        st.pyplot(fig_bar)
        download_figure(fig_bar,f"{title}_GO_Bar")

    # PIE CHART
    with col2:
        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(
            go_df["intersection_size"],
            labels=go_df["name"],
            autopct="%1.1f%%"
        )
        ax_pie.set_title(f"{title} GO Pie Chart")
        st.pyplot(fig_pie)
        download_figure(fig_pie,f"{title}_GO_Pie")

go_visualization(up_en,"Upregulated")
go_visualization(down_en,"Downregulated")

# ==========================================================
# REPORT EXPORT
# ==========================================================
st.header("📄 Export Report")

if st.button("Generate PDF"):
    buffer=io.BytesIO()
    with PdfPages(buffer) as pdf:
        for name,fig in ALL_FIGURES:
            pdf.savefig(fig)

    st.download_button(
        "Download Full Report",
        buffer.getvalue(),
        "Phoenix_Report.pdf"
    )

# ==========================================================
# METADATA
# ==========================================================
st.header("🔁 Metadata")

st.json({
    "Timestamp":datetime.utcnow().isoformat(),
    "DEGs":len(deg),
    "Up":len(up_genes),
    "Down":len(down_genes)
})
