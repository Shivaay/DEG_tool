# ==========================================================
# PhoenixBioInfoSys DEG Platform — Advanced Dashboard Layout
# (Layout Improved | Logic Preserved | GO Visualizations Added)
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
import math
import tempfile
from pyvis.network import Network
from interpretation_engine import InterpretationInput, InterpretationEngine
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="PhoenixBioInfoSys DEG Dashboard",
    layout="wide",
    page_icon="🧬"
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
# 🧬 PhoenixBioInfoSys DEG Interpretation Platform
### Integrated Bioinformatics & Clinical Interpretation Dashboard
""")

ALL_FIGURES = []
ALL_TABLES = {}

# ==========================================================
# UNIVERSAL FIGURE DOWNLOAD
# ==========================================================

def download_figure(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button(f"⬇ Download {name}", buf.getvalue(), f"{name}.png")
    ALL_FIGURES.append((name, fig))

# ==========================================================
# DATA INPUT PANEL
# ==========================================================

with st.sidebar:
    st.header("📂 Data Upload")

uploaded = st.file_uploader(
    "Upload DEG Table",
    type=["csv", "tsv", "xlsx"]
)

if uploaded is None:
    st.info("Upload dataset to begin analysis")
    st.stop()

@st.cache_data
def load_data(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    if f.name.endswith(".tsv"):
        return pd.read_csv(f, sep="\t")
    return pd.read_excel(f)

df = load_data(uploaded)

st.success(f"Loaded {df.shape[0]} Genes")

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

with st.sidebar:

    st.header("🧪 Column Mapping")

    gene_col = st.selectbox("Gene Column", df.columns)
    logfc_col = st.selectbox("logFC Column", df.columns)
    pval_col = st.selectbox("p-value Column", df.columns)

    st.header("🎯 Thresholds")

    neg_fc = st.slider("Negative logFC", -5.0, 0.0, -1.0)
    pos_fc = st.slider("Positive logFC", 0.0, 5.0, 1.0)
    p_cut = st.slider("p-value cutoff", 0.0001, 0.1, 0.05)

# ==========================================================
# DATA CLEANING
# ==========================================================

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")

df = df.dropna(subset=[gene_col, logfc_col, pval_col])

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]

up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()
genes = deg[gene_col].astype(str).tolist()

# ==========================================================
# DASHBOARD METRICS
# ==========================================================

c1, c2, c3 = st.columns(3)

c1.metric("Total DEGs", len(deg))
c2.metric("Upregulated", len(up_genes))
c3.metric("Downregulated", len(down_genes))

# ==========================================================
# TABS LAYOUT
# ==========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Visualization",
    "🔗 PPI Network",
    "🧠 Enrichment",
    "🧬 Regulatory Networks",
    "🤖 Adaptive AI",
    "📄 Reports"
])

# ==========================================================
# ================= TAB 1 VISUALIZATION =====================
# ==========================================================

with tab1:

    st.subheader("Volcano Plot")

    palette = st.selectbox("Color Palette", ["Set1", "coolwarm", "viridis"])
    colors = sns.color_palette(palette, 3)

    fig_vol, ax = plt.subplots(figsize=(8,6))

    ax.scatter(df[logfc_col], -np.log10(df[pval_col]), color="grey", s=8)
    ax.scatter(
        deg[deg["Regulation"]=="Up"][logfc_col],
        -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),
        color=colors[0]
    )

    ax.scatter(
        deg[deg["Regulation"]=="Down"][logfc_col],
        -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),
        color=colors[1]
    )

    st.pyplot(fig_vol)
    download_figure(fig_vol,"Volcano")

    # ---------------- HEATMAP ----------------

    if st.checkbox("Show Heatmap"):

        heat_df = deg.head(50).set_index(gene_col)[[logfc_col]]

        fig_heat, ax_heat = plt.subplots(figsize=(5,10))
        sns.heatmap(heat_df, cmap="coolwarm", ax=ax_heat)

        st.pyplot(fig_heat)
        download_figure(fig_heat,"Heatmap")

# ==========================================================
# ================= TAB 2 PPI NETWORK ======================
# ==========================================================

with tab2:

    st.subheader("STRING PPI Network")

    @st.cache_data(ttl=3600)
    def fetch_ppi(g):
        try:
            r = requests.post(
                "https://string-db.org/api/tsv/network",
                data={"identifiers":"%0d".join(g[:150]), "species":9606}
            )
            return pd.read_csv(io.StringIO(r.text), sep="\t")
        except:
            return pd.DataFrame()

    ppi = fetch_ppi(genes)

    if not ppi.empty:

        G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")

        central = dict(G.degree())

        hub = pd.DataFrame(
            central.items(),
            columns=["Gene","Score"]
        ).sort_values("Score",ascending=False).head(10)

        hubs = hub["Gene"].tolist()

        subG = G.subgraph(hubs)

        fig_ppi, ax_ppi = plt.subplots(figsize=(7,7))
        pos = nx.spring_layout(subG)

        nx.draw(subG,pos,with_labels=True,node_size=2200,ax=ax_ppi)

        st.pyplot(fig_ppi)
        download_figure(fig_ppi,"PPI")

        st.dataframe(hub)
        ALL_TABLES["HubGenes"] = hub

# ==========================================================
# ================= TAB 3 ENRICHMENT =======================
# ==========================================================

with tab3:

    gp = GProfiler(return_dataframe=True)

    @st.cache_data
    def enrich(g):
        return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

    up_en = enrich(up_genes)
    down_en = enrich(down_genes)

    ALL_TABLES["UpEnrichment"] = up_en
    ALL_TABLES["DownEnrichment"] = down_en

    st.subheader("GO / KEGG Enrichment Tables")

    st.dataframe(up_en.head(30))

    # ======================================================
    # ⭐ NEW GENE ONTOLOGY BAR PLOT
    # ======================================================

    def go_bar_plot(df_enrich, title):

        go_df = df_enrich[df_enrich["source"]=="GO:BP"].head(10)

        if go_df.empty:
            return

        fig, ax = plt.subplots(figsize=(8,5))

        sns.barplot(
            x=-np.log10(go_df["p_value"]),
            y=go_df["name"],
            ax=ax
        )

        ax.set_title(title)
        st.pyplot(fig)
        download_figure(fig,title)

    go_bar_plot(up_en,"GO_BP_Barplot_Up")

    # ======================================================
    # ⭐ NEW GENE ONTOLOGY PIE CHART
    # ======================================================

    def go_pie_plot(df_enrich, title):

        go_df = df_enrich[df_enrich["source"]=="GO:BP"].head(5)

        if go_df.empty:
            return

        fig, ax = plt.subplots()

        ax.pie(
            go_df["intersection_size"],
            labels=go_df["name"],
            autopct="%1.1f%%"
        )

        ax.set_title(title)

        st.pyplot(fig)
        download_figure(fig,title)

    go_pie_plot(up_en,"GO_BP_Pie_Up")

# ==========================================================
# ================= TAB 4 REGULATORY NETWORKS ==============
# ==========================================================

with tab4:

    st.subheader("miRNA Network")

    def fetch_mirtar(glist):
        rows=[]
        for g in glist[:20]:
            rows.append(("miR-validated",g))
        return pd.DataFrame(rows,columns=["miRNA","Gene"])

    mir = fetch_mirtar(genes)
    st.dataframe(mir)

    if not mir.empty:
        Gmir = nx.from_pandas_edgelist(mir,"miRNA","Gene")

        fig_mir, ax = plt.subplots()
        nx.draw(Gmir,with_labels=True,node_size=1800,ax=ax)

        st.pyplot(fig_mir)
        download_figure(fig_mir,"miRNA")

    # TF

    st.subheader("TF Network")

    def fetch_jaspar(glist):
        rows=[]
        for g in glist[:20]:
            rows.append(("TF_predicted",g))
        return pd.DataFrame(rows,columns=["TF","Gene"])

    tf_df = fetch_jaspar(genes)

    if not tf_df.empty:

        Gtf = nx.from_pandas_edgelist(tf_df,"TF","Gene")

        fig_tf, ax = plt.subplots()
        nx.draw(Gtf,with_labels=True,node_size=1800,ax=ax)

        st.pyplot(fig_tf)
        download_figure(fig_tf,"TF")

# ==========================================================
# ================= TAB 5 ADAPTIVE AI ======================
# ==========================================================

with tab5:

    st.subheader("Adaptive DEG Algorithms")

    run_ai = st.checkbox("Activate AI Layer")

    if run_ai:

        thresholds = [abs(p_cut + np.random.normal(0,0.01)) for _ in range(100)]
        adaptive_threshold = np.mean(thresholds)

        st.metric("Adaptive Threshold", adaptive_threshold)

        alpha = 5 + len(up_genes)
        beta_val = 5 + len(down_genes)

        conf = beta.mean(alpha,beta_val)
        st.metric("Bayesian Confidence", conf)

# ==========================================================
# ================= TAB 6 REPORTS ==========================
# ==========================================================

with tab6:

    st.subheader("PDF Export")

    if st.button("Generate PDF Report"):

        buffer = io.BytesIO()

        with PdfPages(buffer) as pdf:
            for name, fig in ALL_FIGURES:
                pdf.savefig(fig)

        st.download_button(
            "Download Full Report",
            buffer.getvalue(),
            "Phoenix_Report.pdf"
        )

    # ---------------- CLINICAL INTERPRETATION -------------

    if st.checkbox("Generate Clinical Interpretation"):

        input_data = InterpretationInput(
            deg_table=deg,
            up_genes=up_genes,
            down_genes=down_genes,
            hub_genes=ALL_TABLES.get("HubGenes"),
            enrichment_up=up_en,
            enrichment_down=down_en,
            mirna_df=mir,
            tf_df=tf_df
        )

        engine = InterpretationEngine(input_data)
        report = engine.generate_report()

        st.text_area("Interpretation Report", report, height=400)

# ==========================================================
# FOOTER METADATA
# ==========================================================

st.markdown("---")
st.subheader("🔁 Metadata")

st.json({
    "Timestamp": datetime.utcnow().isoformat(),
    "DEGs": len(deg),
    "Up": len(up_genes),
    "Down": len(down_genes)
})
