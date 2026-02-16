# ==========================================================
# PhoenixBioInfoSys DEG Platform
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
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

from interpretation_engine import InterpretationInput, InterpretationEngine
from biomath_layer import run_biomath_layer
from interpreter_layer import run_interpreter_layer

# ---------------- SESSION STATE ----------------
if "biomath_df" not in st.session_state:
    st.session_state["biomath_df"] = None

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — DEG Interpretation Platform")

ALL_FIGURES = []
ALL_TABLES = {}

# ==================================================
# UNIVERSAL DOWNLOAD BUFFER
# ==================================================
def download_figure(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button(f"Download {name}", buf.getvalue(), f"{name}.png")
    ALL_FIGURES.append((name, fig))

# ==================================================
# TABS
# ==================================================
tabs = st.tabs([
    "📂 Upload & DEG",
    "📊 Volcano",
    "🔗 PPI",
    "🧠 Enrichment",
    "🧬 Regulatory Networks",
    "🤖 Adaptive Layer",
    "📄 Export & Interpretation"
])

# ==================================================
# TAB 1 — DATA INPUT
# ==================================================
with tabs[0]:

    uploaded = st.file_uploader("Upload DEG table", type=["csv","tsv","xlsx"])

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

    st.sidebar.header("Column Mapping")
    gene_col = st.sidebar.selectbox("Gene column", df.columns)
    logfc_col = st.sidebar.selectbox("logFC column", df.columns)
    pval_col = st.sidebar.selectbox("p-value column", df.columns)

    df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df.dropna(subset=[gene_col, logfc_col, pval_col])

    st.sidebar.header("Thresholds")
    neg_fc = st.sidebar.slider("Negative logFC", -10.0, 0.0, -1.0)
    pos_fc = st.sidebar.slider("Positive logFC", 0.0, 10.0, 1.0)
    p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

    df["Regulation"] = "Neutral"
    df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
    df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

    deg = df[df["Regulation"] != "Neutral"]
    up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
    down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()
    genes = deg[gene_col].astype(str).tolist()

    st.metric("Total DEGs", len(deg))
    st.metric("Upregulated", len(up_genes))
    st.metric("Downregulated", len(down_genes))

    # ✅ SHOW FILTERED TABLE
    st.subheader("Filtered DEG Table")
    st.dataframe(deg)

# ==================================================
# TAB 2 — VOLCANO
# ==================================================
with tabs[1]:

    fig, ax = plt.subplots()
    ax.scatter(df[logfc_col], -np.log10(df[pval_col]), color="grey", s=8)

    ax.scatter(deg[deg["Regulation"]=="Up"][logfc_col],
               -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),
               color="red")

    ax.scatter(deg[deg["Regulation"]=="Down"][logfc_col],
               -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),
               color="blue")

    st.pyplot(fig)
    download_figure(fig,"Volcano")

# ==================================================
# TAB 3 — PPI
# ==================================================
with tabs[2]:

    @st.cache_data
    def fetch_ppi(g):
        if len(g)==0:
            return pd.DataFrame()
        try:
            r = requests.post(
                "https://string-db.org/api/tsv/network",
                data={"identifiers":"%0d".join(g[:150]),"species":9606}
            )
            return pd.read_csv(io.StringIO(r.text), sep="\t")
        except:
            return pd.DataFrame()

    ppi = fetch_ppi(genes)

    if not ppi.empty:

        G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")
        scores = dict(G.degree())

        hub = pd.DataFrame(scores.items(),columns=["Gene","Score"])\
            .sort_values("Score",ascending=False).head(10)

        fig, ax = plt.subplots()
        nx.draw(G.subgraph(hub["Gene"]), with_labels=True, ax=ax)

        st.pyplot(fig)
        download_figure(fig,"PPI")

        ALL_TABLES["HubGenes"] = hub

# ==================================================
# TAB 4 — ENRICHMENT
# ==================================================
with tabs[3]:

    gp = GProfiler(return_dataframe=True)

    def enrich(g):
        return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

    up_en = enrich(up_genes)
    down_en = enrich(down_genes)

    st.dataframe(up_en.head(10))
    st.dataframe(down_en.head(10))

# ==================================================
# TAB 5 — REGULATORY NETWORK
# ==================================================
with tabs[4]:

    mir = pd.DataFrame({"miRNA":["miR-test"],"Gene":[genes[0]]}) if genes else pd.DataFrame()
    tf_df = pd.DataFrame({"TF":["TF-test"],"Gene":[genes[0]]}) if genes else pd.DataFrame()

    st.dataframe(mir)
    st.dataframe(tf_df)

# ==================================================
# TAB 6 — BIOMATH
# ==================================================
with tabs[5]:

    st.subheader("Adaptive BioMathematical Analysis")

    if st.checkbox("Enable BioMathematical Layer"):

        if st.button("Run Biomath Engine"):

            if not ppi.empty:
                ppi_edges = ppi[["preferredName_A","preferredName_B"]]
                st.session_state["biomath_df"] = run_biomath_layer(deg, ppi_edges)

                st.success("Biomath analysis complete")
                st.dataframe(st.session_state["biomath_df"].head(20))

# ==================================================
# TAB 7 — EXPORT & INTERPRETATION
# ==================================================
with tabs[6]:

    if st.button("Generate PDF Report"):
        buffer = io.BytesIO()
        with PdfPages(buffer) as pdf:
            for name, fig in ALL_FIGURES:
                pdf.savefig(fig)

        st.download_button("Download PDF", buffer.getvalue(),"Phoenix_Report.pdf")

    if st.checkbox("Generate Interpretation Report"):

        biomath_df = st.session_state.get("biomath_df")

        if biomath_df is not None and not ppi.empty:
            ppi_edges = ppi[["preferredName_A","preferredName_B"]]
            run_interpreter_layer(biomath_df, ppi_edges)

        input_data = InterpretationInput(
            deg_table=deg,
            up_genes=pd.DataFrame(up_genes, columns=["Gene"]),
            down_genes=pd.DataFrame(down_genes, columns=["Gene"]),
            hub_genes=ALL_TABLES.get("HubGenes"),
            enrichment_up=up_en,
            enrichment_down=down_en,
            mirna_df=mir,
            tf_df=tf_df
        )

        engine = InterpretationEngine(input_data)
        st.text_area("Interpretation", engine.generate_report(), height=400)

# ==================================================
# METADATA
# ==================================================
st.json({"Timestamp": datetime.utcnow().isoformat(),"DEGs": len(deg)})
