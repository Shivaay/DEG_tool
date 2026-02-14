# ==========================================================
# PhoenixBioInfoSys DEG Platform
# UI / Layout Refactor ONLY
# Functional Logic Preserved
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import io
import warnings
from gprofiler import GProfiler
from interpretation_engine import InterpretationInput, InterpretationEngine

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="PhoenixBioInfoSys DEG Platform",
    page_icon="🧬",
    layout="wide"
)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("""
# 🧬 PhoenixBioInfoSys DEG Interpretation Platform
### Multi-Layer Systems Transcriptomic Intelligence Suite
""")

st.markdown("---")

ALL_DOWNLOADS = {}


# ==========================================================
# 📂 DATA LOADING
# ==========================================================
st.sidebar.header("📥 Data Upload")

uploaded = st.sidebar.file_uploader(
    "Upload DEG table",
    type=["csv", "tsv", "xlsx"]
)

if uploaded is None:
    st.info("Upload DEG file to begin analysis")
    st.stop()


@st.cache_data
def load_data(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    elif f.name.endswith(".tsv"):
        return pd.read_csv(f, sep="\t")
    else:
        return pd.read_excel(f)


df = load_data(uploaded)

st.success(f"Dataset Loaded → {df.shape[0]} genes")


# ==========================================================
# 🔬 COLUMN MAPPING
# ==========================================================
st.sidebar.header("🔧 Column Mapping")

gene_col = st.sidebar.selectbox("Gene Column", df.columns)
logfc_col = st.sidebar.selectbox("log2FC Column", df.columns)
pval_col = st.sidebar.selectbox("p-value Column", df.columns)


# ==========================================================
# 🎚 THRESHOLDS
# ==========================================================
st.sidebar.header("Threshold Filters")

neg_fc = st.sidebar.slider("Negative logFC", -5.0, 0.0, -1.0)
pos_fc = st.sidebar.slider("Positive logFC", 0.0, 5.0, 1.0)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]

genes = deg[gene_col].astype(str).tolist()
up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()

st.markdown("---")


# ==========================================================
# 📊 SUMMARY PANEL
# ==========================================================
st.subheader("📊 DEG Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total DEGs", len(genes))
col2.metric("Upregulated", len(up_genes))
col3.metric("Downregulated", len(down_genes))


# ==========================================================
# 📈 VOLCANO VISUALIZATION
# ==========================================================
st.subheader("📈 Volcano Plot")

fig, ax = plt.subplots()

ax.scatter(
    df[logfc_col],
    -np.log10(df[pval_col]),
    c=df["Regulation"].map({"Up": "red", "Down": "blue", "Neutral": "gray"}),
    alpha=0.6
)

ax.set_xlabel("log2FC")
ax.set_ylabel("-log10(p-value)")

st.pyplot(fig)

ALL_DOWNLOADS["Volcano Plot"] = fig


# ==========================================================
# 🧪 GENE ONTOLOGY ANALYSIS
# ==========================================================
st.subheader("🧬 Functional Enrichment (Gene Ontology)")

if len(genes) > 5:

    gp = GProfiler(return_dataframe=True)
    go_df = gp.profile(organism="hsapiens", query=genes)

    if not go_df.empty:

        top_go = go_df.head(10)

        # -----------------------
        # GO BAR CHART
        # -----------------------
        st.markdown("### GO Term Enrichment — Bar Plot")

        fig_bar, ax_bar = plt.subplots()

        ax_bar.barh(
            top_go["name"],
            -np.log10(top_go["p_value"])
        )

        ax_bar.set_xlabel("-log10(p-value)")
        ax_bar.invert_yaxis()

        st.pyplot(fig_bar)

        ALL_DOWNLOADS["GO Bar Plot"] = fig_bar

        # -----------------------
        # GO PIE CHART
        # -----------------------
        st.markdown("### GO Category Distribution — Pie Plot")

        cat_counts = top_go["source"].value_counts()

        fig_pie, ax_pie = plt.subplots()
        ax_pie.pie(
            cat_counts.values,
            labels=cat_counts.index,
            autopct="%1.1f%%"
        )

        st.pyplot(fig_pie)

        ALL_DOWNLOADS["GO Pie Plot"] = fig_pie

        ALL_DOWNLOADS["GO Table"] = top_go

    else:
        st.warning("No GO enrichment found")


# ==========================================================
# 🧠 INTERPRETATION ENGINE PANEL
# ==========================================================
st.subheader("🧠 Multi-Layer Transcriptomic Interpretation Engine")

if st.button("Run Interpretation Engine"):

    engine_input = InterpretationInput(
        deg_table=deg
    )

    engine = InterpretationEngine(engine_input)
    report = engine.run()

    st.json(report)

    ALL_DOWNLOADS["Interpretation Report"] = pd.DataFrame(report)


# ==========================================================
# 📦 DOWNLOAD HUB (Moved to Bottom)
# ==========================================================
st.markdown("---")
st.header("⬇️ Export & Downloads")

for name, obj in ALL_DOWNLOADS.items():

    if isinstance(obj, plt.Figure):

        buf = io.BytesIO()
        obj.savefig(buf, dpi=300, bbox_inches="tight")

        st.download_button(
            f"Download {name}",
            buf.getvalue(),
            f"{name}.png"
        )

    elif isinstance(obj, pd.DataFrame):

        st.download_button(
            f"Download {name}",
            obj.to_csv(index=False),
            f"{name}.csv"
        )
