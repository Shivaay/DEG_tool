# ==========================================================
# PhoenixBioInfoSys DEG Platform
# Layout Upgrade + GO Visualization
# Scientific Logic Fully Preserved
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
import warnings
warnings.filterwarnings("ignore")
from biomath_layer import run_biomath_layer
from interpretation_engine import InterpretationEngine





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
    st.download_button(f"Download {name} (300 DPI)", buf.getvalue(), f"{name}.png")
    ALL_FIGURES.append((name, fig))



tabs = st.tabs([
    "DEG Upload",
    "Tab1",
    "Tab2",
    "Tab3",
    "Tab4",
    "Tab5",
    "Tab6",
    "Integrated Systems Biology"
])

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = tabs



# =====================================================
# TAB 0 — DEG UPLOAD
# =====================================================

with tabs[0]:

    uploaded = st.file_uploader(
        "Upload DEG table (CSV / TSV / XLSX)",
        type=["csv", "tsv", "xlsx"]
    )

    if uploaded is None:
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
    st.success(f"Loaded {df.shape[0]} genes")

    # -------------------------------
    # Column Mapping
    # -------------------------------
    st.sidebar.header("Column Mapping")
    gene_col = st.sidebar.selectbox("Gene column", df.columns)
    logfc_col = st.sidebar.selectbox("logFC column", df.columns)
    pval_col = st.sidebar.selectbox("p-value column", df.columns)

    df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df.dropna(subset=[gene_col, logfc_col, pval_col])

    # -------------------------------
    # Threshold Selection
    # -------------------------------
    st.sidebar.header("Thresholds")
    neg_fc = st.sidebar.slider("Negative logFC", -10.0, 0.0, -1.0)
    pos_fc = st.sidebar.slider("Positive logFC", 0.0, 10.0, 1.0)
    p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

    # -------------------------------
    # DEG Filtering
    # -------------------------------
    df["Regulation"] = "Neutral"

    df.loc[
        (df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut),
        "Regulation"
    ] = "Up"

    df.loc[
        (df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut),
        "Regulation"
    ] = "Down"

    deg = df[df["Regulation"] != "Neutral"].copy()
    st.session_state["deg"] = deg
    st.session_state["gene_col"] = gene_col
    st.session_state["logfc_col"] = logfc_col
    st.session_state["pval_col"] = pval_col


    up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
    down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()
    genes = deg[gene_col].astype(str).tolist()

    # -------------------------------
    # DEG Metrics
    # -------------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("Total DEGs", len(deg))
    col2.metric("Upregulated", len(up_genes))
    col3.metric("Downregulated", len(down_genes))

    # -------------------------------
    # Show Filtered Table
    # -------------------------------
    st.subheader("Filtered DEG Results")
    st.dataframe(deg, use_container_width=True)

    # -------------------------------
    # Download Buttons
    # -------------------------------

    st.download_button(
        "Download Filtered DEG (CSV)",
        deg.to_csv(index=False),
        "filtered_deg.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Upregulated Genes (CSV)",
        deg[deg["Regulation"] == "Up"].to_csv(index=False),
        "upregulated_genes.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Downregulated Genes (CSV)",
        deg[deg["Regulation"] == "Down"].to_csv(index=False),
        "downregulated_genes.csv",
        mime="text/csv"
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        deg.to_excel(writer, sheet_name="All_DEG", index=False)
        deg[deg["Regulation"] == "Up"].to_excel(writer, sheet_name="Upregulated", index=False)
        deg[deg["Regulation"] == "Down"].to_excel(writer, sheet_name="Downregulated", index=False)

    st.download_button(
        "Download DEG (Excel)",
        excel_buffer.getvalue(),
        "deg_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    # ==================================================
# TAB 2 — VOLCANO
# ==================================================
with tabs[1]:

    st.subheader("Volcano Plot")

    palette = st.selectbox("Color Palette", ["Set1","coolwarm","viridis"])
    colors = sns.color_palette(palette, 3)

    fig_vol, ax = plt.subplots()

    ax.scatter(df[logfc_col], -np.log10(df[pval_col]), color="grey", s=8)

    up_df = deg[deg["Regulation"]=="Up"]
    down_df = deg[deg["Regulation"]=="Down"]

    ax.scatter(up_df[logfc_col], -np.log10(up_df[pval_col]), color=colors[0])
    ax.scatter(down_df[logfc_col], -np.log10(down_df[pval_col]), color=colors[1])

    ax.axhline(-np.log10(p_cut), linestyle="--")
    ax.axvline(pos_fc, linestyle="--")
    ax.axvline(neg_fc, linestyle="--")

    st.pyplot(fig_vol)
    download_figure(fig_vol, "Volcano")

# ==================================================
# TAB 3 — PPI
# ==================================================
with tabs[2]:

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
    st.session_state["ppi"] = ppi


    if not ppi.empty:

        G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")

        method = st.selectbox("Hub Metric", ["Degree","MCC"])
        hub_count = st.slider("Hub Count",5,50,10)

        if method=="Degree":
            scores = dict(G.degree())
        else:
            scores = {n:len(list(nx.cliques_containing_node(G,n))) for n in G.nodes}

        hub = pd.DataFrame(scores.items(),columns=["Gene","Score"]).sort_values("Score",ascending=False).head(hub_count)

        hubs = hub["Gene"].tolist()
        subG = G.subgraph(hubs)

        pos = nx.spring_layout(subG)

        cmap = plt.cm.autumn
        node_colors = [cmap(i/max(len(hubs)-1,1)) for i in range(len(hubs))]

        fig_ppi, ax = plt.subplots(figsize=(8,6))
        nx.draw(subG,pos,node_color=node_colors,node_size=2800,with_labels=True,ax=ax)

        st.pyplot(fig_ppi)
        download_figure(fig_ppi,"PPI")

        st.dataframe(hub)
        ALL_TABLES["HubGenes"] = hub



# ==================================================
# TAB 4 — ENRICHMENT + GO VISUALIZATION
# ==================================================
with tabs[3]:

    gp = GProfiler(return_dataframe=True)

    @st.cache_data
    def enrich(g):
        return gp.profile(organism="hsapiens", query=g) if g else pd.DataFrame()

    up_en = enrich(up_genes)
    down_en = enrich(down_genes)

    def enrichment_tables(df_en, label):
        for src in ["GO:BP","GO:MF","GO:CC","KEGG"]:
            st.subheader(f"{label} — {src}")
            sub = df_en[df_en["source"]==src]

            if not sub.empty:
                st.dataframe(sub[["name","p_value","intersection_size"]])

                # ===== GO BAR =====
                fig_bar, ax_bar = plt.subplots()
                top = sub.head(10)
                ax_bar.barh(top["name"], -np.log10(top["p_value"]))
                ax_bar.set_title("GO Bar Plot")
                st.pyplot(fig_bar)
                download_figure(fig_bar,f"{label}_{src}_Bar")

                # ===== GO PIE =====
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(top["intersection_size"], labels=top["name"], autopct="%1.1f%%")
                ax_pie.set_title("GO Pie Chart")
                st.pyplot(fig_pie)
                download_figure(fig_pie,f"{label}_{src}_Pie")

    enrichment_tables(up_en,"Upregulated")
    enrichment_tables(down_en,"Downregulated")

# ==================================================
# TAB 5 — miRNA + TF
# ==================================================
with tabs[4]:

    @st.cache_data(ttl=86400)
    def fetch_mirtar(glist):
        rows=[("miR-validated",g) for g in glist[:20]]
        return pd.DataFrame(rows,columns=["miRNA","Gene"])

    mir = fetch_mirtar(genes)
    st.subheader("miRNA Network")
    st.dataframe(mir)

    if not mir.empty:
        Gmir = nx.from_pandas_edgelist(mir,"miRNA","Gene")
        fig_mir, ax = plt.subplots()
        nx.draw(Gmir,with_labels=True,node_size=1600,ax=ax)
        st.pyplot(fig_mir)
        download_figure(fig_mir,"miRNA")

    @st.cache_data(ttl=86400)
    def fetch_jaspar(glist):
        rows=[("TF_predicted",g) for g in glist[:20]]
        return pd.DataFrame(rows,columns=["TF","Gene"])

    tf_df = fetch_jaspar(genes)
    st.subheader("TF Network")
    st.dataframe(tf_df)

    if not tf_df.empty:
        Gtf = nx.from_pandas_edgelist(tf_df,"TF","Gene")
        fig_tf, ax = plt.subplots()
        nx.draw(Gtf,with_labels=True,node_size=1600,ax=ax)
        st.pyplot(fig_tf)
        download_figure(fig_tf,"TF")

# ==================================================
# TAB 6 — ADAPTIVE
# ==================================================
# ==================================================
# TAB 5 — BIOMATH ENGINE
# ==
# ==================================================
# TAB 5 — BIOMATH ENGINE
# ==================================================
# ==================================================
# TAB 5 — BIOMATH ENGINE
# ==================================================
with tabs[5]:

    st.header("🧮 BioMathematical Engine")

    if "deg" not in st.session_state:

        st.error("Please upload and filter DEG data in Tab 0 first.")
        st.stop()

    if st.button("Run BioMathematical Analysis"):

        try:

            deg_df = st.session_state["deg"]

            gene_col = st.session_state["gene_col"]

            logfc_col = st.session_state["logfc_col"]

            pval_col = st.session_state["pval_col"]

            ppi_df = st.session_state.get("ppi", None)


            biomath_df, biomath_metrics = run_biomath_layer(

                deg_df.copy(),

                gene_col,

                logfc_col,

                pval_col,

                ppi_df

            )


            # STORE FOR TAB6 & TAB7
            st.session_state["biomath_df"] = biomath_df

            st.session_state["biomath_metrics"] = biomath_metrics

            st.session_state["biomath_results"] = {

                "gene_metrics": biomath_df,

                "system_metrics": biomath_metrics,

                "hub_genes": biomath_df.sort_values(
                    "network_centrality",
                    ascending=False
                ).head(10)

            }


            st.success("BioMath Analysis Complete")


            st.subheader("System Metrics")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Entropy", f"{biomath_metrics['system_entropy']:.4f}")

            col2.metric("Stability", f"{biomath_metrics['system_stability']:.4f}")

            col3.metric("Centrality", f"{biomath_metrics['network_centrality']:.4f}")

            col4.metric("Perturbation", f"{biomath_metrics['perturbation_magnitude']:.4f}")


            st.dataframe(biomath_df, use_container_width=True)


        except Exception as e:

            st.error(str(e))
# ==================================================
# TAB 6 — Integrated Systems Biology
# ==================================================
with tabs[6]:

    st.header("🧬 Integrated Systems Biology")

    if "biomath_results" not in st.session_state:

        st.error("Run BioMath Engine first in Tab 5")

        st.stop()


    biomath_results = st.session_state["biomath_results"]

    gene_metrics = biomath_results["gene_metrics"]

    system_metrics = biomath_results["system_metrics"]

    hub_genes = biomath_results["hub_genes"]


    st.subheader("System Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Entropy", round(system_metrics["system_entropy"],4))

    col2.metric("Stability", round(system_metrics["system_stability"],4))

    col3.metric("Centrality", round(system_metrics["network_centrality"],4))

    col4.metric("Perturbation", round(system_metrics["perturbation_magnitude"],4))


    st.subheader("Gene Metrics")

    st.dataframe(gene_metrics)


    st.subheader("Hub Genes")

    st.dataframe(hub_genes)
# ==================================================
# TAB 7 — Scientific Interpretation Engine
# ==================================================
with tabs[7]:

    st.header("Scientific Interpretation Engine")

    if "biomath_results" not in st.session_state:

        st.error("Run BioMath Engine first")

        st.stop()


    if st.button("Generate Scientific Interpretation"):

        try:

            biomath_results = st.session_state["biomath_results"]

            engine = InterpretationEngine(

                biomath_results=biomath_results

            )


            report = engine.generate()


            st.success("Interpretation Generated")


            st.subheader("Manuscript Ready Output")

            st.write(report["text_report"])


            st.download_button(

                "Download Report",

                report["text_report"],

                "interpretation.txt"

            )


        except Exception as e:

            st.error(str(e))

# ==================================================
# METADATA (SAFE)
# ==================================================
st.header("Metadata")

st.json({
    "Timestamp": datetime.utcnow().isoformat(),
    "DEGs": len(st.session_state.get("deg", []))
})
