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
# ==================================================
# TAB 3 — PPI NETWORK (STABLE + ALL 12 CYTOHUBBA)
# ==================================================
# ==================================================
# TAB 2 — PPI NETWORK (STRING + 12 cytoHubba)
# ==================================================
with tabs[2]:

    st.header("🔗 Protein–Protein Interaction Network (STRING v11.5)")

    st.caption("""
    Database: STRING v11.5 (Homo sapiens – 9606)  
    Confidence filter: ≥ 0.700 (High confidence)  
    Hub detection: 12 cytoHubba topological algorithms  
    """)

    network_mode = st.radio(
        "Select Network Type",
        ["All DEGs", "Upregulated Only", "Downregulated Only"]
    )

    if network_mode == "Upregulated Only":
        selected_genes = up_genes
    elif network_mode == "Downregulated Only":
        selected_genes = down_genes
    else:
        selected_genes = genes

    if not selected_genes:
        st.warning("No genes available.")
        st.stop()

    # --------------------------------------------------
    # SAFE STRING API CALL
    # --------------------------------------------------
    @st.cache_data(ttl=3600)
    def fetch_ppi(gene_list):

        if not gene_list:
            return pd.DataFrame()

        try:
            response = requests.post(
                "https://string-db.org/api/tsv/network",
                data={
                    "identifiers": "%0d".join(gene_list[:150]),
                    "species": 9606
                },
                timeout=30
            )

            if response.status_code != 200:
                return pd.DataFrame()

            df = pd.read_csv(io.StringIO(response.text), sep="\t")

            if df.empty:
                return pd.DataFrame()

            # Dynamically detect score column
            score_col = None
            for col in df.columns:
                if "score" in col.lower():
                    score_col = col
                    break

            if score_col is None:
                return pd.DataFrame()

            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

            # High confidence ≥ 0.700
            df = df[df[score_col] >= 700]

            return df

        except Exception:
            return pd.DataFrame()

    ppi = fetch_ppi(selected_genes)

    if ppi.empty:
        st.warning("No high-confidence STRING interactions found.")
        st.stop()

    # --------------------------------------------------
    # BUILD NETWORK
    # --------------------------------------------------
    G = nx.from_pandas_edgelist(
        ppi,
        source="preferredName_A",
        target="preferredName_B"
    )

    if G.number_of_nodes() == 0:
        st.warning("Network construction failed.")
        st.stop()

    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    st.success(f"Nodes: {H.number_of_nodes()} | Edges: {H.number_of_edges()}")

    # --------------------------------------------------
    # HUB ALGORITHM SELECTION
    # --------------------------------------------------
    method = st.selectbox(
        "Hub Detection Algorithm",
        [
            "Degree",
            "Betweenness Centrality",
            "Closeness Centrality",
            "Stress Centrality",
            "Radiality",
            "EPC",
            "Bottleneck",
            "Eccentricity",
            "Clustering Coefficient",
            "MNC",
            "DMNC",
            "MCC"
        ]
    )

    hub_count = st.slider("Number of Top Hubs", 5, 30, 10)

    # --------------------------------------------------
    # 12 CYTOHUBBA IMPLEMENTATIONS
    # --------------------------------------------------

    if method == "Degree":
        scores = dict(H.degree())

    elif method == "Betweenness Centrality":
        scores = nx.betweenness_centrality(H)

    elif method == "Closeness Centrality":
        scores = nx.closeness_centrality(H)

    elif method == "Stress Centrality":
        scores = {n: 0 for n in H.nodes()}
        for s in H.nodes():
            paths = nx.single_source_shortest_path(H, s)
            for t in paths:
                if s != t:
                    for node in paths[t]:
                        if node not in (s, t):
                            scores[node] += 1

    elif method == "Radiality":
        ecc = nx.eccentricity(H)
        diameter = max(ecc.values())
        scores = {
            n: sum(
                diameter + 1 - nx.shortest_path_length(H, n, t)
                for t in H.nodes() if t != n
            ) / (len(H.nodes()) - 1)
            for n in H.nodes()
        }

    elif method == "EPC":
        import random
        scores = {}
        for node in H.nodes():
            total = 0
            for _ in range(20):
                temp = H.copy()
                for edge in list(temp.edges()):
                    if random.random() < 0.3:
                        temp.remove_edge(*edge)
                if node in temp:
                    total += len(nx.node_connected_component(temp, node))
            scores[node] = total / 20

    elif method == "Bottleneck":
        scores = nx.betweenness_centrality(H)

    elif method == "Eccentricity":
        scores = nx.eccentricity(H)

    elif method == "Clustering Coefficient":
        scores = nx.clustering(H)

    elif method == "MNC":
        scores = {}
        for node in H.nodes():
            neighbors = list(H.neighbors(node))
            sub = H.subgraph(neighbors)
            if len(sub) == 0:
                scores[node] = 0
            else:
                largest = max(nx.connected_components(sub), key=len)
                scores[node] = len(largest)

    elif method == "DMNC":
        scores = {}
        for node in H.nodes():
            neighbors = list(H.neighbors(node))
            sub = H.subgraph(neighbors)
            if sub.number_of_nodes() < 2:
                scores[node] = 0
            else:
                largest_nodes = max(nx.connected_components(sub), key=len)
                largest = sub.subgraph(largest_nodes)
                E = largest.number_of_edges()
                N = largest.number_of_nodes()
                scores[node] = E / (N ** 1.7)

    elif method == "MCC":
        scores = {n: 0 for n in H.nodes()}
        cliques = list(nx.find_cliques(H))
        for clique in cliques:
            k = len(clique)
            if k >= 3:
                weight = (k - 1) * (k - 2) // 2
                for node in clique:
                    scores[node] += weight

    # --------------------------------------------------
    # RANK HUBS
    # --------------------------------------------------
    ranking = (
        pd.DataFrame(scores.items(), columns=["Gene", "Score"])
        .sort_values("Score", ascending=False)
    )

    hub_df = ranking.head(hub_count)

    st.subheader("Hub Gene Ranking")
    st.dataframe(hub_df)

    # --------------------------------------------------
    # NETWORK VISUALIZATION (Top 3 ORANGE, Bottom 3 YELLOW)
    # --------------------------------------------------
    pos = nx.spring_layout(H, seed=42)

    top3 = hub_df["Gene"].head(3).tolist()
    bottom3 = hub_df["Gene"].tail(3).tolist()

    node_colors = []
    for node in H.nodes():
        if node in top3:
            node_colors.append("orange")
        elif node in bottom3:
            node_colors.append("yellow")
        else:
            node_colors.append("skyblue")

    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(
        H,
        pos,
        node_color=node_colors,
        node_size=800,
        with_labels=True,
        font_size=8,
        ax=ax
    )

    st.pyplot(fig)
# ==================================================
# TAB 4 — ENRICHMENT + GO VISUALIZATION
# ==================================================
with tabs[3]:

    st.header("🧠 Functional Enrichment (FDR Corrected)")

    gp = GProfiler(return_dataframe=True)

    @st.cache_data
    def enrich(g):
        return gp.profile(
            organism="hsapiens",
            query=g,
            significance_threshold_method="fdr"
        )

    up_en = enrich(up_genes)
    down_en = enrich(down_genes)

    for df,label in [(up_en,"Upregulated"),(down_en,"Downregulated")]:
        df = df[df["p_value"] < 0.05].sort_values("p_value")

        for src in ["GO:BP","GO:MF","GO:CC","KEGG","REAC"]:
            sub = df[df["source"]==src]
            if not sub.empty:
                st.subheader(f"{label} — {src}")
                st.dataframe(sub[["name","p_value","intersection_size"]].head(15))
# ==================================================
# TAB 5 — miRNA + TF Regulatory Networks
# ==================================================
with tabs[4]:

    st.header("🧬 Regulatory Networks")

    # ---------------- miRTarBase ----------------
    st.subheader("miRNA Regulatory Network (miRTarBase Validated)")

    @st.cache_data(ttl=86400)
    def fetch_mirtarbase():
        url = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_MTI.xlsx"
        df = pd.read_excel(url)
        return df[df["Species (Target Gene)"]=="Homo sapiens"]

    mirtar = fetch_mirtarbase()

    mir_filtered = mirtar[mirtar["Target Gene"].isin(genes)]
    mir_filtered = mir_filtered[["miRNA","Target Gene","Support Type"]]

    st.dataframe(mir_filtered.head(50))

    if not mir_filtered.empty:
        Gmir = nx.from_pandas_edgelist(
            mir_filtered, "miRNA","Target Gene"
        )
        fig_mir, ax = plt.subplots()
        nx.draw(Gmir,with_labels=True,node_size=1500,ax=ax)
        st.pyplot(fig_mir)

    # ---------------- TRRUST ----------------
    st.subheader("Transcription Factor Network (TRRUST v2)")

    @st.cache_data(ttl=86400)
    def fetch_trrust():
        url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
        df = pd.read_csv(url, sep="\t", header=None)
        df.columns=["TF","Target","Regulation","PMID"]
        return df

    trrust = fetch_trrust()
    tf_filtered = trrust[trrust["Target"].isin(genes)]

    st.dataframe(tf_filtered.head(50))

    if not tf_filtered.empty:
        Gtf = nx.from_pandas_edgelist(
            tf_filtered,"TF","Target"
        )
        fig_tf, ax = plt.subplots()
        nx.draw(Gtf,with_labels=True,node_size=1500,ax=ax)
        st.pyplot(fig_tf)
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
# ==================================================
# TAB 7 — Scientific Interpretation Engine
# ==================================================
with tabs[7]:

    st.header("🧠 Scientific Interpretation Engine")

    # Check BioMath results exist
    if "biomath_results" not in st.session_state:

        st.error("Please run BioMath Engine first in Tab 5.")

        st.stop()


    # Button to generate interpretation
    if st.button("Generate Scientific Interpretation"):

        try:

            # Import engine and input class
            from interpretation_engine import (
                InterpretationEngine,
                InterpretationInput
            )


            # Retrieve BioMath results
            biomath_results = st.session_state["biomath_results"]


            gene_metrics = biomath_results["gene_metrics"]

            system_metrics = biomath_results["system_metrics"]

            hub_genes = biomath_results["hub_genes"]


            # Create correct input object
            input_data = InterpretationInput(

                deg_table = gene_metrics,

                biomath_metrics = system_metrics,

                hub_genes = hub_genes

            )


            # Run interpretation engine
            engine = InterpretationEngine(input_data)

            report = engine.generate()


            # Show success
            st.success("Scientific Interpretation Generated Successfully")


            # Show manuscript text
            st.subheader("📜 Manuscript-Ready Interpretation")

            st.write(report["text_report"])


            # Download button
            st.download_button(

                label = "Download Interpretation Report",

                data = report["text_report"],

                file_name = "PhoenixBioInfoSys_Interpretation.txt",

                mime = "text/plain"

            )


        except Exception:

            # Hide technical error from client
            st.error("Interpretation Engine Failed. Please verify BioMath results.")

# ==================================================
# METADATA (SAFE)
# ==================================================
st.header("Metadata")

st.json({
    "Timestamp": datetime.utcnow().isoformat(),
    "DEGs": len(st.session_state.get("deg", []))
})
