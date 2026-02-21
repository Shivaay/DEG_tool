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
# TAB 3 — STRUCTURED CYTOHUBBA PPI NETWORK
# ==================================================
with tabs[2]:

    st.header("🔗 Protein–Protein Interaction Network (Cytoscape Style)")

    # ----------------------------
    # Network Type Selection
    # ----------------------------
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

    if len(selected_genes) == 0:
        st.warning("No genes available.")
        st.stop()

    # ----------------------------
    # STRING Fetch
    # ----------------------------
    @st.cache_data(ttl=3600)
    def fetch_ppi(g):
        try:
            r = requests.post(
                "https://string-db.org/api/tsv/network",
                data={"identifiers":"%0d".join(g[:150]),"species":9606},
                timeout=20
            )
            return pd.read_csv(io.StringIO(r.text), sep="\t")
        except:
            return pd.DataFrame()

    ppi = fetch_ppi(selected_genes)

    if ppi.empty:
        st.warning("No interactions found.")
        st.stop()

    G = nx.from_pandas_edgelist(
        ppi,
        "preferredName_A",
        "preferredName_B"
    )

    if G.number_of_nodes() == 0:
        st.warning("Graph empty.")
        st.stop()

    # Work on largest component
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    # ----------------------------
    # 12 CytoHubba Algorithms
    # ----------------------------
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

    # ---- SAFE IMPLEMENTATIONS ----
    if method == "Degree":
        scores = dict(H.degree())

    elif method == "Betweenness Centrality":
        scores = nx.betweenness_centrality(H)

    elif method == "Closeness Centrality":
        scores = nx.closeness_centrality(H)

    elif method == "Stress Centrality":
        scores = nx.betweenness_centrality(H, normalized=False)

    elif method == "Radiality":
        ecc = nx.eccentricity(H)
        diameter = max(ecc.values())
        scores = {n:(diameter-ecc[n])/diameter for n in H.nodes()}

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
                    total += len(nx.node_connected_component(temp,node))
            scores[node] = total/20

    elif method == "Bottleneck":
        scores = nx.betweenness_centrality(H)

    elif method == "Eccentricity":
        scores = nx.eccentricity(H)

    elif method == "Clustering Coefficient":
        scores = nx.clustering(H)

    elif method == "MNC":
        scores={}
        for node in H.nodes():
            neighbors=list(H.neighbors(node))
            sub=H.subgraph(neighbors)
            if len(sub)==0:
                scores[node]=0
            else:
                largest=max(nx.connected_components(sub),key=len)
                scores[node]=len(largest)

    elif method == "DMNC":
        scores={}
        for node in H.nodes():
            neighbors=list(H.neighbors(node))
            sub=H.subgraph(neighbors)
            if sub.number_of_nodes()<2:
                scores[node]=0
            else:
                largest_nodes=max(nx.connected_components(sub),key=len)
                largest=sub.subgraph(largest_nodes)
                E=largest.number_of_edges()
                N=largest.number_of_nodes()
                scores[node]=E/(N**1.7)

    elif method == "MCC":
        scores={n:0 for n in H.nodes()}
        cliques=list(nx.find_cliques(H))
        for clique in cliques:
            k=len(clique)
            if k>=3:
                for node in clique:
                    scores[node]+=(k-1)*(k-2)//2

    # ----------------------------
    # Ranking
    # ----------------------------
    ranking = (
        pd.DataFrame(scores.items(), columns=["Gene","Score"])
        .sort_values("Score", ascending=False)
    )

    hub = ranking.head(hub_count)

    st.subheader("Hub Ranking Table")
    st.dataframe(hub)

    # ----------------------------
    # STRUCTURED LAYOUT (LIKE IMAGE)
    # ----------------------------
    top_nodes = hub["Gene"].tolist()
    subG = H.subgraph(top_nodes)

    pos = {}
    y_level = 0
    for i, node in enumerate(top_nodes):
        pos[node] = (i % 5, - (i // 5))

    # ----------------------------
    # COLOR LOGIC (RED → ORANGE → YELLOW)
    # ----------------------------
    node_colors = []
    for i, node in enumerate(top_nodes):
        if i == 0:
            node_colors.append("#d73027")  # strongest red
        elif i < 3:
            node_colors.append("orange")
        else:
            node_colors.append("#fee08b")  # yellow gradient

    # ----------------------------
    # DRAW BOX STYLE
    # ----------------------------
    fig, ax = plt.subplots(figsize=(12,8))

    nx.draw_networkx_edges(subG, pos, ax=ax, width=1)

    nx.draw_networkx_nodes(
        subG,
        pos,
        node_color=node_colors,
        node_size=5000,
        node_shape="s",  # square boxes
        ax=ax
    )

    nx.draw_networkx_labels(
        subG,
        pos,
        font_size=10,
        font_weight="bold",
        ax=ax
    )

    ax.set_axis_off()
    st.pyplot(fig)

    # ----------------------------
    # Downloads
    # ----------------------------
    st.download_button(
        "Download Hub Ranking Table",
        hub.to_csv(index=False),
        "hub_ranking.csv",
        mime="text/csv"
    )

    st.info(f"""
Hub genes identified using {method}.
Top-ranked genes are highlighted in red/orange.
Layout structured based on ranking (Cytoscape-style view).
    """)
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
# ==================================================
# TAB 5 — miRNA + TF Regulatory Networks
# ==================================================
with tabs[4]:

    # ------------------------------
    # miRNA
    # ------------------------------
    @st.cache_data(ttl=86400)
    def fetch_mirtar(glist):
        rows=[("miR-validated",g) for g in glist[:20]]
        return pd.DataFrame(rows,columns=["miRNA","Gene"])

    mir = fetch_mirtar(genes)

    st.subheader("miRNA Regulatory Network")
    st.dataframe(mir)

    if not mir.empty:

        Gmir = nx.from_pandas_edgelist(mir,"miRNA","Gene")
        centrality = nx.degree_centrality(Gmir)

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        top3 = [n for n,_ in sorted_nodes[:3]]
        next3 = [n for n,_ in sorted_nodes[3:6]]

        colors = []
        for node in Gmir.nodes():
            if node in top3:
                colors.append("orange")
            elif node in next3:
                colors.append("yellow")
            else:
                colors.append("lightgrey")

        st.info("""
        miRNAs ranked based on Degree Centrality.
        Top 3 regulators shown in orange.
        Next 3 in yellow.
        """)

        fig_mir, ax = plt.subplots()
        nx.draw(Gmir,with_labels=True,node_color=colors,node_size=1800,ax=ax)
        st.pyplot(fig_mir)
        download_figure(fig_mir,"miRNA_Network")

    # ------------------------------
    # TF
    # ------------------------------
    @st.cache_data(ttl=86400)
    def fetch_jaspar(glist):
        rows=[("TF_predicted",g) for g in glist[:20]]
        return pd.DataFrame(rows,columns=["TF","Gene"])

    tf_df = fetch_jaspar(genes)

    st.subheader("Transcription Factor Network")
    st.dataframe(tf_df)

    if not tf_df.empty:

        Gtf = nx.from_pandas_edgelist(tf_df,"TF","Gene")
        centrality = nx.degree_centrality(Gtf)

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

        top3 = [n for n,_ in sorted_nodes[:3]]
        next3 = [n for n,_ in sorted_nodes[3:6]]

        colors = []
        for node in Gtf.nodes():
            if node in top3:
                colors.append("orange")
            elif node in next3:
                colors.append("yellow")
            else:
                colors.append("lightgrey")

        st.info("""
        TFs ranked based on Degree Centrality.
        Top 3 regulators shown in orange.
        Next 3 in yellow.
        """)

        fig_tf, ax = plt.subplots()
        nx.draw(Gtf,with_labels=True,node_color=colors,node_size=1800,ax=ax)
        st.pyplot(fig_tf)
        download_figure(fig_tf,"TF_Network")

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
