import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
from gprofiler import GProfiler
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys – Transcriptomics")

st.title("PhoenixBioInfoSys – DEG, Enrichment & Network Analysis")

# ============================================================
# FILE UPLOAD
# ============================================================
file = st.file_uploader("Upload DEG / Expression file (CSV / TSV / XLSX)", type=["csv", "tsv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        df = pd.read_csv(file, sep="\t")
    else:
        df = pd.read_excel(file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # ============================================================
    # COLUMN SELECTION
    # ============================================================
    st.sidebar.header("Column Mapping")

    gene_col = st.sidebar.selectbox("Gene Column", df.columns)
    logfc_col = st.sidebar.selectbox("log2FC Column", df.columns)
    pval_col = st.sidebar.selectbox("p-value Column", df.columns)

    df = df.rename(columns={
        gene_col: "gene",
        logfc_col: "logFC",
        pval_col: "pvalue"
    })

    # ============================================================
    # SAFETY FIXES
    # ============================================================
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    df["gene"] = df["gene"].astype(str)
    df = df.dropna(subset=["gene", "logFC", "pvalue"])

    # ============================================================
    # FILTERS
    # ============================================================
    st.sidebar.header("Filtering Thresholds")
    pos_fc = st.sidebar.slider("Positive logFC", 1, 10, 1)
    neg_fc = st.sidebar.slider("Negative logFC", -10, -1, -1)
    p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

    # ============================================================
    # DEG CLASSIFICATION
    # ============================================================
    df["Regulation"] = "NS"
    df.loc[(df["logFC"] >= pos_fc) & (df["pvalue"] <= p_cut), "Regulation"] = "Up"
    df.loc[(df["logFC"] <= neg_fc) & (df["pvalue"] <= p_cut), "Regulation"] = "Down"

    up = df[df["Regulation"] == "Up"]
    down = df[df["Regulation"] == "Down"]

    st.subheader("DEG Summary")
    st.json({
        "Upregulated": up.shape[0],
        "Downregulated": down.shape[0],
        "Non-significant": df[df["Regulation"] == "NS"].shape[0]
    })

    # ============================================================
    # VOLCANO PLOT
    # ============================================================
    fig_volcano, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df["logFC"], -np.log10(df["pvalue"]), c="lightgrey", s=10)
    ax.scatter(up["logFC"], -np.log10(up["pvalue"]), c="red", label="Up")
    ax.scatter(down["logFC"], -np.log10(down["pvalue"]), c="blue", label="Down")
    ax.axvline(pos_fc, linestyle="--")
    ax.axvline(neg_fc, linestyle="--")
    ax.axhline(-np.log10(p_cut), linestyle="--")
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.legend()
    st.pyplot(fig_volcano)

    # ============================================================
    # HEATMAP (ONLY IF EXPRESSION EXISTS)
    # ============================================================
    expr_cols = df.select_dtypes(include=[np.number]).columns.drop(["logFC", "pvalue"], errors="ignore")

    fig_heatmap = None
    if len(expr_cols) >= 2:
        fig_heatmap = sns.clustermap(
            df.set_index("gene")[expr_cols],
            z_score=0,
            figsize=(8,8)
        ).fig
        st.pyplot(fig_heatmap)

    # ============================================================
    # FUNCTIONAL ENRICHMENT
    # ============================================================
    gp = GProfiler(return_dataframe=True)

    def enrichment(glist, label):
        if len(glist) == 0:
            return None
        res = gp.profile(organism="hsapiens", query=glist)
        if res is None or res.empty:
            return None
        st.markdown(f"### {label}")
        for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
            tab = res[res["source"] == src]
            if not tab.empty:
                st.markdown(f"**{src}**")
                st.dataframe(tab)
        return res

    enrich_up = enrichment(up["gene"].unique().tolist(), "Upregulated Genes")
    enrich_down = enrichment(down["gene"].unique().tolist(), "Downregulated Genes")

    # ============================================================
    # STRING PPI (REAL-TIME)
    # ============================================================
    st.subheader("Protein–Protein Interaction Network")

    hub_method = st.selectbox("Hub scoring method", ["Degree", "MCC"])
    top_n = st.selectbox("Top hub genes", [10, 20, 50])

    genes = list(set(up["gene"].tolist() + down["gene"].tolist()))[:200]

    G = nx.Graph()

    def fetch_string_ppi(gene_list):
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": "%0d".join(gene_list),
            "species": 9606,
            "required_score": 700
        }
        r = requests.post(url, data=params)
        return r.json() if r.status_code == 200 else []

    try:
        interactions = fetch_string_ppi(genes)
        for i in interactions:
            G.add_edge(i["preferredName_A"], i["preferredName_B"], weight=i["score"])
    except:
        st.warning("STRING API unavailable. PPI skipped.")

    # ============================================================
    # HUB SCORING
    # ============================================================
    if hub_method == "Degree":
        scores = dict(G.degree())
    else:
        scores = {}
        cliques = list(nx.find_cliques(G))
        for n in G.nodes():
            scores[n] = sum(len(c) for c in cliques if n in c)

    hubs = sorted(scores, key=scores.get, reverse=True)[:top_n]

    pos = nx.spring_layout(G, k=1.1, seed=42)

    fig_ppi, ax = plt.subplots(figsize=(8,8))
    colors = []
    for n in G.nodes():
        if n in hubs[:3]:
            colors.append("darkred")
        elif n in hubs:
            colors.append("orange")
        else:
            colors.append("yellow")

    nx.draw(
        G, pos,
        node_color=colors,
        node_size=[scores.get(n,1)*40 for n in G.nodes()],
        with_labels=False,
        ax=ax
    )
    st.pyplot(fig_ppi)

    hub_df = pd.DataFrame({"Gene": hubs, "Score": [scores[h] for h in hubs]})
    st.dataframe(hub_df)

    # ============================================================
    # PDF REPORT
    # ============================================================
    st.subheader("Automated PDF Report")

    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0,10,"Differential Expression Summary", ln=True)
        pdf.cell(0,10,f"Upregulated: {up.shape[0]}", ln=True)
        pdf.cell(0,10,f"Downregulated: {down.shape[0]}", ln=True)

        pdf.cell(0,10,"Top Hub Genes", ln=True)
        for h in hubs[:10]:
            pdf.cell(0,8,h, ln=True)

        buffer = io.BytesIO()
        pdf.output(buffer)
        st.download_button("Download PDF Report", buffer.getvalue(), "DEG_Report.pdf")
