import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from gprofiler import GProfiler

st.set_page_config(layout="wide", page_title="DEG Analysis Tool")

st.title("Differential Gene Expression Analysis Tool")

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("Upload DEG file (CSV / TSV / XLSX)", type=["csv", "tsv", "xlsx"])

if file:
    # -------------------------
    # READ FILE
    # -------------------------
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.name.endswith(".tsv"):
        df = pd.read_csv(file, sep="\t")
    else:
        df = pd.read_excel(file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # -------------------------
    # COLUMN SELECTION
    # -------------------------
    st.sidebar.header("Column Selection")

    gene_col = st.sidebar.selectbox("Gene Column", df.columns)
    logfc_col = st.sidebar.selectbox("log2FC Column", df.columns)
    pval_col = st.sidebar.selectbox("p-value Column", df.columns)

    # -------------------------
    # RENAME INTERNALLY (NO LOGIC CHANGE)
    # -------------------------
    df = df.rename(columns={
        gene_col: "gene",
        logfc_col: "logFC",
        pval_col: "pvalue"
    })

    # =========================
    # 🔒 CRITICAL SAFETY FIXES
    # =========================

    # Force numeric conversion
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["gene", "logFC", "pvalue"])

    # Ensure gene column is string
    df["gene"] = df["gene"].astype(str)

    # =========================
    # FILTERS
    # =========================
    st.sidebar.header("Thresholds")

    pos_fc = st.sidebar.slider("Positive log2FC cutoff", 1, 10, 1)
    neg_fc = st.sidebar.slider("Negative log2FC cutoff", -10, -1, -1)
    p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

    # =========================
    # REGULATION ASSIGNMENT
    # =========================
    df["Regulation"] = "NS"

    df.loc[
        (df["logFC"] >= pos_fc) & (df["pvalue"] <= p_cut),
        "Regulation"
    ] = "Up"

    df.loc[
        (df["logFC"] <= neg_fc) & (df["pvalue"] <= p_cut),
        "Regulation"
    ] = "Down"

    up = df[df["Regulation"] == "Up"]
    down = df[df["Regulation"] == "Down"]

    st.subheader("DEG Summary")
    st.write({
        "Upregulated": up.shape[0],
        "Downregulated": down.shape[0],
        "Non-significant": df[df["Regulation"] == "NS"].shape[0]
    })

    # =========================
    # VOLCANO PLOT
    # =========================
    st.subheader("Volcano Plot")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["logFC"], -np.log10(df["pvalue"]), c="lightgrey", s=10)

    ax.scatter(up["logFC"], -np.log10(up["pvalue"]), c="red", s=12, label="Up")
    ax.scatter(down["logFC"], -np.log10(down["pvalue"]), c="blue", s=12, label="Down")

    ax.axvline(pos_fc, linestyle="--")
    ax.axvline(neg_fc, linestyle="--")
    ax.axhline(-np.log10(p_cut), linestyle="--")

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.legend()

    st.pyplot(fig)

    # =========================
    # DOWNLOADS
    # =========================
    st.subheader("Download DEG Lists")

    st.download_button(
        "Download Upregulated Genes",
        up.to_csv(index=False),
        file_name="upregulated_genes.csv"
    )

    st.download_button(
        "Download Downregulated Genes",
        down.to_csv(index=False),
        file_name="downregulated_genes.csv"
    )

    # =========================
    # FUNCTIONAL ENRICHMENT
    # =========================
    st.subheader("Functional Enrichment Analysis (g:Profiler)")

    # 🔒 CLEAN GENE LIST (CRITICAL FIX)
    genes = (
        pd.concat([up["gene"], down["gene"]])
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if len(genes) > 0:
        gp = GProfiler(return_dataframe=True)

        try:
            enrich = gp.profile(
                organism="hsapiens",
                query=genes
            )

            if enrich is not None and not enrich.empty:
                for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
                    tab = enrich[enrich["source"] == src]
                    if not tab.empty:
                        st.markdown(f"### {src}")
                        st.dataframe(tab)
            else:
                st.info("No enrichment results found.")

        except Exception as e:
            st.error("Enrichment failed due to invalid gene values.")
            st.exception(e)
    else:
        st.warning("No valid genes available for enrichment.")

