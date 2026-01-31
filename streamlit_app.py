import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys DEG Tool")

# ============================================================
# BASE APP (UNCHANGED LOGIC & LAYOUT)
# ============================================================

st.title("PhoenixBioInfoSys – Differential Expression Analysis")

uploaded_file = st.file_uploader(
    "Upload DEG file (CSV / TSV / XLSX, up to 1GB)",
    type=["csv", "tsv", "xlsx"]
)

if uploaded_file is None:
    st.stop()

# File loading (already supported in base)
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".tsv"):
    df = pd.read_csv(uploaded_file, sep="\t")
elif uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
else:
    st.error("Unsupported file format")
    st.stop()

# Required columns (base logic preserved)
required_cols = ["gene", "logFC", "pvalue"]
if not all(c in df.columns for c in required_cols):
    st.error("Input must contain: gene, logFC, pvalue")
    st.stop()

# Thresholds
st.sidebar.header("DEG Thresholds")
logfc_cut = st.sidebar.selectbox("log2FC cutoff", [1, 2, 3, 4], index=0)
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001], index=0)

df["Regulation"] = "NS"
df.loc[(df.logFC >= logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= -logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation == "Up"]
down = df[df.Regulation == "Down"]

# Downloads
st.sidebar.download_button(
    "Download Upregulated Genes",
    up.to_csv(index=False),
    "upregulated_genes.csv"
)

st.sidebar.download_button(
    "Download Downregulated Genes",
    down.to_csv(index=False),
    "downregulated_genes.csv"
)

# Volcano plot
st.subheader("Volcano Plot")
fig, ax = plt.subplots(figsize=(6, 5))
sns.scatterplot(
    data=df,
    x="logFC",
    y=-np.log10(df.pvalue),
    hue="Regulation",
    ax=ax
)
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(p-value)")
st.pyplot(fig)

fig.savefig("volcano.png", dpi=300)
st.download_button(
    "Download Volcano Plot (300 DPI)",
    open("volcano.png", "rb"),
    "volcano.png"
)

# Enrichment
gp = GProfiler(return_dataframe=True)
genes = up.gene.tolist() + down.gene.tolist()
enrich = gp.profile(organism="hsapiens", query=genes) if genes else pd.DataFrame()

def show_table(source, title):
    if enrich.empty:
        return
    tab = enrich[enrich.source == source]
    if not tab.empty:
        st.subheader(title)
        st.dataframe(tab[["name", "p_value", "intersection_size"]])

show_table("GO:BP", "GO Biological Process")
show_table("GO:CC", "GO Cellular Component")
show_table("GO:MF", "GO Molecular Function")
show_table("KEGG", "KEGG Pathways")

# PPI Network
st.subheader("PPI Network")

hub_method = st.sidebar.selectbox("Hub method", ["Degree", "MCC"])
top_n = st.sidebar.selectbox("Top hub genes", [10, 20, 50], index=0)

ppi_genes = genes[:100]
edges = [(a, b) for i, a in enumerate(ppi_genes) for b in ppi_genes[i+1:i+4]]

G_ppi = nx.Graph()
G_ppi.add_edges_from(edges)

scores = dict(G_ppi.degree()) if hub_method == "Degree" else nx.clustering(G_ppi)

hub_df = (
    pd.DataFrame(scores.items(), columns=["Gene", "Score"])
    .sort_values("Score", ascending=False)
    .head(top_n)
)

st.dataframe(hub_df)

fig_ppi, ax_ppi = plt.subplots(figsize=(6, 6))
nx.draw(G_ppi, ax=ax_ppi, node_size=30)
st.pyplot(fig_ppi)

fig_ppi.savefig("ppi.png", dpi=300)
st.download_button(
    "Download PPI Network (300 DPI)",
    open("ppi.png", "rb"),
    "ppi.png"
)

# ============================================================
# ADDITIONAL FEATURE: miRNA ANALYSIS (PURELY ADDITIVE)
# ============================================================

st.markdown("---")
st.subheader("miRNA–Gene Regulatory Analysis")

enable_mirna = st.checkbox("Enable miRNA analysis", value=False)

if enable_mirna:

    mirna_file = st.file_uploader(
        "Upload miRTarBase file (CSV / TSV / XLSX)",
        type=["csv", "tsv", "xlsx"]
    )

    if mirna_file is None:
        st.info("Please upload a miRTarBase file to continue.")
        st.stop()

    # Load miRTarBase
    if mirna_file.name.endswith(".csv"):
        mirna_df = pd.read_csv(mirna_file)
    elif mirna_file.name.endswith(".tsv"):
        mirna_df = pd.read_csv(mirna_file, sep="\t")
    elif mirna_file.name.endswith(".xlsx"):
        mirna_df = pd.read_excel(mirna_file)

    required_mirna_cols = [
        "miRNA",
        "Target Gene",
        "Support Type",
        "PMID",
        "Species (Target Gene)"
    ]

    if not all(c in mirna_df.columns for c in required_mirna_cols):
        st.error("miRTarBase file missing required columns.")
        st.stop()

    # Species selector
    species = st.selectbox(
        "Select species",
        sorted(mirna_df["Species (Target Gene)"].dropna().unique())
    )

    mirna_df = mirna_df[mirna_df["Species (Target Gene)"] == species]

    # Evidence filters
    strong_only = st.checkbox("Strong experimental evidence only")

    strong_terms = [
        "Reporter assay",
        "Western blot",
        "qPCR",
        "CLIP-seq"
    ]

    if strong_only:
        mirna_df = mirna_df[
            mirna_df["Support Type"].str.contains(
                "|".join(strong_terms),
                case=False,
                na=False
            )
        ]

    base_genes = hub_df["Gene"].tolist()
    mirna_hits = mirna_df[mirna_df["Target Gene"].isin(base_genes)]

    if mirna_hits.empty:
        st.warning("No miRNA interactions found for hub genes.")
        st.stop()

    # PMID auto-links
    mirna_hits["PMID"] = mirna_hits["PMID"].astype(str)
    mirna_hits["PMID_Link"] = mirna_hits["PMID"].apply(
        lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}/"
    )

    st.markdown("### miRNA–Gene Interaction Table")
    st.dataframe(
        mirna_hits[["miRNA", "Target Gene", "Support Type", "PMID_Link"]],
        use_container_width=True
    )

    # miRNA enrichment
    st.markdown("### miRNA Enrichment")

    mirna_enrich = (
        mirna_hits["miRNA"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "miRNA", "miRNA": "Target_Count"})
    )

    st.dataframe(mirna_enrich)

    fig_mirna, ax_mirna = plt.subplots()
    mirna_enrich.head(20).plot(
        kind="bar",
        x="miRNA",
        y="Target_Count",
        ax=ax_mirna,
        legend=False
    )
    ax_mirna.set_ylabel("Number of target genes")
    ax_mirna.set_title("Top miRNAs regulating hub genes")
    st.pyplot(fig_mirna)

    fig_mirna.savefig("mirna_enrichment.png", dpi=300)
    st.download_button(
        "Download miRNA Enrichment Plot (300 DPI)",
        open("mirna_enrichment.png", "rb"),
        "miRNA_enrichment.png"
    )

    # Evidence-weighted miRNA network
    st.markdown("### Evidence-weighted miRNA Network")

    G_mirna = nx.Graph()

    for _, r in mirna_hits.iterrows():
        weight = 2 if any(
            t.lower() in r["Support Type"].lower() for t in strong_terms
        ) else 1
        G_mirna.add_edge(r["miRNA"], r["Target Gene"], weight=weight)

    fig_net, ax_net = plt.subplots(figsize=(7, 7))
    widths = [d["weight"] for _, _, d in G_mirna.edges(data=True)]

    nx.draw(
        G_mirna,
        ax=ax_net,
        node_size=30,
        width=widths,
        with_labels=False
    )
    st.pyplot(fig_net)

    fig_net.savefig("mirna_network.png", dpi=300)
    st.download_button(
        "Download miRNA Network (300 DPI)",
        open("mirna_network.png", "rb"),
        "miRNA_network.png"
    )
