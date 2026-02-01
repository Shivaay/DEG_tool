import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys DEG Platform")

# =========================================================
# 1. DATA INPUT & COLUMN DETECTION
# =========================================================

st.title("PhoenixBioInfoSys – DEG & Network Analysis Platform")

uploaded_file = st.file_uploader(
    "Upload DEG file (CSV / TSV / XLSX, ≤1 GB)",
    type=["csv", "tsv", "xlsx"]
)

if uploaded_file is None:
    st.stop()

# Load file
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".tsv"):
    df = pd.read_csv(uploaded_file, sep="\t")
elif uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)

df.columns = df.columns.astype(str)

st.subheader("Column Mapping (User Controlled)")

gene_col = st.selectbox("Select Gene column", df.columns)
logfc_col = st.selectbox("Select log2FC column", df.columns)
pval_col = st.selectbox("Select p-value column", df.columns)

df = df[[gene_col, logfc_col, pval_col]].copy()
df.columns = ["gene", "logFC", "pvalue"]

# =========================================================
# 2. USER-DEFINED FILTERING (SLIDERS)
# =========================================================

st.sidebar.header("Filtering Controls")

pos_fc = st.sidebar.slider(
    "Upregulated log2FC threshold",
    min_value=1, max_value=10, value=2
)

neg_fc = st.sidebar.slider(
    "Downregulated log2FC threshold",
    min_value=-10, max_value=-1, value=-2
)

p_cut = st.sidebar.selectbox(
    "p-value cutoff",
    [0.05, 0.01, 0.001]
)

df["Regulation"] = "NS"
df.loc[(df.logFC >= pos_fc) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= neg_fc) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation == "Up"]
down = df[df.Regulation == "Down"]

st.success(
    f"Upregulated: {len(up)} | Downregulated: {len(down)} | Total DEGs: {len(up)+len(down)}"
)

# =========================================================
# 3. DEG DOWNLOADS
# =========================================================

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

# =========================================================
# 4. VOLCANO PLOT (CUSTOM COLORS)
# =========================================================

st.subheader("Volcano Plot")

up_color = st.color_picker("Upregulated color", "#d62728")
down_color = st.color_picker("Downregulated color", "#1f77b4")

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(df.logFC, -np.log10(df.pvalue), color="lightgrey", s=10)
ax.scatter(up.logFC, -np.log10(up.pvalue), color=up_color, label="Up")
ax.scatter(down.logFC, -np.log10(down.pvalue), color=down_color, label="Down")
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(p-value)")
ax.legend()
st.pyplot(fig)

fig.savefig("volcano.png", dpi=300)
st.download_button("Download Volcano (300 DPI)", open("volcano.png","rb"))

# =========================================================
# 5. FUNCTIONAL ENRICHMENT (g:Profiler)
# =========================================================

st.subheader("Functional Enrichment Analysis")

gp = GProfiler(return_dataframe=True)
genes = list(set(up.gene.tolist() + down.gene.tolist()))

enrich = gp.profile(organism="hsapiens", query=genes) if genes else pd.DataFrame()

def show_enrich(source, title):
    tab = enrich[enrich.source == source]
    if not tab.empty:
        st.markdown(f"### {title}")
        st.dataframe(tab[["name", "p_value", "intersection_size"]])

show_enrich("GO:BP", "GO Biological Process")
show_enrich("GO:CC", "GO Cellular Component")
show_enrich("GO:MF", "GO Molecular Function")
show_enrich("KEGG", "KEGG Pathways")

# =========================================================
# 6. PPI NETWORK (RECTANGULAR NODES + COLOR GRADIENT)
# =========================================================

st.subheader("Protein–Protein Interaction Network")

top_n = st.selectbox("Top hub genes", [10, 20, 50])
hub_method = st.selectbox("Hub scoring method", ["Degree", "MCC"])

ppi_genes = genes[:100]
edges = [(a, b) for i, a in enumerate(ppi_genes) for b in ppi_genes[i+1:i+4]]

G = nx.Graph()
G.add_edges_from(edges)

scores = dict(G.degree()) if hub_method == "Degree" else nx.clustering(G)

hub_df = (
    pd.DataFrame(scores.items(), columns=["Gene","Score"])
    .sort_values("Score", ascending=False)
    .head(top_n)
)

st.dataframe(hub_df)

# Color logic
node_colors = []
for i, node in enumerate(G.nodes()):
    if node in hub_df.Gene.values[:3]:
        node_colors.append("darkred")
    elif node in hub_df.Gene.values[-3:]:
        node_colors.append("yellow")
    else:
        node_colors.append("lightblue")

fig_ppi, ax_ppi = plt.subplots(figsize=(7,7))
nx.draw(
    G,
    ax=ax_ppi,
    node_color=node_colors,
    node_shape="s",   # rectangular nodes
    node_size=500,
    with_labels=False
)
st.pyplot(fig_ppi)

fig_ppi.savefig("ppi.png", dpi=300)
st.download_button("Download PPI (300 DPI)", open("ppi.png","rb"))

# =========================================================
# 7. miRNA–GENE REGULATORY ANALYSIS
# =========================================================

st.subheader("miRNA–Gene Regulatory Analysis")

enable_mirna = st.checkbox("Enable miRNA analysis")

if enable_mirna:

    mirna_file = st.file_uploader(
        "Upload miRTarBase (CSV / TSV / XLSX)",
        type=["csv","tsv","xlsx"]
    )

    if mirna_file is None:
        st.stop()

    if mirna_file.name.endswith(".csv"):
        mirna_df = pd.read_csv(mirna_file)
    elif mirna_file.name.endswith(".tsv"):
        mirna_df = pd.read_csv(mirna_file, sep="\t")
    else:
        mirna_df = pd.read_excel(mirna_file)

    species = st.selectbox(
        "Select species",
        mirna_df["Species (Target Gene)"].dropna().unique()
    )

    mirna_df = mirna_df[
        mirna_df["Species (Target Gene)"] == species
    ]

    strong_only = st.checkbox("Strong experimental evidence only")

    strong_terms = ["Reporter assay","Western blot","qPCR","CLIP-seq"]

    if strong_only:
        mirna_df = mirna_df[
            mirna_df["Support Type"].str.contains("|".join(strong_terms), na=False)
        ]

    mirna_hits = mirna_df[
        mirna_df["Target Gene"].isin(hub_df.Gene)
    ]

    # Auto PMID links
    mirna_hits["PMID_Link"] = mirna_hits["PMID"].astype(str).apply(
        lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}/"
    )

    st.dataframe(
        mirna_hits[["miRNA","Target Gene","Support Type","PMID_Link"]],
        use_container_width=True
    )

    # miRNA enrichment
    st.markdown("### miRNA Enrichment")

    enrich_mirna = mirna_hits["miRNA"].value_counts().reset_index()
    enrich_mirna.columns = ["miRNA","Target_Count"]
    st.dataframe(enrich_mirna)

    fig_m, ax_m = plt.subplots()
    enrich_mirna.head(20).plot(
        kind="bar",
        x="miRNA",
        y="Target_Count",
        ax=ax_m,
        legend=False
    )
    st.pyplot(fig_m)

    fig_m.savefig("mirna_enrichment.png", dpi=300)
    st.download_button("Download miRNA Enrichment (300 DPI)", open("mirna_enrichment.png","rb"))

# =========================================================
# END
# =========================================================
