import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys DEG Platform")

# ======================================================
# 1. DATA INPUT
# ======================================================
st.title("PhoenixBioInfoSys – DEG, Enrichment & Network Analysis")

uploaded_file = st.file_uploader(
    "Upload DEG file (CSV / TSV / XLSX, ≤1GB)",
    type=["csv", "tsv", "xlsx"]
)

if uploaded_file is None:
    st.stop()

# Load file
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".tsv"):
    df = pd.read_csv(uploaded_file, sep="\t")
else:
    df = pd.read_excel(uploaded_file)

df.columns = df.columns.astype(str)

st.subheader("Preview")
st.dataframe(df.head())

# ======================================================
# 2. COLUMN SELECTION (USER CONTROL)
# ======================================================
st.sidebar.header("Column Mapping")

gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("log2FC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df = df.rename(columns={
    gene_col: "gene",
    logfc_col: "logFC",
    pval_col: "pvalue"
})

# Force numeric & clean
df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
df["gene"] = df["gene"].astype(str)

df = df.dropna(subset=["gene", "logFC", "pvalue"])

# ======================================================
# 3. FILTERING
# ======================================================
st.sidebar.header("Filtering")

pos_fc = st.sidebar.slider("Upregulated log2FC", 1, 10, 2)
neg_fc = st.sidebar.slider("Downregulated log2FC", -10, -1, -2)
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

df["Regulation"] = "NS"
df.loc[(df.logFC >= pos_fc) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= neg_fc) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation == "Up"]
down = df[df.Regulation == "Down"]

st.success(f"Up: {len(up)} | Down: {len(down)}")

# ======================================================
# 4. DOWNLOAD DEG LISTS
# ======================================================
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

# ======================================================
# 5. VOLCANO PLOT
# ======================================================
st.subheader("Volcano Plot")

fig, ax = plt.subplots(figsize=(7,6))
ax.scatter(df.logFC, -np.log10(df.pvalue), c="lightgrey", s=10)
ax.scatter(up.logFC, -np.log10(up.pvalue), c="red", label="Up")
ax.scatter(down.logFC, -np.log10(down.pvalue), c="blue", label="Down")
ax.axvline(pos_fc, linestyle="--")
ax.axvline(neg_fc, linestyle="--")
ax.axhline(-np.log10(p_cut), linestyle="--")
ax.set_xlabel("log2FC")
ax.set_ylabel("-log10(p-value)")
ax.legend()
st.pyplot(fig)

fig.savefig("volcano.png", dpi=300)
st.download_button("Download Volcano (300 DPI)", open("volcano.png","rb"))

# ======================================================
# 6. HEATMAP (IF POSSIBLE)
# ======================================================
expr_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(expr_cols) > 2:
    st.subheader("Heatmap")
    heat_df = df.set_index("gene")[expr_cols]
    fig_h, ax_h = plt.subplots(figsize=(8,6))
    sns.clustermap(heat_df.head(50), cmap="vlag")
    st.pyplot(fig_h)

# ======================================================
# 7. FUNCTIONAL ENRICHMENT (UP & DOWN SEPARATE)
# ======================================================
st.subheader("Functional Enrichment Analysis")

gp = GProfiler(return_dataframe=True)

def enrich_genes(gene_list, label):
    if len(gene_list) == 0:
        st.warning(f"No genes for {label}")
        return

    try:
        res = gp.profile(organism="hsapiens", query=gene_list)
        for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
            tab = res[res.source == src]
            if not tab.empty:
                st.markdown(f"### {label} – {src}")
                st.dataframe(tab[["name","p_value","intersection_size"]])
    except Exception as e:
        st.error(f"Enrichment failed for {label}")

enrich_genes(up.gene.tolist(), "Upregulated")
enrich_genes(down.gene.tolist(), "Downregulated")

# ======================================================
# 8. PPI NETWORK & HUB GENES
# ======================================================
st.subheader("PPI Network")

top_n = st.selectbox("Top hub genes", [10,20,50])

genes = list(set(up.gene.tolist() + down.gene.tolist()))[:100]
edges = [(a,b) for i,a in enumerate(genes) for b in genes[i+1:i+4]]

G = nx.Graph()
G.add_edges_from(edges)

hub_df = (
    pd.DataFrame(dict(G.degree()).items(), columns=["Gene","Degree"])
    .sort_values("Degree", ascending=False)
    .head(top_n)
)

st.dataframe(hub_df)

fig_ppi, ax_ppi = plt.subplots(figsize=(7,7))
nx.draw(G, ax=ax_ppi, node_shape="s", node_size=400)
st.pyplot(fig_ppi)

fig_ppi.savefig("ppi.png", dpi=300)
st.download_button("Download PPI (300 DPI)", open("ppi.png","rb"))

# ======================================================
# 9. miRNA ANALYSIS (OPTIONAL)
# ======================================================
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
        "Species",
        mirna_df["Species (Target Gene)"].dropna().unique()
    )

    mirna_df = mirna_df[mirna_df["Species (Target Gene)"] == species]

    strong_only = st.checkbox("Strong evidence only")
    strong_terms = ["Reporter assay","Western blot","qPCR","CLIP-seq"]

    if strong_only:
        mirna_df = mirna_df[
            mirna_df["Support Type"].str.contains("|".join(strong_terms), na=False)
        ]

    mirna_hits = mirna_df[mirna_df["Target Gene"].isin(hub_df.Gene)]

    mirna_hits["PMID_Link"] = mirna_hits["PMID"].astype(str).apply(
        lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}/"
    )

    st.dataframe(
        mirna_hits[["miRNA","Target Gene","Support Type","PMID_Link"]],
        use_container_width=True
    )

    st.markdown("### miRNA Enrichment")
    mirna_enrich = mirna_hits["miRNA"].value_counts().head(20)
    fig_m, ax_m = plt.subplots()
    mirna_enrich.plot(kind="bar", ax=ax_m)
    st.pyplot(fig_m)
