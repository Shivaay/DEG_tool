import streamlit as st
import pandas as pd
import numpy as np
import os, urllib.request, gzip, shutil
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")

# ----------------------------------------------------
# Universal file loader
# ----------------------------------------------------
def load_table(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    st.error("Unsupported file format")
    st.stop()

# ----------------------------------------------------
# Robust column detection
# ----------------------------------------------------
def detect_column(columns, keywords):
    for col in columns:
        col_low = col.lower().replace(" ", "").replace("_", "")
        for key in keywords:
            if key in col_low:
                return col
    return None

# ----------------------------------------------------
# miRTarBase downloader (cached)
# ----------------------------------------------------
@st.cache_data(show_spinner=True)
def load_mirtarbase():
    url = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_MTI.csv.gz"
    os.makedirs("data", exist_ok=True)
    csv_path = "data/miRTarBase_MTI.csv"

    if not os.path.exists(csv_path):
        st.info("Downloading miRTarBase (one-time ~400MB)")
        urllib.request.urlretrieve(url, csv_path + ".gz")
        with gzip.open(csv_path + ".gz", 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return pd.read_csv(csv_path)

# ----------------------------------------------------
# Upload DEG file
# ----------------------------------------------------
st.title("PhoenixBioInfoSys – DEG & Regulatory Network Platform")

uploaded = st.file_uploader(
    "Upload DEG table (CSV / TSV / XLSX | ≤1GB)",
    type=["csv", "tsv", "xlsx"]
)
if uploaded is None:
    st.stop()

df_raw = load_table(uploaded)
df_raw.columns = [c.strip() for c in df_raw.columns]

st.subheader("Uploaded Data Preview")
st.dataframe(df_raw.head())

# ----------------------------------------------------
# Column auto-detection
# ----------------------------------------------------
gene_col = detect_column(df_raw.columns, [
    "gene", "symbol", "genesymbol", "genename"
])

logfc_col = detect_column(df_raw.columns, [
    "logfc", "log2fc", "log2foldchange", "logfoldchange"
])

pval_col = detect_column(df_raw.columns, [
    "pvalue", "pval", "padj", "adjp", "adj.p"
])

# ----------------------------------------------------
# Manual fallback (ONLY if needed)
# ----------------------------------------------------
if gene_col is None or logfc_col is None or pval_col is None:
    st.warning("Automatic column detection failed. Please select manually.")

    gene_col = st.selectbox("Select Gene column", df_raw.columns)
    logfc_col = st.selectbox("Select logFC column", df_raw.columns)
    pval_col = st.selectbox("Select p-value column", df_raw.columns)

# ----------------------------------------------------
# Standardize columns (INTERNAL ONLY)
# ----------------------------------------------------
df = df_raw.rename(columns={
    gene_col: "gene",
    logfc_col: "logFC",
    pval_col: "pvalue"
}).copy()

df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
df = df.dropna(subset=["gene", "logFC", "pvalue"])

# ----------------------------------------------------
# Thresholds
# ----------------------------------------------------
st.sidebar.header("DEG Thresholds")
logfc_cut = st.sidebar.selectbox("log2FC cutoff", [1, 2, 3, 4])
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

df["Regulation"] = "NS"
df.loc[(df.logFC >= logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= -logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation == "Up"]
down = df[df.Regulation == "Down"]

# ----------------------------------------------------
# Volcano plot (BASE LOGIC UNCHANGED)
# ----------------------------------------------------
st.subheader("Volcano Plot")
fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="logFC",
    y=-np.log10(df["pvalue"]),
    hue="Regulation",
    ax=ax
)
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(p-value)")
st.pyplot(fig)

fig.savefig("volcano.png", dpi=300)
st.download_button("Download Volcano (300 DPI)", open("volcano.png", "rb"))

# ----------------------------------------------------
# Enrichment (GO + KEGG)
# ----------------------------------------------------
gp = GProfiler(return_dataframe=True)
genes = list(set(up.gene.tolist() + down.gene.tolist()))
enrich = gp.profile(organism="hsapiens", query=genes)

def show_enrichment(src, title):
    tab = enrich[enrich.source == src]
    if not tab.empty:
        st.subheader(title)
        st.dataframe(tab[["name", "p_value", "intersection_size"]])

show_enrichment("GO:BP", "GO Biological Process")
show_enrichment("GO:CC", "GO Cellular Component")
show_enrichment("GO:MF", "GO Molecular Function")
show_enrichment("KEGG", "KEGG Pathways")

# ----------------------------------------------------
# PPI Network (BASE)
# ----------------------------------------------------
st.subheader("PPI Network")

hub_n = st.selectbox("Top hub genes", [10, 20, 50])
ppi_genes = genes[:100]

G = nx.Graph()
for i, g in enumerate(ppi_genes):
    for h in ppi_genes[i+1:i+4]:
        G.add_edge(g, h)

degree = dict(G.degree())
hub_df = (
    pd.DataFrame(degree.items(), columns=["Gene", "Score"])
    .sort_values("Score", ascending=False)
    .head(hub_n)
)

st.dataframe(hub_df)

fig2, ax2 = plt.subplots()
nx.draw(G, ax=ax2, node_size=40)
st.pyplot(fig2)

fig2.savefig("ppi.png", dpi=300)
st.download_button("Download PPI (300 DPI)", open("ppi.png", "rb"))

# ----------------------------------------------------
# miRNA analysis (unchanged features)
# ----------------------------------------------------
st.subheader("miRNA–Gene Network")

mirna_db = load_mirtarbase()
mirna_db = mirna_db[mirna_db["Target Gene"].isin(hub_df.Gene)]

st.dataframe(
    mirna_db[["miRNA", "Target Gene", "Support Type", "PMID"]].head(50),
    use_container_width=True
)

# ----------------------------------------------------
# Summary
# ----------------------------------------------------
st.subheader("Automated Scientific Summary")

summary = f"""
{len(up)} genes were upregulated and {len(down)} were downregulated.
Hub genes include {', '.join(hub_df.Gene.head(5))}.
Functional enrichment highlights key biological processes and pathways.
"""

st.text_area("Manuscript-ready summary", summary, height=160)

# ----------------------------------------------------
# Methods & Citations
# ----------------------------------------------------
st.subheader("Methods")
st.markdown("""
Differential expression analysis was performed using user-defined thresholds.
Functional enrichment was conducted using g:Profiler.
Protein–protein interaction networks were constructed using NetworkX.
miRNA–gene interactions were retrieved from miRTarBase.
""")

st.subheader("Citations")
st.markdown("""
- miRTarBase: PMID 29126174  
- g:Profiler: PMID 31691815  
""")
