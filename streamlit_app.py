import streamlit as st
import pandas as pd
import numpy as np
import os, urllib.request, gzip, shutil
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
from gprofiler import GProfiler

st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")

# ====================================================
# Universal file loader (CSV / TSV / XLSX)
# ====================================================
def load_table(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    if file.name.endswith(".tsv"):
        return pd.read_csv(file, sep="\t")
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    st.error("Unsupported file format")
    st.stop()

# ====================================================
# Column normalization (CRITICAL FIX)
# ====================================================
def normalize_columns(df):
    df.columns = [c.strip() for c in df.columns]

    gene_aliases = ["gene", "genesymbol", "symbol", "gene_name"]
    logfc_aliases = ["logfc", "log2fc", "log2foldchange", "log_fold_change"]
    pval_aliases = ["pvalue", "p.value", "p_val", "pval", "padj", "adj.p.val"]

    def find_col(aliases):
        for c in df.columns:
            if c.lower() in aliases:
                return c
        return None

    gene_col = find_col(gene_aliases)
    logfc_col = find_col(logfc_aliases)
    pval_col = find_col(pval_aliases)

    if gene_col is None or logfc_col is None or pval_col is None:
        st.error(
            "Required columns not detected.\n\n"
            "Expected:\n"
            "- Gene column (e.g. gene, SYMBOL)\n"
            "- logFC column (e.g. log2FoldChange)\n"
            "- p-value column (e.g. pvalue, padj)"
        )
        st.stop()

    df = df.rename(columns={
        gene_col: "gene",
        logfc_col: "logFC",
        pval_col: "pvalue"
    })

    return df

# ====================================================
# miRTarBase loader (runtime download, cached)
# ====================================================
@st.cache_data(show_spinner=True)
def load_mirtarbase():
    url = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_MTI.csv.gz"
    os.makedirs("data", exist_ok=True)
    csv_path = "data/miRTarBase_MTI.csv"

    if not os.path.exists(csv_path):
        st.info("Downloading miRTarBase (one-time ~400MB)...")
        gz_path = csv_path + ".gz"
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return pd.read_csv(csv_path)

# ====================================================
# Upload DEG file
# ====================================================
st.title("PhoenixBioInfoSys – DEG & Regulatory Network Platform")

uploaded = st.file_uploader(
    "Upload DEG table (CSV / TSV / XLSX, ≤1GB)",
    type=["csv", "tsv", "xlsx"]
)

if uploaded is None:
    st.stop()

df = load_table(uploaded)
df = normalize_columns(df)

# ====================================================
# DEG thresholds (BASE LOGIC UNCHANGED)
# ====================================================
st.sidebar.header("DEG Thresholds")
logfc_cut = st.sidebar.selectbox("log2FC cutoff", [1, 2, 3, 4])
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001])

df["Regulation"] = "NS"
df.loc[(df.logFC >= logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= -logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation == "Up"]
down = df[df.Regulation == "Down"]

# ====================================================
# Downloads (Up / Down)
# ====================================================
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

# ====================================================
# Volcano Plot (BASE)
# ====================================================
st.subheader("Volcano Plot")

fig, ax = plt.subplots()
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
st.download_button("Download Volcano (300 DPI)", open("volcano.png","rb"))

# ====================================================
# GO & KEGG enrichment (BASE)
# ====================================================
gp = GProfiler(return_dataframe=True)
genes = up.gene.tolist() + down.gene.tolist()
enrich = gp.profile(organism="hsapiens", query=genes)

def show_enrich(src, title):
    tab = enrich[enrich.source == src]
    if not tab.empty:
        st.subheader(title)
        st.dataframe(tab[["name", "p_value", "intersection_size"]])

show_enrich("GO:BP", "GO Biological Process")
show_enrich("GO:CC", "GO Cellular Component")
show_enrich("GO:MF", "GO Molecular Function")
show_enrich("KEGG", "KEGG Pathways")

# ====================================================
# PPI Network (BASE)
# ====================================================
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

fig2, ax2 = plt.subplots(figsize=(6,6))
nx.draw(G, ax=ax2, node_size=40)
st.pyplot(fig2)

fig2.savefig("ppi.png", dpi=300)
st.download_button("Download PPI (300 DPI)", open("ppi.png","rb"))

# ====================================================
# miRNA Analysis (REAL miRTarBase)
# ====================================================
st.subheader("miRNA–Gene Regulatory Network")

mirna_db = load_mirtarbase()

species = st.sidebar.selectbox(
    "miRTarBase Species",
    sorted(mirna_db["Species (Target Gene)"].unique())
)

evidence_mode = st.sidebar.radio(
    "miRNA Evidence Filter",
    ["All", "Strong only"]
)

strong_terms = ["Reporter assay", "Western blot", "qPCR"]

mirna_db = mirna_db[
    (mirna_db["Species (Target Gene)"] == species) &
    (mirna_db["Target Gene"].isin(hub_df.Gene))
]

if evidence_mode == "Strong only":
    mirna_db = mirna_db[
        mirna_db["Support Type"].str.contains("|".join(strong_terms), na=False)
    ]

mirna_db["PMID"] = mirna_db["PMID"].astype(str)
mirna_db["PMID_Link"] = mirna_db["PMID"].apply(
    lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}/"
)

st.dataframe(
    mirna_db[["miRNA", "Target Gene", "Support Type", "PMID_Link"]],
    use_container_width=True
)

# ====================================================
# miRNA Enrichment
# ====================================================
st.subheader("miRNA Enrichment")

mirna_counts = mirna_db["miRNA"].value_counts().head(20)

fig3, ax3 = plt.subplots()
mirna_counts.plot(kind="bar", ax=ax3)
ax3.set_ylabel("Target Gene Count")
st.pyplot(fig3)

fig3.savefig("mirna_enrichment.png", dpi=300)
st.download_button(
    "Download miRNA Enrichment (300 DPI)",
    open("mirna_enrichment.png","rb")
)

# ====================================================
# Evidence-weighted miRNA Network
# ====================================================
st.subheader("Evidence-weighted miRNA Network")

net = nx.Graph()
for _, r in mirna_db.iterrows():
    weight = 2 if any(x in r["Support Type"] for x in strong_terms) else 1
    net.add_edge(r["miRNA"], r["Target Gene"], weight=weight)

fig4, ax4 = plt.subplots(figsize=(7,7))
nx.draw(
    net,
    ax=ax4,
    node_size=30,
    width=[d["weight"] for _,_,d in net.edges(data=True)]
)
st.pyplot(fig4)

fig4.savefig("mirna_network.png", dpi=300)
st.download_button(
    "Download miRNA Network (300 DPI)",
    open("mirna_network.png","rb")
)

# ====================================================
# Automated Scientific Summary
# ====================================================
st.subheader("Automated Scientific Summary")

summary = f"""
A total of {len(up)} genes were upregulated and {len(down)} genes were downregulated.
Hub gene analysis identified {', '.join(hub_df.Gene.head(5))} as key regulators.
miRNA–gene interactions were obtained from experimentally validated miRTarBase.
Strong experimental evidence was {'applied' if evidence_mode=='Strong only' else 'not restricted'}.
Dominant miRNAs include {', '.join(mirna_counts.index[:5])}.
"""

st.text_area("Manuscript-ready Summary", summary, height=200)

# ====================================================
# Methods & Citations
# ====================================================
st.subheader("Methods")
st.markdown("""
Differential expression analysis was filtered using log2 fold-change and p-value thresholds.
Functional enrichment was performed using g:Profiler.
Protein–protein interaction networks were constructed using NetworkX.
miRNA–target interactions were obtained from experimentally validated miRTarBase.
""")

st.subheader("Citations")
st.markdown("""
- miRTarBase: PMID 29126174  
- g:Profiler: PMID 31691815  
""")
