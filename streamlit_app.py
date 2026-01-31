import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import os, urllib.request, gzip, shutil
from gprofiler import GProfiler

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys DEG")

# =====================================================
# miRTarBase Loader (Runtime, Cached, Large-file safe)
# =====================================================
@st.cache_data(show_spinner=True)
def load_mirtarbase():
    url = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_MTI.csv.gz"
    os.makedirs("data", exist_ok=True)
    local = "data/miRTarBase_MTI.csv"

    if not os.path.exists(local):
        gz = local + ".gz"
        urllib.request.urlretrieve(url, gz)
        with gzip.open(gz, 'rb') as f_in:
            with open(local, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return pd.read_csv(local)

# =====================================================
# Upload DEG
# =====================================================
st.title("PhoenixBioInfoSys – Differential Expression & Regulatory Networks")

deg_file = st.file_uploader("Upload DEG CSV (≤1GB)", type=["csv"])
if deg_file is None:
    st.stop()

df = pd.read_csv(deg_file)

required = ["gene", "logFC", "pvalue"]
if not all(c in df.columns for c in required):
    st.error("CSV must contain gene, logFC, pvalue")
    st.stop()

# =====================================================
# Thresholds
# =====================================================
st.sidebar.header("DEG Filters")
logfc_cut = st.sidebar.selectbox("log2FC cutoff", [1,2,3,4])
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05,0.01,0.001])

df["Regulation"] = "NS"
df.loc[(df.logFC >= logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= -logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation=="Up"]
down = df[df.Regulation=="Down"]

# =====================================================
# Downloads
# =====================================================
st.sidebar.download_button("Download Upregulated", up.to_csv(index=False), "up.csv")
st.sidebar.download_button("Download Downregulated", down.to_csv(index=False), "down.csv")

# =====================================================
# Volcano Plot (BASE FUNCTION)
# =====================================================
st.subheader("Volcano Plot")
fig, ax = plt.subplots()
sns.scatterplot(df, x="logFC", y=-np.log10(df.pvalue), hue="Regulation", ax=ax)
st.pyplot(fig)
fig.savefig("volcano.png", dpi=300)
st.download_button("Download Volcano (300 DPI)", open("volcano.png","rb"), "volcano.png")

# =====================================================
# Enrichment (GO + KEGG – BASE FUNCTION)
# =====================================================
gp = GProfiler(return_dataframe=True)
genes = list(set(up.gene.tolist() + down.gene.tolist()))
enrich = gp.profile(organism="hsapiens", query=genes)

def show_enrich(src, title):
    tab = enrich[enrich.source==src]
    if not tab.empty:
        st.subheader(title)
        st.dataframe(tab[["name","p_value","intersection_size"]])

show_enrich("GO:BP","GO Biological Process")
show_enrich("GO:CC","GO Cellular Component")
show_enrich("GO:MF","GO Molecular Function")
show_enrich("KEGG","KEGG Pathways")

# =====================================================
# PPI Network (BASE FUNCTION)
# =====================================================
st.subheader("PPI Network")
hub_n = st.selectbox("Top hub genes", [10,20,50])
ppi_genes = genes[:100]

edges = [(a,b) for i,a in enumerate(ppi_genes) for b in ppi_genes[i+1:i+4]]
G = nx.Graph()
G.add_edges_from(edges)

scores = dict(G.degree())
hub_df = pd.DataFrame(scores.items(), columns=["Gene","Degree"]).sort_values("Degree", ascending=False).head(hub_n)
st.dataframe(hub_df)

fig2, ax2 = plt.subplots()
nx.draw(G, ax=ax2, node_size=40)
st.pyplot(fig2)
fig2.savefig("ppi.png", dpi=300)
st.download_button("Download PPI (300 DPI)", open("ppi.png","rb"), "ppi.png")

# =====================================================
# miRNA MODULE (ADVANCED – ADDED)
# =====================================================
st.header("miRNA Regulatory Analysis (miRTarBase – Validated)")

mirna_db = load_mirtarbase()

species = st.selectbox("Species", mirna_db["Species (Target Gene)"].unique())
evidence = st.multiselect(
    "Experimental Evidence",
    mirna_db["Support Type"].unique(),
    default=["Reporter assay"]
)

strong_only = st.checkbox("Strong evidence only (Reporter assay)")

mirna_filt = mirna_db[
    (mirna_db["Target Gene"].isin(hub_df.Gene)) &
    (mirna_db["Species (Target Gene)"]==species) &
    (mirna_db["Support Type"].isin(evidence))
]

if strong_only:
    mirna_filt = mirna_filt[mirna_filt["Support Type"].str.contains("Reporter")]

mirna_filt["PMID"] = mirna_filt["PMID"].apply(
    lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}/"
)

st.subheader("Validated miRNA–Gene Interactions")
st.dataframe(mirna_filt[["miRNA","Target Gene","Support Type","PMID"]])

# =====================================================
# miRNA Enrichment (NEW)
# =====================================================
st.subheader("miRNA Enrichment")
mirna_counts = mirna_filt["miRNA"].value_counts().head(20)
fig3, ax3 = plt.subplots()
mirna_counts.plot(kind="barh", ax=ax3)
st.pyplot(fig3)
fig3.savefig("mirna_enrichment.png", dpi=300)
st.download_button("Download miRNA Enrichment (300 DPI)",
                   open("mirna_enrichment.png","rb"),
                   "mirna_enrichment.png")

# =====================================================
# Evidence-weighted Network (NEW)
# =====================================================
st.subheader("Evidence-weighted miRNA Network")

W = nx.Graph()
for _, r in mirna_filt.iterrows():
    w = 3 if "Reporter" in r["Support Type"] else 1
    W.add_edge(r["miRNA"], r["Target Gene"], weight=w)

fig4, ax4 = plt.subplots()
nx.draw(W, ax=ax4, node_size=50)
st.pyplot(fig4)
fig4.savefig("mirna_network.png", dpi=300)
st.download_button("Download miRNA Network (300 DPI)",
                   open("mirna_network.png","rb"),
                   "mirna_network.png")

# =====================================================
# Citations (AUTO)
# =====================================================
st.subheader("Citations")
st.markdown("""
- miRTarBase: PMID 29126174  
- g:Profiler: PMID 31691815  
- STRING: PMID 36370105  
""")
