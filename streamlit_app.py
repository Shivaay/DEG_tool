import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import os, urllib.request, gzip, shutil
from gprofiler import GProfiler
from fpdf import FPDF

st.set_page_config(layout="wide", page_title="PhoenixBioInfoSys DEG")

# =====================================================
# Upload  (BASE CODE — UNCHANGED)
# =====================================================
st.title("PhoenixBioInfoSys – DEG & Network Analysis Platform")

uploaded_file = st.file_uploader(
    "Upload DEG table (CSV / TSV / XLSX, ≤1GB)",
    type=["csv", "tsv", "xlsx"]
)
if uploaded_file is None:
    st.stop()

# --- ADDITION: flexible file loading (NON-BREAKING)
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".tsv"):
    df = pd.read_csv(uploaded_file, sep="\t")
elif uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
else:
    st.error("Unsupported file format")
    st.stop()

# --- ADDITION: Excel safety cleanup
df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed", case=False)]
df.columns = df.columns.astype(str).str.strip()

# Required columns check (BASE LOGIC)
required_cols = ["gene", "logFC", "pvalue"]
if not all(col in df.columns for col in required_cols):
    st.error("Input must contain: gene, logFC, pvalue")
    st.stop()

# =====================================================
# Thresholds  (BASE CODE — UNCHANGED)
# =====================================================
st.sidebar.header("DEG Thresholds")
logfc_cut = st.sidebar.selectbox("log2FC cutoff", [1, 2, 3, 4], index=0)
p_cut = st.sidebar.selectbox("p-value cutoff", [0.05, 0.01, 0.001], index=0)

df["Regulation"] = "NS"
df.loc[(df.logFC >= logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Up"
df.loc[(df.logFC <= -logfc_cut) & (df.pvalue <= p_cut), "Regulation"] = "Down"

up = df[df.Regulation == "Up"]
down = df[df.Regulation == "Down"]

# =====================================================
# Downloads  (BASE CODE — UNCHANGED)
# =====================================================
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

# =====================================================
# Volcano Plot  (BASE CODE — UNCHANGED)
# =====================================================
st.subheader("Volcano Plot")
fig, ax = plt.subplots(figsize=(6,5))
sns.scatterplot(
    data=df, x="logFC", y=-np.log10(df.pvalue),
    hue="Regulation", ax=ax
)
ax.set_xlabel("log2 Fold Change")
ax.set_ylabel("-log10(p-value)")
st.pyplot(fig)

fig.savefig("volcano.png", dpi=300)
st.download_button(
    "Download Volcano (300 DPI)",
    open("volcano.png","rb"),
    "volcano.png"
)

# =====================================================
# Enrichment (GO + KEGG)  (BASE CODE — UNCHANGED)
# =====================================================
gp = GProfiler(return_dataframe=True)
genes = up.gene.tolist() + down.gene.tolist()

enrich = gp.profile(organism="hsapiens", query=genes) if genes else pd.DataFrame()

def show_table(source, title):
    if enrich.empty:
        return
    tab = enrich[enrich.source == source]
    if not tab.empty:
        st.subheader(title)
        st.dataframe(tab[["name","p_value","intersection_size"]])

show_table("GO:BP", "GO Biological Process")
show_table("GO:CC", "GO Cellular Component")
show_table("GO:MF", "GO Molecular Function")
show_table("KEGG", "KEGG Pathways")

# =====================================================
# PPI Network  (BASE CODE — UNCHANGED)
# =====================================================
st.subheader("PPI Network")

ppi_mode = st.sidebar.selectbox("Hub scoring method", ["Degree", "MCC"])

ppi_genes = genes[:100]
edges = [(a,b) for i,a in enumerate(ppi_genes) for b in ppi_genes[i+1:i+4]]

G = nx.Graph()
G.add_edges_from(edges)

scores = dict(G.degree()) if ppi_mode == "Degree" else nx.clustering(G)

hub_n = st.sidebar.selectbox("Top hub genes", [10,20,50])
hub_df = (
    pd.DataFrame(scores.items(), columns=["Gene","Score"])
    .sort_values("Score", ascending=False)
    .head(hub_n)
)

st.dataframe(hub_df)

fig2, ax2 = plt.subplots(figsize=(6,6))
nx.draw(G, ax=ax2, node_size=30)
st.pyplot(fig2)

fig2.savefig("ppi.png", dpi=300)
st.download_button(
    "Download PPI (300 DPI)",
    open("ppi.png","rb"),
    "ppi.png"
)

# =====================================================
# ADDITIONAL FEATURE: miRTarBase Loader (SAFE)
# =====================================================
@st.cache_data(show_spinner=True)
def load_mirtarbase():
    url = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_MTI.csv.gz"
    os.makedirs("data", exist_ok=True)
    csv_path = "data/miRTarBase_MTI.csv"

    if not os.path.exists(csv_path):
        urllib.request.urlretrieve(url, csv_path + ".gz")
        with gzip.open(csv_path + ".gz", 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    return pd.read_csv(csv_path)

# =====================================================
# miRNA Network (EXTENDED, BASE PRESERVED)
# =====================================================
st.subheader("miRNA–Gene Network (miRTarBase)")

enable_mirna = st.sidebar.checkbox("Enable miRNA analysis", value=False)

if enable_mirna:
    mirna_db = load_mirtarbase()

    species = st.sidebar.selectbox(
        "miRNA Species",
        sorted(mirna_db["Species (Target Gene)"].dropna().unique())
    )

    evidence_mode = st.sidebar.radio(
        "miRNA Evidence",
        ["All", "Strong only"]
    )

    strong_terms = ["Reporter assay", "Western blot", "qPCR"]

    mirna_hits = mirna_db[
        (mirna_db["Target Gene"].isin(hub_df.Gene)) &
        (mirna_db["Species (Target Gene)"] == species)
    ]

    if evidence_mode == "Strong only":
        mirna_hits = mirna_hits[
            mirna_hits["Support Type"].str.contains(
                "|".join(strong_terms), na=False
            )
        ]

    if mirna_hits.empty:
        st.warning("No miRNA interactions found.")
    else:
        mirna_hits["PMID"] = mirna_hits["PMID"].astype(str)
        mirna_hits["PMID_Link"] = mirna_hits["PMID"].apply(
            lambda x: f"https://pubmed.ncbi.nlm.nih.gov/{x}/"
        )

        st.dataframe(
            mirna_hits[["miRNA","Target Gene","Support Type","PMID_Link"]],
            use_container_width=True
        )

        # --- miRNA enrichment
        st.subheader("miRNA Enrichment")
        mirna_counts = mirna_hits["miRNA"].value_counts().head(20)

        fig3, ax3 = plt.subplots()
        mirna_counts.plot(kind="bar", ax=ax3)
        ax3.set_ylabel("Target gene count")
        st.pyplot(fig3)

        fig3.savefig("mirna_enrichment.png", dpi=300)
        st.download_button(
            "Download miRNA Enrichment (300 DPI)",
            open("mirna_enrichment.png","rb")
        )

        # --- Evidence-weighted network
        st.subheader("Evidence-weighted miRNA Network")
        net = nx.Graph()
        for _, r in mirna_hits.iterrows():
            w = 2 if any(t in r["Support Type"] for t in strong_terms) else 1
            net.add_edge(r["miRNA"], r["Target Gene"], weight=w)

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

# =====================================================
# AI Summary  (BASE CODE — UNCHANGED)
# =====================================================
st.subheader("Automated Scientific Summary")

summary = f"""
A total of {len(up)} genes were upregulated and {len(down)} were downregulated.
Hub gene analysis identified {', '.join(hub_df.Gene.head(5))} as central regulators.
Validated miRNA interactions were obtained from miRTarBase.
"""

st.text_area("Manuscript-ready summary", summary, height=200)

# =====================================================
# PDF Report  (BASE CODE — UNCHANGED)
# =====================================================
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=10)
pdf.multi_cell(0,8, summary)
pdf.output("report.pdf")

st.download_button(
    "Download PDF Report",
    open("report.pdf","rb"),
    "DEG_Report.pdf"
)

# =====================================================
# Methods & Citations  (BASE CODE — UNCHANGED)
# =====================================================
st.subheader("Methods")
st.markdown("""
DEGs were filtered using log2FC and p-value thresholds.
Functional enrichment was performed using g:Profiler.
PPI networks were constructed using NetworkX.
miRNA interactions were obtained from miRTarBase (validated).
TF and drug interactions were inferred using Enrichr.
""")

st.subheader("Citations")
st.markdown("""
- miRTarBase: PMID 29126174  
- g:Profiler: PMID 31691815  
- Enrichr: PMID 33780170  
- STRING: PMID 36370105  
""")
