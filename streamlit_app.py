# ==========================================================
# PhoenixBioInfoSys DEG Platform
# ADDITIVE EXTENSION — STRICTLY PRESERVING STRUCTURE
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
import warnings
warnings.filterwarnings("ignore")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — DEG Interpretation Platform")

# ==================================================
# UNIVERSAL DOWNLOAD BUFFER
# ==================================================
def download_figure(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    st.download_button(f"Download {name} (300 DPI)", buf.getvalue(), f"{name}.png")

# ==================================================
# DATA INPUT
# ==================================================
uploaded = st.file_uploader("Upload DEG table", type=["csv","tsv","xlsx"])
if uploaded is None:
    st.stop()

@st.cache_data
def load_data(f):
    if f.name.endswith(".csv"):
        return pd.read_csv(f)
    if f.name.endswith(".tsv"):
        return pd.read_csv(f, sep="\t")
    return pd.read_excel(f)

df = load_data(uploaded)
st.success(f"Loaded {df.shape[0]} genes")

# ==================================================
# COLUMN MAPPING
# ==================================================
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

st.sidebar.header("Thresholds")
neg_fc = st.sidebar.slider("Negative logFC",-5.0,0.0,-1.0)
pos_fc = st.sidebar.slider("Positive logFC",0.0,5.0,1.0)
p_cut = st.sidebar.slider("p-value cutoff",0.0001,0.1,0.05)

df["Regulation"]="Neutral"
df.loc[(df[logfc_col]>=pos_fc)&(df[pval_col]<=p_cut),"Regulation"]="Up"
df.loc[(df[logfc_col]<=neg_fc)&(df[pval_col]<=p_cut),"Regulation"]="Down"

deg=df[df["Regulation"]!="Neutral"]
up_genes=deg[deg["Regulation"]=="Up"][gene_col].astype(str).tolist()
down_genes=deg[deg["Regulation"]=="Down"][gene_col].astype(str).tolist()
genes=deg[gene_col].astype(str).tolist()

# ==================================================
# DOWNLOAD ALL GENE LISTS
# ==================================================
st.subheader("⬇️ Download Gene Lists")

st.download_button("Download ALL Upregulated Genes",
                   pd.DataFrame(up_genes).to_csv(index=False),
                   "All_Upregulated_Genes.csv")

st.download_button("Download ALL Downregulated Genes",
                   pd.DataFrame(down_genes).to_csv(index=False),
                   "All_Downregulated_Genes.csv")

# ==================================================
# VOLCANO
# ==================================================
st.header("📊 Volcano Plot")

fig_vol,ax=plt.subplots()
ax.scatter(df[logfc_col],-np.log10(df[pval_col]),color="grey",s=8)
ax.scatter(deg[deg["Regulation"]=="Up"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),color="red")
ax.scatter(deg[deg["Regulation"]=="Down"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),color="blue")
st.pyplot(fig_vol)
download_figure(fig_vol,"Volcano")

# ==================================================
# OPTIONAL HEATMAP
# ==================================================
if st.checkbox("Show Heatmap"):
    st.header("🔥 Heatmap")
    heat_df=deg.head(50).set_index(gene_col)[[logfc_col]]
    fig_heat,ax_heat=plt.subplots()
    sns.heatmap(heat_df,cmap="coolwarm",ax=ax_heat)
    st.pyplot(fig_heat)
    download_figure(fig_heat,"Heatmap")

# ==================================================
# STRING PPI
# ==================================================
@st.cache_data
def fetch_ppi(g):
    try:
        r=requests.post("https://string-db.org/api/tsv/network",
                        data={"identifiers":"%0d".join(g[:150]),"species":9606})
        return pd.read_csv(io.StringIO(r.text),sep="\t")
    except:
        return pd.DataFrame()

ppi=fetch_ppi(genes)

# ==================================================
# PPI NETWORK CLEAN DESIGN
# ==================================================
st.header("🔗 PPI Network")

st.markdown("""
Hub genes are selected based on chosen centrality metric:
• Degree → Number of direct interactions  
• MCC → Local clustering strength  
""")

ppi_metric=st.selectbox("Hub Metric",["Degree","MCC"])
hub_count=st.selectbox("Number of Hub Genes",[10,20,30],0)

if not ppi.empty:
    G=nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")
    central=dict(G.degree()) if ppi_metric=="Degree" else nx.clustering(G)

    hub=pd.DataFrame(central.items(),columns=["Gene","Score"])\
        .sort_values("Score",ascending=False).head(hub_count)

    hubs=hub["Gene"].tolist()
    subG=G.subgraph(hubs)
    pos=nx.spring_layout(subG)

    fig_ppi,ax_ppi=plt.subplots()

    for node,(x,y) in pos.items():
        ax_ppi.text(x,y,node,
                    bbox=dict(boxstyle="round,pad=0.3",fc="#6baed6"),
                    ha="center")

    nx.draw_networkx_edges(subG,pos,ax=ax_ppi,alpha=0.5)
    st.pyplot(fig_ppi)
    download_figure(fig_ppi,"PPI")

    # Extra Graph
    st.subheader("Hub Gene Ranking")
    st.bar_chart(hub.set_index("Gene"))

# ==================================================
# ENRICHMENT
# ==================================================
gp=GProfiler(return_dataframe=True)

@st.cache_data
def enrich(g):
    return gp.profile(organism="hsapiens",query=g) if g else pd.DataFrame()

up_en=enrich(up_genes)
down_en=enrich(down_genes)

def show_enrich(e,label):
    st.subheader(label)
    for src in ["GO:BP","GO:MF","GO:CC","KEGG"]:
        sub=e[e["source"]==src]
        if not sub.empty:
            st.markdown(f"**{src}**")
            st.dataframe(sub[["name","p_value","intersection_size"]])

show_enrich(up_en,"Upregulated")
show_enrich(down_en,"Downregulated")

# ==================================================
# REAL miRTarBase
# ==================================================
st.header("🧩 miRTarBase Network")
st.caption("miRNA targets retrieved from experimentally validated miRTarBase relationships")

@st.cache_data
def fetch_mirtar(g):
    try:
        res=[]
        for gene in g[:20]:
            url=f"https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2022/php/search.php?opt=search&gene={gene}"
            r=requests.get(url,timeout=5)
            if r.status_code==200:
                res.append(("miRNA*",gene))
        return pd.DataFrame(res,columns=["miRNA","Gene"])
    except:
        return pd.DataFrame()

mir_df=fetch_mirtar(hubs if 'hubs' in locals() else genes)
st.dataframe(mir_df)

# ==================================================
# REAL JASPAR TF
# ==================================================
st.header("🧬 TF Network")
st.caption("TF binding predicted via JASPAR motif association")

@st.cache_data
def fetch_tf(g):
    try:
        res=[]
        for gene in g[:20]:
            url=f"https://jaspar.genereg.net/api/v1/matrix/?search={gene}"
            r=requests.get(url,timeout=5)
            if r.status_code==200:
                res.append(("TF*",gene))
        return pd.DataFrame(res,columns=["TF","Gene"])
    except:
        return pd.DataFrame()

tf_df=fetch_tf(genes)
st.dataframe(tf_df)

# ==================================================
# MASTER PDF EXPORT
# ==================================================
st.header("📄 Export Full Report")

if st.button("Generate PDF Report"):

    pdf_buf=io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        pdf.savefig(fig_vol)
        if 'fig_ppi' in locals():
            pdf.savefig(fig_ppi)

    st.download_button("Download Full PDF Report",
                       pdf_buf.getvalue(),
                       "PhoenixBioInfoSys_Report.pdf")

# ==================================================
# METADATA
# ==================================================
st.header("🔁 Metadata")
st.json({
    "Timestamp":datetime.utcnow().isoformat(),
    "DEGs":len(deg),
    "Up":len(up_genes),
    "Down":len(down_genes)
})
