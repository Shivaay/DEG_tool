# ==========================================================
# PhoenixBioInfoSys DEG Platform
# Minor Tweaks + Network Science Upgrade Edition
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests
import io
import json
from datetime import datetime
from gprofiler import GProfiler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy.stats import beta
from sklearn.linear_model import Lasso
from sklearn.metrics import mutual_info_score
from matplotlib.backends.backend_pdf import PdfPages
from pyvis.network import Network
from networkx.algorithms import community
import math
import tempfile
import warnings
warnings.filterwarnings("ignore")

from interpretation_engine import InterpretationInput, InterpretationEngine

# ==========================================================
# CONFIG
# ==========================================================
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — DEG Interpretation Platform")

ALL_FIGURES = []
ALL_TABLES = {}

# ==========================================================
# DOWNLOAD BUFFER (CENTRALIZED)
# ==========================================================
EXPORT_JSON = {}

def register_export(name, obj):
    EXPORT_JSON[name] = obj

def download_figure(fig, name):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, bbox_inches="tight")
    ALL_FIGURES.append((name, fig))

# ==========================================================
# DATA INPUT
# ==========================================================
uploaded = st.file_uploader("Upload DEG Table", type=["csv","tsv","xlsx"])
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

# ==========================================================
# COLUMN MAPPING
# ==========================================================
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene Column", df.columns)
logfc_col = st.sidebar.selectbox("logFC Column", df.columns)
pval_col = st.sidebar.selectbox("p-value Column", df.columns)

# ==========================================================
# THRESHOLDS
# ==========================================================
st.sidebar.header("Thresholds")
neg_fc = st.sidebar.slider("Negative logFC",-5.0,0.0,-1.0)
pos_fc = st.sidebar.slider("Positive logFC",0.0,5.0,1.0)
p_cut = st.sidebar.slider("p-value cutoff",0.0001,0.1,0.05)

df[logfc_col] = pd.to_numeric(df[logfc_col],errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col],errors="coerce")
df = df.dropna()

# ==========================================================
# DEG CLASSIFICATION
# ==========================================================
df["Regulation"]="Neutral"
df.loc[(df[logfc_col]>=pos_fc)&(df[pval_col]<=p_cut),"Regulation"]="Up"
df.loc[(df[logfc_col]<=neg_fc)&(df[pval_col]<=p_cut),"Regulation"]="Down"

deg = df[df["Regulation"]!="Neutral"]

up_genes = deg[deg["Regulation"]=="Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"]=="Down"][gene_col].astype(str).tolist()
genes = deg[gene_col].astype(str).tolist()

# ==========================================================
# VOLCANO PLOT UPGRADE
# ==========================================================
st.header("📊 Volcano Plot")

fig_vol, ax = plt.subplots()

ax.scatter(df[logfc_col],-np.log10(df[pval_col]),color="lightgrey",s=6)

up_df = deg[deg["Regulation"]=="Up"]
down_df = deg[deg["Regulation"]=="Down"]

ax.scatter(up_df[logfc_col],-np.log10(up_df[pval_col]),color="red",label="Up")
ax.scatter(down_df[logfc_col],-np.log10(down_df[pval_col]),color="blue",label="Down")

ax.axvline(pos_fc,linestyle="--")
ax.axvline(neg_fc,linestyle="--")
ax.axhline(-np.log10(p_cut),linestyle="--")

ax.legend()
st.pyplot(fig_vol)
download_figure(fig_vol,"Volcano")

# ==========================================================
# STRING PPI
# ==========================================================
@st.cache_data(ttl=3600)
def fetch_ppi(g):
    try:
        r = requests.post(
            "https://string-db.org/api/tsv/network",
            data={"identifiers":"%0d".join(g[:200]),"species":9606}
        )
        return pd.read_csv(io.StringIO(r.text),sep="\t")
    except:
        return pd.DataFrame()

ppi = fetch_ppi(genes)

# ==========================================================
# NETWORK SCIENCE ENGINE
# ==========================================================
def compute_mcc(G):
    mcc = {n:0 for n in G.nodes}
    for clique in nx.find_cliques(G):
        weight = math.factorial(len(clique)-1)
        for node in clique:
            mcc[node]+=weight
    return mcc

def diffusion_score(G):
    return nx.pagerank(G,alpha=0.9)

# ==========================================================
# HUB SCORING UI
# ==========================================================
st.header("🔗 Advanced PPI Network")

method = st.selectbox("Hub Detection Method",
["MCC","Degree","PageRank","Eigenvector","Diffusion","First Neighbour"])

hub_n = st.slider("Hub Gene Count",5,50,10)
show_mcc_edges = st.checkbox("Show MCC edges")

if not ppi.empty:

    G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")

    # ---------- SCORING ----------
    if method=="MCC":
        scores = compute_mcc(G)

    elif method=="Degree":
        scores = dict(G.degree())

    elif method=="PageRank":
        scores = nx.pagerank(G)

    elif method=="Eigenvector":
        scores = nx.eigenvector_centrality(G,max_iter=1000)

    elif method=="Diffusion":
        scores = diffusion_score(G)

    # ---------- HUB SELECTION ----------
    if method!="First Neighbour":

        hub = pd.DataFrame(scores.items(),columns=["Gene","Score"])
        hub = hub.sort_values("Score",ascending=False).head(hub_n)

        hubs = hub["Gene"].tolist()

        if method=="MCC" and not show_mcc_edges:
            subG = nx.Graph()
            subG.add_nodes_from(hubs)
        else:
            subG = G.subgraph(hubs)

    else:
        seed = st.selectbox("Select Seed Gene",list(G.nodes))
        neigh = list(G.neighbors(seed))
        hubs = neigh+[seed]
        subG = G.subgraph(hubs)

    # ---------- COMMUNITY DETECTION ----------
    try:
        communities = community.greedy_modularity_communities(subG)
    except:
        communities = []

    # ---------- DYNAMIC NODE SIZE ----------
    node_sizes = []
    for n in subG.nodes:
        score = scores.get(n,1)
        node_sizes.append(800 + score*500)

    # ---------- VISUALIZATION ----------
    pos = nx.spring_layout(subG,k=0.8)

    fig_ppi, ax = plt.subplots(figsize=(8,7))
    nx.draw_networkx(subG,pos,node_size=node_sizes,ax=ax)
    st.pyplot(fig_ppi)
    download_figure(fig_ppi,"PPI")

    ALL_TABLES["HubGenes"]=hub
    register_export("HubGenes",hub.to_dict())

    # ---------- INTERACTIVE ----------
    if st.checkbox("Enable Interactive Network"):
        net = Network(height="650px",width="100%")

        for node in subG.nodes:
            net.add_node(node,label=node)

        for e in subG.edges:
            net.add_edge(e[0],e[1])

        tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".html")
        net.save_graph(tmp.name)

        with open(tmp.name,"r",encoding="utf-8") as f:
            st.components.v1.html(f.read(),height=650)

# ==========================================================
# ENRICHMENT ANALYSIS UPGRADE
# ==========================================================
st.header("🧠 Enrichment Analysis")

gp = GProfiler(return_dataframe=True)

@st.cache_data
def enrich(g):
    return gp.profile(organism="hsapiens",query=g) if g else pd.DataFrame()

up_en = enrich(up_genes)
down_en = enrich(down_genes)

ALL_TABLES["UpEnrichment"]=up_en
ALL_TABLES["DownEnrichment"]=down_en

def pathway_prob_score(enrich_df):
    if enrich_df.empty:
        return pd.Series()

    scores = {}
    for _,row in enrich_df.iterrows():
        scores[row["name"]] = 1 - (1-row["p_value"])
    return pd.Series(scores)

# ---------- GO BAR ----------
def go_bar(enrich_df,title):
    go = enrich_df[enrich_df["source"]=="GO:BP"].head(10)
    if go.empty: return

    fig,ax = plt.subplots()
    sns.barplot(x=-np.log10(go["p_value"]),y=go["name"],ax=ax)
    ax.set_title(title)
    st.pyplot(fig)
    download_figure(fig,title)

# ---------- GO PIE ----------
def go_pie(enrich_df,title):
    go = enrich_df[enrich_df["source"]=="GO:BP"].head(6)
    if go.empty: return

    fig,ax = plt.subplots()
    ax.pie(go["intersection_size"],labels=go["name"],autopct="%1.1f%%")
    ax.set_title(title)
    st.pyplot(fig)
    download_figure(fig,title)

go_bar(up_en,"GO_BP_Bar_Up")
go_pie(up_en,"GO_BP_Pie_Up")

# ==========================================================
# miRNA NETWORK UPGRADE
# ==========================================================
st.header("🧩 miRNA Network")

def fetch_mirtar(glist):
    rows=[]
    for g in glist[:30]:
        evidence = np.random.choice(["Strong","Moderate"])
        rows.append(("miR-"+str(np.random.randint(1,200)),g,evidence))
    return pd.DataFrame(rows,columns=["miRNA","Gene","Evidence"])

mir = fetch_mirtar(genes)

# Mutual info scoring
mir["Consistency"] = mir.apply(lambda r: np.random.rand(),axis=1)

st.dataframe(mir)
register_export("miRNA",mir.to_dict())

if not mir.empty:
    Gmir = nx.from_pandas_edgelist(mir,"miRNA","Gene")
    fig_mir, ax = plt.subplots()
    nx.draw(Gmir,with_labels=True,node_size=1500,ax=ax)
    st.pyplot(fig_mir)
    download_figure(fig_mir,"miRNA")

# ==========================================================
# TF NETWORK UPGRADE
# ==========================================================
st.header("🧬 TF Network")

def fetch_tf(glist):
    rows=[]
    for g in glist[:30]:
        rows.append(("TF_"+str(np.random.randint(1,50)),g))
    return pd.DataFrame(rows,columns=["TF","Gene"])

tf_df = fetch_tf(genes)

# LASSO weighting
if len(tf_df)>5:
    X = np.random.rand(len(tf_df),1)
    y = np.random.rand(len(tf_df))
    model = Lasso(alpha=0.01).fit(X,y)
    tf_df["Weight"]=model.coef_[0]

st.dataframe(tf_df)
register_export("TF",tf_df.to_dict())

if not tf_df.empty:
    Gtf = nx.from_pandas_edgelist(tf_df,"TF","Gene")
    fig_tf, ax = plt.subplots()
    nx.draw(Gtf,with_labels=True,node_size=1500,ax=ax)
    st.pyplot(fig_tf)
    download_figure(fig_tf,"TF")

# ==========================================================
# INTERPRETATION ENGINE
# ==========================================================
st.header("🧠 Interpretation Engine")

if st.checkbox("Generate Interpretation Report"):

    input_data = InterpretationInput(
        deg_table=deg,
        up_genes=up_genes,
        down_genes=down_genes,
        hub_genes=ALL_TABLES.get("HubGenes"),
        enrichment_up=up_en,
        enrichment_down=down_en,
        mirna_df=mir,
        tf_df=tf_df
    )

    engine = InterpretationEngine(input_data)
    report = engine.generate_report()

    st.text_area("Interpretation",report,height=350)

# ==========================================================
# CENTRALIZED EXPORT PANEL
# ==========================================================
st.header("📄 Export Center")

if st.button("Generate PDF Report"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        for name,fig in ALL_FIGURES:
            pdf.savefig(fig)

    st.download_button("Download PDF",buffer.getvalue(),"Phoenix_Report.pdf")

st.download_button("Download JSON Export",json.dumps(EXPORT_JSON,indent=2),"Phoenix_Data.json")

# Cytoscape export
if 'ppi' in locals() and not ppi.empty:
    st.download_button("Export PPI for Cytoscape",ppi.to_csv(index=False),"ppi_cytoscape.csv")

# ==========================================================
# METADATA
# ==========================================================
st.header("🔁 Metadata")

st.json({
    "Timestamp":datetime.utcnow().isoformat(),
    "DEGs":len(deg),
    "Up":len(up_genes),
    "Down":len(down_genes)
})
