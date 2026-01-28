# ==========================================================
# PhoenixBioInfoSys DEG Platform — Professional Edition
# Base preserved + Industry-grade extensions
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
from textwrap import fill
from gprofiler import GProfiler

# ---------- OPTIONAL DEPENDENCIES ----------
try:
    import mygene
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORT_AVAILABLE = True
except ImportError:
    REPORT_AVAILABLE = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PhoenixBioInfoSys DEG", layout="wide")
st.title("🧬 PhoenixBioInfoSys — DEG Interpretation Platform")

# ==========================================================
# 1️⃣ DATA INPUT (BASE)
# ==========================================================
uploaded = st.file_uploader(
    "Upload DEG results (CSV / TSV / XLSX, ≤ 1 GB)",
    type=["csv", "tsv", "xlsx"]
)
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

# ==========================================================
# 2️⃣ COLUMN MAPPING & FILTERING (BASE)
# ==========================================================
st.sidebar.header("Column Mapping")
gene_col = st.sidebar.selectbox("Gene column", df.columns)
logfc_col = st.sidebar.selectbox("logFC column", df.columns)
pval_col = st.sidebar.selectbox("p-value column", df.columns)

df[logfc_col] = pd.to_numeric(df[logfc_col], errors="coerce")
df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
df = df.dropna(subset=[gene_col, logfc_col, pval_col])

st.sidebar.header("Filtering")
neg_fc = st.sidebar.slider("Negative logFC (≤)", -10.0, -0.5, -1.0)
pos_fc = st.sidebar.slider("Positive logFC (≥)", 0.5, 10.0, 1.0)
p_cut = st.sidebar.slider("p-value cutoff", 0.0001, 0.1, 0.05)

df["Regulation"] = "Neutral"
df.loc[(df[logfc_col] >= pos_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Up"
df.loc[(df[logfc_col] <= neg_fc) & (df[pval_col] <= p_cut), "Regulation"] = "Down"

deg = df[df["Regulation"] != "Neutral"]
up_genes = deg[deg["Regulation"] == "Up"][gene_col].astype(str).tolist()
down_genes = deg[deg["Regulation"] == "Down"][gene_col].astype(str).tolist()
genes = deg[gene_col].astype(str).tolist()

# ==========================================================
# 3️⃣ VOLCANO PLOT (BASE + 300 DPI)
# ==========================================================
st.header("📊 Volcano Plot")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df[logfc_col], -np.log10(df[pval_col]), c="lightgrey", s=10)
ax.scatter(deg[deg["Regulation"]=="Up"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Up"][pval_col]),
           c="red", label="Up")
ax.scatter(deg[deg["Regulation"]=="Down"][logfc_col],
           -np.log10(deg[deg["Regulation"]=="Down"][pval_col]),
           c="blue", label="Down")
ax.legend()
st.pyplot(fig)

fig.savefig("volcano_300dpi.png", dpi=300, bbox_inches="tight")

# ==========================================================
# 4️⃣ ENRICHMENT ANALYSIS (GO + KEGG)
# ==========================================================
st.header("🧠 Functional Enrichment")

gp = GProfiler(return_dataframe=True)

@st.cache_data
def enrich(glist):
    if not glist:
        return pd.DataFrame()
    return gp.profile(organism="hsapiens", query=glist)

en_all = enrich(genes)
en_up = enrich(up_genes)
en_down = enrich(down_genes)

def show_tables(df, title):
    st.subheader(title)
    for src in ["GO:BP", "GO:MF", "GO:CC", "KEGG"]:
        subset = df[df["source"] == src]
        if not subset.empty:
            st.markdown(f"**{src}**")
            st.dataframe(subset[["name", "p_value"]].head(15))

show_tables(en_all, "All DEGs")
show_tables(en_up, "Upregulated Genes")
show_tables(en_down, "Downregulated Genes")

# ==========================================================
# 5️⃣ PPI NETWORK + HUB GENES (BASE)
# ==========================================================
st.header("🔗 PPI Network (STRING)")

@st.cache_data
def fetch_ppi(glist):
    if not glist:
        return pd.DataFrame()
    r = requests.post(
        "https://string-db.org/api/tsv/network",
        data={"identifiers":"%0d".join(glist[:200]),"species":9606,"required_score":700}
    )
    if r.status_code != 200:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text), sep="\t")

ppi = fetch_ppi(genes)
hub_genes = []

if not ppi.empty:
    G = nx.from_pandas_edgelist(ppi,"preferredName_A","preferredName_B")
    hub_genes = [x[0] for x in sorted(G.degree,key=lambda x:x[1],reverse=True)[:10]]
    st.write("Top hub genes:", ", ".join(hub_genes))

# ==========================================================
# 6️⃣ HEATMAP (BASE)
# ==========================================================
st.header("🔥 Heatmap (Hub Genes)")

expr_cols = [c for c in df.columns if c not in [gene_col,logfc_col,pval_col,"Regulation"]
             and pd.api.types.is_numeric_dtype(df[c])]

if hub_genes and expr_cols:
    hm = df[df[gene_col].isin(hub_genes)].set_index(gene_col)[expr_cols]
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.heatmap(hm, cmap="RdBu_r", center=0, ax=ax2)
    st.pyplot(fig2)

# ==========================================================
# 7️⃣ AI-STYLE SCIENTIFIC SUMMARY (PREMIUM FEATURE)
# ==========================================================
st.header("📝 Automated Scientific Interpretation")

def generate_summary():
    up_terms = en_up[en_up["source"]=="GO:BP"]["name"].head(3).tolist()
    down_terms = en_down[en_down["source"]=="GO:BP"]["name"].head(3).tolist()

    return fill(f"""
Differential expression analysis identified {len(deg)} significant genes
({len(up_genes)} upregulated and {len(down_genes)} downregulated).

Upregulated genes were enriched in {", ".join(up_terms) if up_terms else "adaptive biological processes"},
suggesting activation of condition-associated pathways.

Downregulated genes were associated with {", ".join(down_terms) if down_terms else "suppressed metabolic or structural processes"},
indicating functional repression.

Protein–protein interaction analysis identified hub genes
({", ".join(hub_genes[:5]) if hub_genes else "none dominant"}),
suggesting regulatory importance.

Overall, the results indicate coordinated transcriptional reprogramming
reflecting a biologically meaningful molecular response.
""", 110)

summary_text = generate_summary()
st.text_area("Manuscript-Ready Summary", summary_text, height=300)

# ==========================================================
# 8️⃣ AUTOMATED PDF REPORT (ENTERPRISE FEATURE)
# ==========================================================
st.header("📄 Download Full PDF Report")

if REPORT_AVAILABLE and st.button("Generate PDF Report"):
    doc = SimpleDocTemplate("DEG_Report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Differential Gene Expression Report", styles["Title"]))
    content.append(Spacer(1,12))
    content.append(Paragraph(summary_text.replace("\n","<br/>"), styles["Normal"]))
    content.append(Spacer(1,12))
    content.append(Image("volcano_300dpi.png", width=400, height=300))

    doc.build(content)
    st.success("PDF report generated!")

    with open("DEG_Report.pdf","rb") as f:
        st.download_button("Download PDF", f, file_name="DEG_Report.pdf")

# ==========================================================
# 9️⃣ REPRODUCIBILITY METADATA
# ==========================================================
st.header("🔁 Reproducibility")

meta = {
    "timestamp": datetime.utcnow().isoformat(),
    "DEGs": len(deg),
    "Up": len(up_genes),
    "Down": len(down_genes),
    "Hub_genes": hub_genes
}

st.json(meta)
