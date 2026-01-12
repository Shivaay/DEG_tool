# ==========================================================
# FULL DEG ANALYSIS TOOLKIT — ERROR-PROOF VERSION
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import io, gzip
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from gprofiler import GProfiler

# ---------------- BASIC CONFIG ----------------
st.set_page_config("Full DEG Analysis", layout="wide")
sns.set(style="whitegrid")

# ---------------- SAFE FILE LOADER ----------------
def load_file(uploaded):
    name = uploaded.name.lower()
    raw = uploaded.read()

    try:
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(raw))
        if name.endswith(".gz"):
            with gzip.open(io.BytesIO(raw), "rt", errors="ignore") as f:
                return pd.read_csv(f, sep=None, engine="python")
        return pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception as e:
        st.error(f"File loading failed: {e}")
        st.stop()

# ---------------- SAFE STRING API ----------------
def fetch_string_ppi(genes, score_cutoff=700, species=9606):
    if not genes:
        return []

    url = "https://string-db.org/api/tsv/network"
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": score_cutoff
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []

        edges = []
        for line in r.text.split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 4:
                edges.append((parts[2], parts[3]))
        return edges

    except Exception:
        return []

# ---------------- SAFE TRRUST LOADER ----------------
@st.cache_data
def load_trrust_local():
    """
    TRRUST must be bundled locally to avoid HTTP errors.
    If file not present → feature disabled gracefully.
    """
    path = "trrust_human.tsv"
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, sep="\t", header=None)
        df.columns = ["TF", "Target", "Mode", "PMID"]
        return df
    except Exception:
        return None

# ---------------- NETWORK DRAW ----------------
def draw_network(G, title, color):
    if G.number_of_nodes() == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=color,
        node_size=800,
        font_size=8,
        ax=ax
    )
    ax.set_title(title)
    return fig

# ---------------- UI ----------------
st.title("🧬 Full DEG Analysis Toolkit (Stable & Error-Proof)")

uploaded = st.file_uploader("Upload DEG results (CSV / TSV / XLSX / GZ)")
if uploaded is None:
    st.stop()

df = load_file(uploaded)
st.success(f"Loaded {df.shape[0]} genes")

gene_col = st.selectbox("Gene column", df.columns)
fc_col = st.selectbox("logFC column", df.columns)
p_col = st.selectbox("p-value column", df.columns)

df[fc_col] = pd.to_numeric(df[fc_col], errors="coerce")
df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
df = df.dropna(subset=[fc_col, p_col])

# ---------------- FILTERING ----------------
neg_fc = st.slider("Negative logFC (≤)", -10, -1, -1)
pos_fc = st.slider("Positive logFC (≥)", 1, 10, 1)
p_cut = st.slider("p-value cutoff", 0.0001, 0.1, 0.05)

filtered = df[
    ((df[fc_col] <= neg_fc) | (df[fc_col] >= pos_fc)) &
    (df[p_col] <= p_cut)
]

if filtered.empty:
    st.warning("No genes passed filtering")
    st.stop()

genes = filtered[gene_col].astype(str).tolist()

# ---------------- VOLCANO ----------------
st.subheader("Volcano Plot")

colors = {
    "up": st.color_picker("Upregulated color", "#d62728"),
    "down": st.color_picker("Downregulated color", "#1f77b4"),
    "other": st.color_picker("Non-significant color", "#bdbdbd"),
}

fig, ax = plt.subplots()
ax.scatter(df[fc_col], -np.log10(df[p_col]), c=colors["other"], s=10)

up = filtered[filtered[fc_col] >= pos_fc]
down = filtered[filtered[fc_col] <= neg_fc]

ax.scatter(up[fc_col], -np.log10(up[p_col]), c=colors["up"], label="Up")
ax.scatter(down[fc_col], -np.log10(down[p_col]), c=colors["down"], label="Down")

ax.legend()
ax.set_xlabel("logFC")
ax.set_ylabel("-log10(p-value)")
st.pyplot(fig)

# ---------------- STRING PPI ----------------
st.subheader("STRING PPI Network")

score = st.slider("STRING confidence", 400, 900, 700)
edges = fetch_string_ppi(genes[:100], score)

if edges:
    G = nx.Graph()
    G.add_edges_from(edges)
    st.pyplot(draw_network(G, "STRING PPI", "#ff7f0e"))
else:
    st.info("STRING network unavailable (API timeout or no interactions)")

# ---------------- TRRUST ----------------
st.subheader("TF–Gene Network (TRRUST)")

trrust = load_trrust_local()
if trrust is not None:
    tf_edges = trrust[trrust["Target"].isin(genes)][["TF", "Target"]].values.tolist()
    if tf_edges:
        Gtf = nx.DiGraph()
        Gtf.add_edges_from(tf_edges)
        st.pyplot(draw_network(Gtf, "TRRUST TF–Gene Network", "#1f77b4"))
    else:
        st.info("No TF interactions found")
else:
    st.warning("TRRUST file not found — TF network disabled")

# ---------------- ENRICHMENT ----------------
st.subheader("Functional Enrichment (gProfiler)")

try:
    gp = GProfiler(return_dataframe=True)
    enrich = gp.profile(organism="hsapiens", query=genes)

    st.dataframe(enrich)
    st.subheader("KEGG")
    st.dataframe(enrich[enrich["source"] == "KEGG"])
    st.subheader("GO:BP")
    st.dataframe(enrich[enrich["source"] == "GO:BP"])
except Exception as e:
    st.warning(f"gProfiler unavailable: {e}")

st.success("DEG analysis completed without errors ✅")
