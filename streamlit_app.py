# app.py
# Streamlit app — Gene Expression / DEG analysis prototype
# Single-file Streamlit application (publication-ready prototype)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip, io, os, zipfile, tempfile, traceback
from scipy import stats
import statsmodels.stats.multitest as smm
import networkx as nx
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.decomposition import PCA
from fpdf import FPDF

sns.set(style="whitegrid")
st.set_page_config(page_title="Transcriptomics DEG Tool", layout="wide")

# ---------- CONFIG ----------
# Demo file path (the developer requested this local path be available as demo)
demo_local_path = "/main/GSE269528_gene.description.xls.txt.gz"

# ---------- HELPERS ----------
def read_bytes_to_df(b: bytes, name_hint: str = None):
    """Robust loader from bytes for csv/tsv/xlsx/txt.gz (GEO supplementary)."""
    # Excel?
    if name_hint and name_hint.lower().endswith(('.xls', '.xlsx')):
        try:
            return pd.read_excel(io.BytesIO(b))
        except Exception:
            pass
    # gz compressed?
    if name_hint and name_hint.lower().endswith('.gz'):
        try:
            with gzip.open(io.BytesIO(b), 'rt', encoding='utf-8', errors='ignore') as fh:
                txt = fh.read()
            # check for series matrix block
            if "!series_matrix_table_begin" in txt and "!series_matrix_table_end" in txt:
                inner = txt.split("!series_matrix_table_begin")[-1].split("!series_matrix_table_end")[0]
                return pd.read_csv(io.StringIO(inner), sep="\t", engine='python')
            # Try tab first
            try:
                return pd.read_csv(io.StringIO(txt), sep="\t", engine='python')
            except Exception:
                # fallback: auto-detect
                return pd.read_csv(io.StringIO(txt), sep=None, engine='python')
        except Exception:
            pass
    # Try automatic sep detection (pandas)
    bio = io.BytesIO(b)
    try:
        return pd.read_csv(bio, sep=None, engine='python')
    except Exception:
        bio.seek(0)
    # Try tab
    try:
        return pd.read_csv(io.BytesIO(b), sep="\t", engine='python')
    except Exception:
        pass
    # Last resort: whitespace split into columns
    try:
        txt = b.decode('utf-8', errors='ignore')
        rows = [r.strip().split() for r in txt.splitlines() if r.strip()]
        return pd.DataFrame(rows)
    except Exception as e:
        raise ValueError("Unable to parse uploaded file: " + str(e))

def load_from_uploaded(uploaded_file):
    b = uploaded_file.read()
    return read_bytes_to_df(b, uploaded_file.name)

def load_from_local(path):
    with open(path, 'rb') as f:
        b = f.read()
    return read_bytes_to_df(b, path)

def is_deg_table(df):
    cols = [str(c).lower() for c in df.columns.astype(str)]
    has_gene = any(any(k in c for k in ["gene","symbol","ensembl","id"]) for c in cols)
    has_logfc = any("logfc" in c or "log2fc" in c or "fold" in c for c in cols)
    has_p = any("pval" in c or "p.value" in c or "p_value" in c or "p-value" in c or "padj" in c for c in cols)
    return has_gene and (has_logfc or has_p)

def autodetect_gene_logfc_pval(df):
    gene=None; logfc=None; p=None
    for c in df.columns:
        lc = str(c).lower()
        if gene is None and any(k in lc for k in ["gene","symbol","ensembl","id"]):
            gene = c
        if logfc is None and any(k in lc for k in ["logfc","log2fc","fold","foldchange"]):
            logfc = c
        if p is None and any(k in lc for k in ["pval","p.value","p_value","p-value","padj"]):
            p = c
    return gene, logfc, p

def compute_de_from_expr(expr_df, groupA_cols, groupB_cols):
    """expr_df: DataFrame indexed by gene, columns are samples (numeric)."""
    mat = expr_df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    logmat = np.log2(mat + 1.0)
    rows=[]
    for gene, row in logmat.iterrows():
        a = row[groupA_cols].dropna().astype(float)
        b = row[groupB_cols].dropna().astype(float)
        if len(a)<1 or len(b)<1:
            continue
        log2fc = a.mean() - b.mean()
        try:
            _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
            if np.isnan(p): p = 1.0
        except Exception:
            p = 1.0
        rows.append((gene, a.mean(), b.mean(), log2fc, p))
    de = pd.DataFrame(rows, columns=["Gene","meanA","meanB","log2FC","pvalue"]).set_index("Gene")
    de["padj"] = smm.multipletests(de["pvalue"].fillna(1.0), method='fdr_bh')[1]
    return de

def generate_volcano(de_df, out_png, lfc_plus, lfc_minus, lfc_cutoff, pval_cut):
    plt.figure(figsize=(6,5))
    x = de_df["log2FC"]
    y = -np.log10(de_df["pvalue"].replace(0,1e-300))
    plt.scatter(x, y, s=8, color='gray', alpha=0.6)
    sig = de_df[(de_df["padj"]<=pval_cut) & (de_df["log2FC"].abs()>=lfc_cutoff)]
    if not sig.empty:
        plt.scatter(sig["log2FC"], -np.log10(sig["pvalue"].replace(0,1e-300)), s=12, color='red')
    plt.axvline(lfc_plus, color='blue', linestyle='--')
    plt.axvline(lfc_minus, color='blue', linestyle='--')
    plt.axhline(-np.log10(pval_cut), color='black', linestyle=':')
    plt.xlabel("log2FC"); plt.ylabel("-log10(p-value)")
    plt.title("Volcano")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def generate_heatmap(expr_df, genes, out_png):
    sub = expr_df.loc[genes].apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if sub.shape[0]==0:
        raise ValueError("No rows available for heatmap.")
    sub_z = (sub.sub(sub.mean(axis=1), axis=0)).div(sub.std(axis=1).replace(0,1), axis=0)
    plt.figure(figsize=(8,6))
    sns.heatmap(sub_z, cmap="vlag", yticklabels=True)
    plt.title("Heatmap (z-score)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def generate_pca(expr_df, out_png, group_map=None):
    X = np.log2(expr_df + 1.0).T.fillna(0)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(6,5))
    colors = []
    for s in X.index:
        if group_map and s in group_map:
            colors.append('red' if group_map[s]=="HBV" else 'blue')
        else:
            colors.append('black')
    ax.scatter(pcs[:,0], pcs[:,1], c=colors)
    for i,s in enumerate(X.index):
        ax.text(pcs[i,0], pcs[i,1], s, fontsize=6)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("PCA (samples)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close()

def query_string_and_make_network(gene_list, out_png, species=10090):
    """Best-effort STRING query; returns hub genes file path or None."""
    if len(gene_list)==0:
        return None, None
    url = "https://string-db.org/api/json/network"
    ids = "%0d".join(gene_list[:150])
    r = requests.get(url, params={"identifiers": ids, "species": species}, timeout=60)
    if r.status_code != 200:
        return None, None
    data = r.json()
    if len(data)==0:
        return None, None
    G = nx.Graph()
    for it in data:
        a = it.get('preferredName_A'); b = it.get('preferredName_B'); s = it.get('score',0.0)
        if a and b:
            G.add_edge(a,b, weight=s)
    if G.number_of_nodes()==0:
        return None, None
    # draw largest component
    comp = max(nx.connected_components(G), key=len)
    Gsub = G.subgraph(comp).copy()
    pos = nx.spring_layout(Gsub, seed=42)
    plt.figure(figsize=(10,8))
    nx.draw_networkx(Gsub, pos, node_size=50, font_size=6)
    plt.title("STRING PPI (largest component)")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    # hub genes
    deg = nx.degree_centrality(G)
    hub_df = pd.Series(deg).sort_values(ascending=False).rename("degree").to_frame()
    return out_png, hub_df

def build_pdf(pdf_path, top100_df, volcano_png, heatmap_png, pca_png=None):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story=[]
    story.append(Paragraph("DEG Analysis — Prototype Report", styles['Title'])); story.append(Spacer(1,12))
    story.append(Paragraph("Top 100 genes (by padj then |log2FC|):", styles['Heading2'])); story.append(Spacer(1,8))
    # small table with first 6 columns
    cols = list(top100_df.columns[:6])
    tbl = [cols]
    for _, r in top100_df.head(100).iterrows():
        row = [str(r.get(c, "")) for c in cols]
        tbl.append(row)
    story.append(Table(tbl)); story.append(Spacer(1,12))
    if volcano_png and os.path.exists(volcano_png):
        story.append(Paragraph("Volcano", styles['Heading3'])); story.append(Image(volcano_png, width=450, height=300)); story.append(Spacer(1,12))
    if heatmap_png and os.path.exists(heatmap_png):
        story.append(Paragraph("Heatmap", styles['Heading3'])); story.append(Image(heatmap_png, width=450, height=300)); story.append(Spacer(1,12))
    if pca_png and os.path.exists(pca_png):
        story.append(Paragraph("PCA", styles['Heading3'])); story.append(Image(pca_png, width=450, height=300)); story.append(Spacer(1,12))
    doc.build(story)

# ---------- STREAMLIT UI ----------
st.title("Streamlined DEG Analysis — Prototype")
st.markdown("Upload expression matrix (genes × samples) or DEG table. Supports `.csv`, `.tsv`, `.txt`, `.gz`, `.xls(x)`.")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload file or use demo", type=['csv','tsv','txt','gz','xls','xlsx'])
    use_demo = st.button("Use demo file")
    df = None
    if use_demo:
        st.info(f"Loading demo from local path: {demo_local_path}")
        try:
            df = load_from_local(demo_local_path)
        except Exception as e:
            st.error("Failed to load demo: " + str(e))
            st.text(traceback.format_exc())
    elif uploaded:
        try:
            df = load_from_uploaded(uploaded)
        except Exception as e:
            st.error("Failed to read uploaded file: " + str(e))
            st.text(traceback.format_exc())
    if df is not None:
        st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        if st.checkbox("Show preview"):
            st.dataframe(df.head(200))

with col2:
    st.header("Filters & options")
    lfc_plus = st.number_input("log2FC + threshold (>=)", value=1.0, step=0.1)
    lfc_minus = st.number_input("log2FC - threshold (<=)", value=-1.0, step=0.1)
    lfc_cutoff = st.number_input("Absolute log2FC cutoff (|log2FC| >=)", value=1.0, step=0.1)
    pval_cut = st.number_input("Adjusted p-value cutoff", value=0.05, format="%.4f")
    do_string = st.checkbox("Query STRING (PPI & hub genes)", value=False)
    run = st.button("Run analysis")

# Run
if run:
    if df is None:
        st.error("No input file. Upload or use demo.")
        st.stop()

    tmpdir = tempfile.mkdtemp(prefix="degtool_")
    outdir = os.path.join(tmpdir, "outputs"); os.makedirs(outdir, exist_ok=True)

    try:
        # Detect structure
        if is_deg_table(df):
            st.info("DEG-like table detected.")
            gene_col, logfc_col, p_col = autodetect_gene_logfc_pval(df)
            if gene_col is None:
                st.error("Could not detect gene column in DEG table.")
                st.stop()
            work = df[[gene_col, logfc_col, p_col]].dropna()
            work.columns = ["Gene","log2FC","pvalue"]
            # ensure numeric
            work["log2FC"] = pd.to_numeric(work["log2FC"], errors='coerce')
            work["pvalue"] = pd.to_numeric(work["pvalue"], errors='coerce')
            work["padj"] = smm.multipletests(work["pvalue"].fillna(1.0), method='fdr_bh')[1]
            de_df = work.set_index("Gene")
            expr_numeric = None
        else:
            st.info("Assuming expression matrix (genes x samples). Detecting gene column & sample columns...")
            # detect gene id column: first column containing non-numeric values
            gene_col = None
            for c in df.columns:
                if df[c].dtype == object:
                    gene_col = c; break
            if gene_col is None:
                gene_col = df.columns[0]
            # convert numeric columns
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                # attempt conversion of other columns
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                st.error("No numeric expression columns detected.")
                st.stop()
            # build expr_numeric with gene index
            if gene_col in df.columns:
                expr_numeric = df.set_index(gene_col)[numeric.columns]
            else:
                expr_numeric = numeric
            st.write("Expression matrix shape:", expr_numeric.shape)
            # auto assign groups
            sample_names = list(expr_numeric.columns)
            group_map={}
            for c in sample_names:
                lc = c.lower()
                if "hbv" in lc and "den" in lc:
                    group_map[c] = "HBV"
                elif "sham" in lc and "den" in lc:
                    group_map[c] = "Sham"
                elif "hbv" in lc:
                    group_map[c] = "HBV"
                elif "sham" in lc or "ctrl" in lc or "control" in lc:
                    group_map[c] = "Sham"
                else:
                    group_map[c] = None
            if all(v is None for v in group_map.values()):
                half = len(sample_names)//2
                for i,c in enumerate(sample_names):
                    group_map[c] = "HBV" if i<half else "Sham"
            hbv_cols = [c for c,v in group_map.items() if v=="HBV"]
            sham_cols = [c for c,v in group_map.items() if v=="Sham"]
            if len(hbv_cols) < 1 or len(sham_cols) < 1:
                st.error("Not enough samples assigned to groups automatically. Edit sample names or supply DEG table.")
                st.stop()
            st.write("Group counts:", {"HBV": len(hbv_cols), "Sham": len(sham_cols)})
            de_df = compute_de_from_expr(expr_numeric, hbv_cols, sham_cols)

        # Save full DE table
        de_csv = os.path.join(outdir, "DE_full_table.csv"); de_df.to_csv(de_csv)

        # Select by thresholds
        de_df = de_df.sort_values(["padj","log2FC"])
        selected = de_df[(de_df["padj"]<=pval_cut) & (de_df["log2FC"].abs()>=lfc_cutoff)]
        selected.to_csv(os.path.join(outdir, f"DE_selected_absFC_{lfc_cutoff}_padj{pval_cut}.csv"))

        top100 = de_df.head(100)
        top100.to_csv(os.path.join(outdir, "DE_top100.csv"))

        # Plots
        volcano_png = os.path.join(outdir, "volcano.png")
        generate_volcano(de_df, volcano_png, lfc_plus, lfc_minus, lfc_cutoff, pval_cut)

        heatmap_png = os.path.join(outdir, "heatmap_top100.png")
        if 'expr_numeric' in locals() and expr_numeric is not None:
            genes_for_heat = list(top100.head(50).index)
            try:
                generate_heatmap(expr_numeric, genes_for_heat, heatmap_png)
            except Exception as e:
                st.warning("Heatmap error: " + str(e))
                heatmap_png = None
        else:
            heatmap_png = None

        # PCA
        pca_png = os.path.join(outdir, "pca.png")
        try:
            if 'expr_numeric' in locals() and expr_numeric is not None:
                generate_pca(expr_numeric, pca_png, group_map)
            else:
                pca_png = None
        except Exception as e:
            st.warning("PCA failed: " + str(e))
            pca_png = None

        # STRING PPI & hub genes
        string_png = os.path.join(outdir, "string_network.png")
        hub_csv = os.path.join(outdir, "hub_genes.csv")
        if do_string:
            try:
                spng, hub_df = query_string_and_make_network(list(top100.head(200).index.astype(str)), string_png, species=10090)
                if hub_df is not None:
                    hub_df.to_csv(hub_csv)
            except Exception as e:
                st.warning("STRING failed: " + str(e))

        # Build PDF (top100 only)
        pdf_path = os.path.join(outdir, "DE_report_top100.pdf")
        build_pdf(pdf_path, top100.reset_index(), volcano_png, heatmap_png, pca_png)

        # ZIP everything
        zip_path = os.path.join(tmpdir, "DE_results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root,_,files in os.walk(outdir):
                for f in files:
                    z.write(os.path.join(root,f), arcname=os.path.join("outputs", f))

        # Offer downloads
        with open(zip_path, "rb") as f:
            st.download_button("Download all results (ZIP)", data=f, file_name="DE_results.zip", mime="application/zip")
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF (top 100)", data=f, file_name="DE_report_top100.pdf", mime="application/pdf")

        # Previews
        if os.path.exists(volcano_png):
            st.image(volcano_png, caption="Volcano", use_column_width=True)
        if heatmap_png and os.path.exists(heatmap_png):
            st.image(heatmap_png, caption="Heatmap (top genes)", use_column_width=True)
        if pca_png and os.path.exists(pca_png):
            st.image(pca_png, caption="PCA samples", use_column_width=True)
        if do_string and os.path.exists(string_png):
            st.image(string_png, caption="STRING PPI (largest comp)", use_column_width=True)
        st.success("Analysis completed. Download the ZIP / PDF above.")

    except Exception as e:
        st.error("Analysis failed: " + str(e))
        st.text(traceback.format_exc())
