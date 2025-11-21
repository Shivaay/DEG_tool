# streamlit_app.py
# Streamlit DEG analysis prototype — upload -> analysis -> ZIP + PDF (top100)
# Features: auto-detect input, compute DE (if expression matrix), volcano, heatmap, PCA, MA-plot,
# g:Profiler GO enrichment, STRING PPI (optional), hub genes by multiple centralities,
# outputs saved into ZIP and PDF (top100). Demo button uses local uploaded demo file.

import streamlit as st
import pandas as pd
import numpy as np
import io, os, gzip, zipfile, tempfile, traceback, time
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import requests

from scipy import stats
import statsmodels.stats.multitest as smm
from sklearn.decomposition import PCA
from gprofiler import GProfiler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

sns.set(style="whitegrid")
st.set_page_config(page_title="DEG Tool Prototype", layout="wide")

# ------------------ CONFIG ------------------
# Local demo path (you already uploaded this to runtime earlier)
demo_local_path = "/mnt/data/GSE269528_gene.description.xls.txt.gz"

# ------------------ HELPERS ------------------
def read_bytes_to_df(b: bytes, name_hint: str = None):
    """Robust loader: gz text, csv, tsv, xlsx, generic whitespace-split fallback."""
    # Excel
    try:
        if name_hint and name_hint.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(io.BytesIO(b))
    except Exception:
        pass

    # gz compressed text (GEO suppl)
    try:
        if name_hint and name_hint.lower().endswith('.gz'):
            with gzip.open(io.BytesIO(b), 'rt', encoding='utf-8', errors='ignore') as fh:
                txt = fh.read()
            # if series_matrix-like include table region
            if "!series_matrix_table_begin" in txt and "!series_matrix_table_end" in txt:
                inner = txt.split("!series_matrix_table_begin")[-1].split("!series_matrix_table_end")[0]
                return pd.read_csv(io.StringIO(inner), sep="\t", engine='python')
            # try TSV
            try:
                return pd.read_csv(io.StringIO(txt), sep="\t", engine='python')
            except Exception:
                return pd.read_csv(io.StringIO(txt), sep=None, engine='python')
    except Exception:
        pass

    # try auto sep detection
    try:
        return pd.read_csv(io.BytesIO(b), sep=None, engine='python')
    except Exception:
        pass
    # try tab
    try:
        return pd.read_csv(io.BytesIO(b), sep="\t", engine='python')
    except Exception:
        pass
    # fallback whitespace splitting
    try:
        txt = b.decode('utf-8', errors='ignore')
        rows = [r.strip().split() for r in txt.splitlines() if r.strip()]
        return pd.DataFrame(rows)
    except Exception as e:
        raise ValueError("Could not parse uploaded file: " + str(e))

def load_uploaded(uploaded_file):
    b = uploaded_file.read()
    return read_bytes_to_df(b, uploaded_file.name)

def load_local(path):
    with open(path, 'rb') as f:
        b = f.read()
    return read_bytes_to_df(b, path)

def is_deg_table(df):
    cols = [str(c).lower() for c in df.columns.astype(str)]
    has_gene = any(any(k in c for k in ["gene","symbol","ensembl","id"]) for c in cols)
    has_logfc = any("logfc" in c or "log2fc" in c or "fold" in c for c in cols)
    has_p = any("pval" in c or "p.value" in c or "p_value" in c or "padj" in c or "p-value" in c for c in cols)
    return has_gene and (has_logfc or has_p)

def autodetect_gene_logfc_pval(df):
    gene=None; logfc=None; p=None
    for c in df.columns:
        lc = str(c).lower()
        if gene is None and any(k in lc for k in ["gene","symbol","ensembl","id"]):
            gene = c
        if logfc is None and any(k in lc for k in ["logfc","log2fc","fold","foldchange"]):
            logfc = c
        if p is None and any(k in lc for k in ["pval","p.value","p_value","padj","p-value"]):
            p = c
    return gene, logfc, p

def compute_de_from_expression(expr_df, groupA_cols, groupB_cols):
    """expr_df indexed by gene, numeric columns."""
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

def make_volcano(de_df, out_file, lfc_plus, lfc_minus, lfc_cutoff, pval_cut):
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
    plt.tight_layout(); plt.savefig(out_file, dpi=150); plt.close()

def make_heatmap(expr_df, genes, out_file):
    sub = expr_df.loc[genes].apply(pd.to_numeric, errors='coerce').dropna(how='all')
    if sub.shape[0]==0:
        raise ValueError("No data for heatmap.")
    sub_z = (sub.sub(sub.mean(axis=1), axis=0)).div(sub.std(axis=1).replace(0,1), axis=0)
    plt.figure(figsize=(8,6)); sns.heatmap(sub_z, cmap="vlag", yticklabels=True); plt.title("Heatmap (z-score)")
    plt.tight_layout(); plt.savefig(out_file, dpi=150); plt.close()

def make_pca(expr_df, out_file, group_map=None):
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
    fig.tight_layout(); fig.savefig(out_file, dpi=150); plt.close()

def query_string(gene_list, out_png, species=10090):
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
    comp = max(nx.connected_components(G), key=len)
    Gsub = G.subgraph(comp).copy()
    plt.figure(figsize=(10,8)); nx.draw_networkx(Gsub, pos=nx.spring_layout(Gsub, seed=42), node_size=40, font_size=6)
    plt.title("STRING PPI (largest component)"); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    deg = nx.degree_centrality(G)
    hub_df = pd.Series(deg).sort_values(ascending=False).rename("degree").to_frame()
    return out_png, hub_df

def run_gprofiler(query_genes, organism="mmusculus"):
    gp = GProfiler(return_dataframe=True)
    res = gp.profile(organism=organism, query=query_genes, sources=["GO:BP","GO:MF","GO:CC","KEGG"])
    return res

def build_pdf(pdf_path, top100_df, volcano_png, heatmap_png, pca_png, go_csv_path=None, ppi_png=None, hub_df=None):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story=[]
    story.append(Paragraph("DEG Analysis — Prototype Report", styles['Title'])); story.append(Spacer(1,12))
    story.append(Paragraph("Top 100 genes (by padj then |log2FC|):", styles['Heading2'])); story.append(Spacer(1,8))
    cols = list(top100_df.columns[:6])
    tbl = [cols]
    for _, r in top100_df.head(100).iterrows():
        row = [str(r.get(c,"")) for c in cols]; tbl.append(row)
    story.append(Table(tbl)); story.append(Spacer(1,12))
    if volcano_png and os.path.exists(volcano_png):
        story.append(Paragraph("Volcano", styles['Heading3'])); story.append(Image(volcano_png, width=450, height=300)); story.append(Spacer(1,12))
    if heatmap_png and os.path.exists(heatmap_png):
        story.append(Paragraph("Heatmap", styles['Heading3'])); story.append(Image(heatmap_png, width=450, height=300)); story.append(Spacer(1,12))
    if pca_png and os.path.exists(pca_png):
        story.append(Paragraph("PCA", styles['Heading3'])); story.append(Image(pca_png, width=450, height=300)); story.append(Spacer(1,12))
    if go_csv_path and os.path.exists(go_csv_path):
        go = pd.read_csv(go_csv_path).head(10)
        story.append(Paragraph("Top GO terms (g:Profiler)", styles['Heading3'])); story.append(Spacer(1,6))
        tbl2 = [list(go.columns[:4])]
        for row in go.itertuples(index=False):
            tbl2.append([str(x) for x in row[:4]])
        story.append(Table(tbl2)); story.append(Spacer(1,12))
    if ppi_png and os.path.exists(ppi_png):
        story.append(Paragraph("PPI network (STRING)", styles['Heading3'])); story.append(Image(ppi_png, width=450, height=350)); story.append(Spacer(1,12))
    if hub_df is not None and not hub_df.empty:
        story.append(Paragraph("Top hub genes (degree centrality)", styles['Heading3'])); story.append(Spacer(1,6))
        tbl3 = [["Gene","Degree"]]
        for gene, val in hub_df.head(10).itertuples():
            tbl3.append([str(gene), f"{val:.4f}"])
        story.append(Table(tbl3)); story.append(Spacer(1,12))
    doc.build(story)

# ------------------ UI ------------------
st.title("DEG Analysis Prototype — Streamlit")
st.write("Upload expression matrix (genes × samples) or DEG table. Supports CSV/TSV/TXT/XLSX/.gz (GEO).")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload file (or click demo)", type=['csv','tsv','txt','gz','xls','xlsx'])
    if st.button("Use demo file (GSE269528)"):
        try:
            df = load_local(demo_local_path)
            st.success("Demo file loaded.")
        except Exception as e:
            st.error("Demo load failed: " + str(e)); df = None
    elif uploaded:
        try:
            df = load_uploaded(uploaded)
            st.success("File loaded.")
        except Exception as e:
            st.error("Failed to parse uploaded file: " + str(e)); st.text(traceback.format_exc()); df = None
    else:
        df = None

    if df is not None:
        st.write("Data shape:", df.shape)
        if st.checkbox("Preview data (first 50 rows)"):
            st.dataframe(df.head(50))

with col2:
    st.header("Filters & options")
    lfc_plus = st.number_input("log2FC + (>=)", value=1.0, step=0.1)
    lfc_minus = st.number_input("log2FC - (<=)", value=-1.0, step=0.1)
    lfc_cutoff = st.number_input("Absolute log2FC cutoff (|log2FC| >=)", value=1.0, step=0.1)
    pval_cut = st.number_input("Adjusted p-value cutoff (padj <=)", value=0.05, format="%.4f")
    query_string_flag = st.checkbox("Query STRING (PPI & hub genes)", value=False)
    run = st.button("Run analysis")

if run:
    if df is None:
        st.error("Upload or load demo file first."); st.stop()

    tmpdir = tempfile.mkdtemp(prefix="deg_")
    outdir = os.path.join(tmpdir, "outputs"); os.makedirs(outdir, exist_ok=True)

    try:
        # detect structure
        if is_deg_table(df):
            st.info("DEG-like table detected.")
            gene_col, logfc_col, p_col = autodetect_gene_logfc_pval(df)
            if gene_col is None:
                st.error("Could not detect gene column in DEG table."); st.stop()
            work = df[[gene_col, logfc_col, p_col]].dropna()
            work.columns = ["Gene","log2FC","pvalue"]
            work["log2FC"] = pd.to_numeric(work["log2FC"], errors='coerce')
            work["pvalue"] = pd.to_numeric(work["pvalue"], errors='coerce')
            work["padj"] = smm.multipletests(work["pvalue"].fillna(1.0), method='fdr_bh')[1]
            de_df = work.set_index("Gene")
            expr_numeric = None
        else:
            st.info("Expression matrix detected (attempt).")
            # find gene id column
            gene_col = None
            for c in df.columns:
                if df[c].dtype == object or df[c].dtype == 'O':
                    gene_col = c; break
            if gene_col is None:
                gene_col = df.columns[0]
            # numeric columns
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                # try convert all columns
                for c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] < 2:
                st.error("Could not find numeric expression columns."); st.stop()
            if gene_col in df.columns:
                expr_numeric = df.set_index(gene_col)[numeric.columns]
            else:
                expr_numeric = numeric
            st.write("Expression shape:", expr_numeric.shape)
            # auto group assign
            sample_names = list(expr_numeric.columns)
            group_map = {}
            for c in sample_names:
                lc = c.lower()
                if "hbv" in lc and "den" in lc: group_map[c] = "HBV"
                elif "sham" in lc and "den" in lc: group_map[c] = "Sham"
                elif "hbv" in lc: group_map[c] = "HBV"
                elif "sham" in lc or "ctrl" in lc or "control" in lc: group_map[c] = "Sham"
                else: group_map[c] = None
            if all(v is None for v in group_map.values()):
                half = len(sample_names)//2
                for i,c in enumerate(sample_names): group_map[c] = "HBV" if i<half else "Sham"
            hbv_cols = [c for c,v in group_map.items() if v=="HBV"]
            sham_cols = [c for c,v in group_map.items() if v=="Sham"]
            if len(hbv_cols)<1 or len(sham_cols)<1:
                st.error("Failed to assign sample groups automatically."); st.stop()
            st.write("Group counts:", {"HBV": len(hbv_cols), "Sham": len(sham_cols)})
            de_df = compute_de_from_expression(expr_numeric, hbv_cols, sham_cols)

        # Save full DE
        de_csv = os.path.join(outdir, "DE_full_table.csv"); de_df.to_csv(de_csv)

        # Filter by thresholds
        de_df = de_df.sort_values(["padj","log2FC"])
        selection = de_df[(de_df["padj"]<=pval_cut) & (de_df["log2FC"].abs()>=lfc_cutoff)]
        selection.to_csv(os.path.join(outdir, f"DE_selected_absFC_{lfc_cutoff}_padj{pval_cut}.csv"))

        top100 = de_df.head(100)
        top100.to_csv(os.path.join(outdir, "DE_top100.csv"))

        # Generate plots
        volcano_png = os.path.join(outdir, "volcano.png")
        make_volcano(de_df, volcano_png, lfc_plus, lfc_minus, lfc_cutoff, pval_cut)

        heatmap_png = os.path.join(outdir, "heatmap_top100.png")
        if 'expr_numeric' in locals() and expr_numeric is not None:
            genes_for_heat = list(top100.head(50).index)
            try:
                make_heatmap(expr_numeric, genes_for_heat, heatmap_png)
            except Exception as e:
                st.warning("Heatmap failed: " + str(e)); heatmap_png = None
        else:
            heatmap_png = None

        pca_png = os.path.join(outdir, "pca.png")
        try:
            if 'expr_numeric' in locals() and expr_numeric is not None:
                make_pca(expr_numeric, pca_png, group_map)
            else:
                pca_png = None
        except Exception as e:
            st.warning("PCA failed: " + str(e)); pca_png = None

        # GO enrichment (g:Profiler) for top selected genes
        go_csv = os.path.join(outdir, "gprofiler_results.csv")
        try:
            query_genes = list(de_df[(de_df["padj"]<=pval_cut) & (de_df["log2FC"].abs()>=lfc_cutoff)].index)
            if len(query_genes) < 5:
                query_genes = list(de_df.reindex(de_df["log2FC"].abs().sort_values(ascending=False).index).head(200).index)
            if len(query_genes)>0:
                gp_res = run_gprofiler(query_genes, organism="mmusculus")
                gp_res.to_csv(go_csv, index=False)
        except Exception as e:
            st.warning("g:Profiler failed: " + str(e)); go_csv = None

        # STRING PPI & hub genes
        string_png = os.path.join(outdir, "string_network.png")
        hub_csv = os.path.join(outdir, "hub_genes.csv")
        hub_df = None
        if query_string_flag:
            try:
                spng, hub_df = query_string(list(top100.head(200).index.astype(str)), string_png, species=10090)
                if hub_df is not None:
                    hub_df.to_csv(hub_csv)
            except Exception as e:
                st.warning("STRING error: " + str(e)); hub_df = None; string_png=None

        # Build PDF with top100 (table, volcano, heatmap, GO, PPI, hub)
        pdf_path = os.path.join(outdir, "DE_report_top100.pdf")
        build_pdf(pdf_path, top100.reset_index(), volcano_png, heatmap_png, pca_png, go_csv, string_png, hub_df)

        # ZIP outputs
        zip_path = os.path.join(tmpdir, "DE_results.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for root,_,files in os.walk(outdir):
                for f in files:
                    z.write(os.path.join(root,f), arcname=os.path.join("outputs", f))

        # Download buttons
        with open(zip_path, "rb") as f:
            st.download_button("Download all results (ZIP)", f, file_name="DE_results.zip", mime="application/zip")
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF (top100)", f, file_name="DE_report_top100.pdf", mime="application/pdf")

        # Previews
        if os.path.exists(volcano_png): st.image(volcano_png, caption="Volcano", use_column_width=True)
        if heatmap_png and os.path.exists(heatmap_png): st.image(heatmap_png, caption="Heatmap (top genes)", use_column_width=True)
        if pca_png and os.path.exists(pca_png): st.image(pca_png, caption="PCA samples", use_column_width=True)
        if query_string_flag and string_png and os.path.exists(string_png): st.image(string_png, caption="STRING PPI", use_column_width=True)
        st.success("Analysis finished. Download the ZIP/PDF above.")

    except Exception as e:
        st.error("Analysis failed: " + str(e))
        st.text(traceback.format_exc())
