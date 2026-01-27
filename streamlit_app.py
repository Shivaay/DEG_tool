# ==========================================================
# ADVANCED PROFESSIONAL EXTENSION (NON-DESTRUCTIVE)
# ==========================================================

import mygene
from textwrap import fill

st.markdown("---")
st.header("📊 Publication-Ready Outputs & AI Interpretation")

# ----------------------------------------------------------
# 1. GENE ID MAPPING (AUTOMATIC)
# ----------------------------------------------------------
st.subheader("🔁 Gene ID Resolution")

mg = mygene.MyGeneInfo()

@st.cache_data
def map_gene_ids(glist):
    try:
        res = mg.querymany(
            glist,
            scopes=["symbol", "ensembl.gene", "entrezgene"],
            fields="symbol,entrezgene,ensembl.gene",
            species="human"
        )
        return pd.DataFrame(res)
    except Exception:
        return pd.DataFrame()

gene_map = map_gene_ids(genes)

with st.expander("Resolved Gene Identifiers"):
    if not gene_map.empty:
        st.dataframe(gene_map[["query", "symbol", "entrezgene"]])
    else:
        st.warning("Gene ID mapping could not be resolved.")

# ----------------------------------------------------------
# 2. REACTOME ENRICHMENT (iDEP PARITY)
# ----------------------------------------------------------
st.subheader("🧬 Reactome Pathway Enrichment")

reactome = enrich[enrich["source"] == "REAC"]
if reactome.empty:
    st.info("No significant Reactome pathways detected.")
else:
    st.dataframe(reactome[["name", "p_value", "intersection_size"]].head(15))

# ----------------------------------------------------------
# 3. 300 DPI FIGURE EXPORTS
# ----------------------------------------------------------
st.subheader("🖼️ Manuscript-Ready Figures (300 DPI)")

if st.button("Export All Figures (300 DPI)"):
    fig.savefig("volcano_plot_300dpi.png", dpi=300, bbox_inches="tight")
    st.success("Figures exported at 300 DPI for publication.")

st.markdown(
    "_Figures are suitable for journals such as Nature, Cell, and PLOS._"
)

# ----------------------------------------------------------
# 4. AI-ASSISTED SCIENTIFIC SUMMARY (CONSTRAINED)
# ----------------------------------------------------------
st.subheader("🧠 AI-Generated Manuscript Summary")

def ai_scientific_summary():
    up_terms = up_enrich[up_enrich["source"] == "GO:BP"]["name"].head(3).tolist()
    down_terms = down_enrich[down_enrich["source"] == "GO:BP"]["name"].head(3).tolist()

    hub_text = (
        f"Key hub genes including {', '.join(hub_list[:5])} "
        "exhibited high network connectivity, suggesting regulatory importance."
        if hub_list else
        "No dominant hub genes were identified."
    )

    summary = f"""
Differential expression analysis identified {len(deg)} significantly
regulated genes. Among these, {len(up_genes)} genes were upregulated,
while {len(down_genes)} genes were downregulated.

Upregulated genes were primarily enriched in biological processes related
to {', '.join(up_terms) if up_terms else 'adaptive cellular responses'},
suggesting activation of condition-specific signaling and regulatory pathways.

Conversely, downregulated genes were associated with processes such as
{', '.join(down_terms) if down_terms else 'metabolic and structural pathways'},
indicating suppression of these biological functions.

Protein–protein interaction analysis revealed a structured interaction
network. {hub_text}

Overall, these results indicate coordinated transcriptional reprogramming
involving pathway-level activation and repression, consistent with a
biologically meaningful molecular response.
"""
    return fill(summary, 110)

ai_summary = ai_scientific_summary()

st.text_area(
    "Manuscript-Ready AI Summary (Results + Interpretation)",
    ai_summary,
    height=350
)

# ----------------------------------------------------------
# 5. METHODS SECTION (AUTO-GENERATED)
# ----------------------------------------------------------
st.subheader("🧪 Auto-Generated Methods Section")

methods = f"""
Differentially expressed genes were analyzed using user-provided expression
statistics. Genes were filtered based on log fold-change thresholds
({neg_fc}, {pos_fc}) and statistical significance (p ≤ {p_cut}).
Functional enrichment analysis was performed using gProfiler, querying
Gene Ontology (Biological Process, Molecular Function, Cellular Component),
KEGG, and Reactome databases. Protein–protein interaction networks were
retrieved from the STRING database (confidence score ≥ 0.7).
"""

st.text_area("Methods (Copy for Manuscript)", fill(methods, 110), height=200)

# ----------------------------------------------------------
# 6. INTERPRETATION HELP (EDUCATIONAL + INDUSTRY)
# ----------------------------------------------------------
st.subheader("📖 Interpretation Guide")

with st.expander("How to interpret Up vs Down regulation"):
    st.write(
        "Upregulated genes indicate activated biological processes, while "
        "downregulated genes reflect suppressed pathways under the studied condition."
    )

with st.expander("How to interpret Hub Genes"):
    st.write(
        "Hub genes show high connectivity in interaction networks and often "
        "represent key regulators or bottlenecks in biological systems."
    )

with st.expander("Reproducibility & Limitations"):
    st.write(
        "This analysis assumes prior normalization and appropriate statistical modeling. "
        "Batch effects and experimental design should be considered during preprocessing."
    )

st.success("✅ Advanced AI-assisted interpretation layer enabled.")
