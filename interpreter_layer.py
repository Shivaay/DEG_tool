"""
Systems Biology Interpretation Layer
------------------------------------
Uses Traditional + Biomath outputs
Generates interpretation and visualization
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# -------------------------
# Hub Gene Identification
# -------------------------
def hub_gene_analysis(ppi_edges):

    G = nx.from_pandas_edgelist(ppi_edges, "source", "target")

    degree = nx.degree_centrality(G)
    between = nx.betweenness_centrality(G)
    eigen = nx.eigenvector_centrality(G, max_iter=500)

    hub_df = pd.DataFrame({
        "gene": degree.keys(),
        "degree": degree.values(),
        "betweenness": between.values(),
        "eigenvector": eigen.values()
    })

    hub_df["hub_score"] = hub_df.iloc[:,1:].mean(axis=1)

    return hub_df, G


# -------------------------
# Pathway Posterior Scoring
# -------------------------
def pathway_posterior_scoring(enrichment_df):

    enrichment_df["posterior_pathway_score"] = (
        -np.log10(enrichment_df["p_value"]) *
        enrichment_df["gene_ratio"]
    )

    return enrichment_df


# -------------------------
# Visualization
# -------------------------
def biomath_volcano_plot(df):

    plt.figure(figsize=(8,6))
    plt.scatter(df["log2FC"], -np.log10(df["adj_pvalue"]),
                c=df["biomath_score"], cmap="viridis")
    plt.xlabel("log2 Fold Change")
    plt.ylabel("-log10 Adjusted P")
    plt.title("BioMathematical Volcano Plot")
    plt.colorbar(label="Biomath Score")
    plt.show()


# -------------------------
# Manuscript Generator
# -------------------------
def generate_manuscript(deg_df, hub_df):

    up_genes = deg_df.sort_values("log2FC", ascending=False)["gene"].head(10)
    down_genes = deg_df.sort_values("log2FC")["gene"].head(10)
    hubs = hub_df.sort_values("hub_score", ascending=False)["gene"].head(10)

    manuscript = f"""

TITLE:
BioMathematical DEG and Systems Analysis Reveals Critical Regulatory Genes

ABSTRACT:
This study integrates differential expression with advanced biomathematical modelling.

RESULTS:
Top Upregulated Genes: {', '.join(up_genes)}
Top Downregulated Genes: {', '.join(down_genes)}
Hub Genes: {', '.join(hubs)}

DISCUSSION:
Integration of probabilistic and network-based modelling improved biological interpretation.

CONCLUSION:
This framework enhances transcriptomic discovery and clinical translation.
"""

    return manuscript


# -------------------------
# MASTER INTERPRETER
# -------------------------
def run_interpreter_layer(deg_df, ppi_edges, enrichment_df=None):

    hub_df, G = hub_gene_analysis(ppi_edges)

    if enrichment_df is not None:
        enrichment_df = pathway_posterior_scoring(enrichment_df)

    biomath_volcano_plot(deg_df)

    manuscript = generate_manuscript(deg_df, hub_df)

    return {
        "hub_genes": hub_df,
        "manuscript": manuscript,
        "enrichment": enrichment_df
    }
