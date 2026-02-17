# =========================================================
# REAL BIOMATH LAYER
# Deterministic Systems Biology Metrics
# =========================================================

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy


# =========================================================
# GENE IMPORTANCE USING STATISTICAL SIGNIFICANCE
# =========================================================

def gene_importance_score(deg_df, logfc_col, pval_col):

    df = deg_df.copy()

    df["abs_logFC"] = np.abs(df[logfc_col])

    df["importance_score"] = (
        df["abs_logFC"] *
        (-np.log10(df[pval_col] + 1e-300))
    )

    return df


# =========================================================
# SYSTEM STABILITY (BASED ON VARIANCE)
# =========================================================

def system_stability_score(deg_df, logfc_col):

    variance = np.var(deg_df[logfc_col])

    stability = 1 / (1 + variance)

    deg_df["system_stability"] = stability

    return deg_df, stability


# =========================================================
# PROBABILITY DISTRIBUTION
# =========================================================

def gene_probability_distribution(deg_df, logfc_col):

    abs_vals = np.abs(deg_df[logfc_col])

    prob = abs_vals / abs_vals.sum()

    deg_df["gene_probability"] = prob

    return deg_df, prob


# =========================================================
# NETWORK INFLUENCE
# =========================================================

def network_influence_score(ppi_edges, deg_df, gene_col):

    if ppi_edges is None or ppi_edges.empty:

        deg_df["network_centrality"] = 0
        return deg_df, 0

    G = nx.from_pandas_edgelist(
        ppi_edges,
        "preferredName_A",
        "preferredName_B"
    )

    centrality = nx.degree_centrality(G)

    deg_df["network_centrality"] = (
        deg_df[gene_col].map(centrality).fillna(0)
    )

    avg_centrality = np.mean(list(centrality.values()))

    return deg_df, avg_centrality


# =========================================================
# SYSTEM ENTROPY
# =========================================================

def system_entropy(prob_distribution):

    return entropy(prob_distribution, base=2)


# =========================================================
# SYSTEM PERTURBATION MAGNITUDE
# =========================================================

def perturbation_magnitude(deg_df, logfc_col):

    return np.mean(np.abs(deg_df[logfc_col]))


# =========================================================
# MAIN BIOMATH EXECUTION
# =========================================================

def run_biomath_layer(
        deg_df,
        gene_col,
        logfc_col,
        pval_col,
        ppi_edges=None
):

    df = deg_df.copy()

    df = gene_importance_score(df, logfc_col, pval_col)

    df, stability = system_stability_score(df, logfc_col)

    df, prob = gene_probability_distribution(df, logfc_col)

    entropy_score = system_entropy(prob)

    df, centrality = network_influence_score(
        ppi_edges,
        df,
        gene_col
    )

    perturb = perturbation_magnitude(df, logfc_col)

    biomath_metrics = {

        "system_entropy": entropy_score,

        "system_stability": stability,

        "network_centrality": centrality,

        "perturbation_magnitude": perturb

    }

    return df, biomath_metrics
    
