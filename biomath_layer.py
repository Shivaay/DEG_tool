"""
BioMathematical Advanced DEG Layer
----------------------------------
Performs advanced mathematical modelling on DEG data
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm


# -------------------------
# Utility
# -------------------------
def normalize(series):
    scaler = MinMaxScaler()
    return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()


# -------------------------
# Genetic Algorithm Fitness
# -------------------------
def genetic_algorithm_score(df):

    df["expression_variance"] = df["baseMean"].apply(lambda x: np.log1p(x))

    df["GA_score"] = (
        normalize(abs(df["log2FC"])) +
        normalize(1 - df["adj_pvalue"]) +
        normalize(df["expression_variance"])
    )

    return df


# -------------------------
# Monte Carlo Stability
# -------------------------
def monte_carlo_stability(df, simulations=300):

    stability_scores = []

    for _, row in df.iterrows():
        stable = 0

        for _ in range(simulations):
            fc = row["log2FC"] + np.random.normal(0, 0.2)
            pv = row["adj_pvalue"] + np.random.normal(0, 0.01)

            if abs(fc) > 1 and pv < 0.05:
                stable += 1

        stability_scores.append(stable / simulations)

    df["stability_score"] = stability_scores
    return df


# -------------------------
# Fuzzy Logic Model
# -------------------------
def fuzzy_model(df):

    def fuzzy(x):
        low = norm.pdf(x, 0, 0.5)
        mid = norm.pdf(x, 1, 0.5)
        high = norm.pdf(x, 2, 0.5)
        return low + mid + high

    df["fuzzy_score"] = df["log2FC"].apply(lambda x: fuzzy(abs(x)))
    return df


# -------------------------
# Bayesian Probability Model
# -------------------------
def bayesian_probability(df):

    prior = 0.5
    likelihood = 1 - df["adj_pvalue"]

    posterior = (likelihood * prior) / (
        likelihood * prior + (1 - likelihood) * (1 - prior)
    )

    df["bayesian_prob"] = posterior
    return df


# -------------------------
# Network Influence Score
# -------------------------
def network_influence_score(ppi_edges, deg_df):

    G = nx.from_pandas_edgelist(ppi_edges, "source", "target")

    degree = nx.degree_centrality(G)

    deg_df["network_score"] = deg_df["gene"].map(degree).fillna(0)

    return deg_df


# -------------------------
# Combined BioMathematical Score
# -------------------------
def combined_biomath_score(df):

    score_cols = [
        "GA_score",
        "stability_score",
        "fuzzy_score",
        "bayesian_prob",
        "network_score"
    ]

    df["biomath_score"] = df[score_cols].mean(axis=1)

    return df


# -------------------------
# MASTER PIPELINE
# -------------------------
def run_biomath_layer(deg_df, ppi_edges):

    deg_df = genetic_algorithm_score(deg_df)
    deg_df = monte_carlo_stability(deg_df)
    deg_df = fuzzy_model(deg_df)
    deg_df = bayesian_probability(deg_df)
    deg_df = network_influence_score(ppi_edges, deg_df)
    deg_df = combined_biomath_score(deg_df)

    return deg_df
