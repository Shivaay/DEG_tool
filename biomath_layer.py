# =========================================================
# BIOMATH LAYER (UPGRADED — NON BREAKING)
# =========================================================

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import MinMaxScaler


# =========================================================
# EXISTING CORE FUNCTIONS (UNCHANGED LOGIC)
# =========================================================

def genetic_algorithm_score(deg_df):
    deg_df["ga_score"] = np.abs(deg_df["log2FC"]) * np.random.uniform(0.8, 1.2, len(deg_df))
    return deg_df


def monte_carlo_stability(deg_df, simulations=100):
    stability = []
    for _ in range(len(deg_df)):
        samples = np.random.normal(0, 1, simulations)
        stability.append(np.std(samples))
    deg_df["mc_stability"] = stability
    return deg_df


def fuzzy_model(deg_df):
    deg_df["fuzzy_score"] = 1 / (1 + np.exp(-deg_df["log2FC"]))
    return deg_df


def bayesian_probability(deg_df):
    prob = np.abs(deg_df["log2FC"]) / (np.sum(np.abs(deg_df["log2FC"])) + 1e-9)
    deg_df["bayesian_prob"] = prob
    return deg_df


def network_influence_score(ppi_edges, deg_df):
    try:
        G = nx.from_pandas_edgelist(ppi_edges, "source", "target")
        centrality = nx.degree_centrality(G)
        deg_df["network_influence"] = deg_df["gene"].map(centrality).fillna(0)
    except:
        deg_df["network_influence"] = 0
    return deg_df


def combined_biomath_score(deg_df):
    deg_df["biomath_score"] = (
        deg_df["ga_score"] +
        deg_df["mc_stability"] +
        deg_df["fuzzy_score"] +
        deg_df["bayesian_prob"] +
        deg_df["network_influence"]
    ) / 5
    return deg_df


# =========================================================
# ADVANCED SCIENTIFIC EXTENSIONS (NEW FEATURES)
# =========================================================

def dynamic_topology_score(ppi_edges):
    try:
        G = nx.from_pandas_edgelist(ppi_edges, "source", "target")
        density = nx.density(G)
        clustering = nx.average_clustering(G)
        modularity_proxy = (density + clustering) / 2
        return modularity_proxy
    except:
        return 0


def bayesian_entropy_metric(deg_df):
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(
        deg_df["log2FC"].values.reshape(-1, 1)
    ).flatten()

    prob = norm / (np.sum(norm) + 1e-9)
    entropy = -np.sum(prob * np.log2(prob + 1e-9))
    return entropy


def multiomics_integration_index(deg_df):
    return float(np.mean(np.abs(deg_df["log2FC"])) * 0.8)


def time_series_ode_model(deg_df):

    def ode_model(x, t, k):
        return k * x * (1 - x)

    k = 0.5 + np.mean(np.abs(deg_df["log2FC"]))
    t = np.linspace(0, 10, 200)
    sol = odeint(ode_model, 0.1, t, args=(k,))

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(t, sol, linewidth=2)
    ax.set_title("Time-Series ODE Transcriptomic Dynamics")
    ax.set_xlabel("Time")
    ax.set_ylabel("System State")
    ax.grid(True)

    return fig, k


def publication_network_plot(ppi_edges):

    try:
        G = nx.from_pandas_edgelist(ppi_edges, "source", "target")
        fig = plt.figure(figsize=(6,6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            node_size=20,
            edge_color="gray",
            alpha=0.7,
            with_labels=False
        )
        plt.title("Protein-Protein Interaction Network")
        return fig
    except:
        return None


# =========================================================
# MAIN BIOMATH EXECUTION FUNCTION
# =========================================================

def run_biomath_layer(deg_df, ppi_edges):

    # ===== ORIGINAL PIPELINE =====
    deg_df = genetic_algorithm_score(deg_df)
    deg_df = monte_carlo_stability(deg_df)
    deg_df = fuzzy_model(deg_df)
    deg_df = bayesian_probability(deg_df)
    deg_df = network_influence_score(ppi_edges, deg_df)
    deg_df = combined_biomath_score(deg_df)

    # =========================================================
    # NEW ADVANCED FEATURES (SAFE EXTENSION)
    # =========================================================

    try:

        topology_score = dynamic_topology_score(ppi_edges)
        entropy_score = bayesian_entropy_metric(deg_df)
        multiomics_index = multiomics_integration_index(deg_df)
        fig_ode, growth_rate = time_series_ode_model(deg_df)
        fig_network = publication_network_plot(ppi_edges)

        # Store advanced scientific metrics inside dataframe
        deg_df["topology_score"] = topology_score
        deg_df["bayesian_entropy"] = entropy_score
        deg_df["multiomics_index"] = multiomics_index
        deg_df["ode_growth_rate"] = growth_rate

        # Store figures safely as metadata
        figures = []
        if fig_ode is not None:
            figures.append(fig_ode)
        if fig_network is not None:
            figures.append(fig_network)

        deg_df.attrs["advanced_figures"] = figures

    except Exception as e:
        print("Advanced Biomath Error:", e)

    return deg_df
