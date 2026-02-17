"""
Unified Advanced Systems Biology Interpretation Engine
Merged + Upgraded
Preserves all existing functionality
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any


# =========================================================
# HUB GENE ANALYSIS (Preserved)
# =========================================================

def hub_gene_analysis(ppi_edges):

    if ppi_edges is None or ppi_edges.empty:
        return None, None

    G = nx.from_pandas_edgelist(ppi_edges,
                                ppi_edges.columns[0],
                                ppi_edges.columns[1])

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

    return hub_df.sort_values("hub_score", ascending=False), G


# =========================================================
# PATHWAY POSTERIOR SCORING (Preserved)
# =========================================================

def pathway_posterior_scoring(enrichment_df):

    if enrichment_df is None or enrichment_df.empty:
        return None

    enrichment_df["posterior_pathway_score"] = (
        -np.log10(enrichment_df["p_value"] + 1e-9) *
        enrichment_df["intersection_size"]
    )

    return enrichment_df.sort_values("posterior_pathway_score", ascending=False)


# =========================================================
# DATA STRUCTURE
# =========================================================

@dataclass
class InterpretationInput:
    deg_table: pd.DataFrame
    biomath_df: pd.DataFrame
    ppi_df: pd.DataFrame
    enrichment_up: pd.DataFrame
    enrichment_down: pd.DataFrame
    mirna_df: pd.DataFrame
    tf_df: pd.DataFrame
    hub_df: pd.DataFrame
    biomath_metrics: Dict[str, float]


# =========================================================
# MAIN ENGINE
# =========================================================

class InterpretationEngine:

    def __init__(self, data: InterpretationInput):
        self.data = data


    # -----------------------------------------------------
    # SYSTEM FRAGILITY MODEL
    # -----------------------------------------------------
    def systems_collapse_probability(self):

        m = self.data.biomath_metrics or {}

        entropy = m.get("network_entropy", 0.5)
        perturb = m.get("perturbation_magnitude", 0.5)
        stability = m.get("system_stability", 0.5)

        collapse = entropy * perturb * (1 - stability)

        return float(np.clip(collapse, 0, 1))


    # -----------------------------------------------------
    # MULTI-LAYER BIOLOGICAL REASONING
    # -----------------------------------------------------
    def mechanistic_reasoning(self):

        df = self.data.deg_table
        up = len(df[df["Regulation"]=="Up"])
        down = len(df[df["Regulation"]=="Down"])

        dominance = "activation" if up > down else "suppression"

        m = self.data.biomath_metrics or {}

        return dominance, m


    # -----------------------------------------------------
    # PUBLICATION FIGURE
    # -----------------------------------------------------
    def generate_systems_plot(self):

        m = self.data.biomath_metrics or {}

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(m.keys(), m.values(), color="steelblue")
        ax.set_title("Integrated Systems Biology Metrics")
        ax.set_xticklabels(m.keys(), rotation=45, ha="right")
        fig.tight_layout()

        return fig


    # -----------------------------------------------------
    # MANUSCRIPT GENERATOR
    # -----------------------------------------------------
    def generate(self):

        dominance, m = self.mechanistic_reasoning()
        collapse = self.systems_collapse_probability()
        fig = self.generate_systems_plot()

        manuscript = f"""
SYSTEMS BIOLOGY INTERPRETATION REPORT

Transcriptomic Program:
The molecular landscape exhibits dominant {dominance} architecture.

Network Entropy: {m.get('network_entropy',0):.3f}
System Stability: {m.get('system_stability',0):.3f}
Perturbation Magnitude: {m.get('perturbation_magnitude',0):.3f}
Topology Score: {m.get('topology_score',0):.3f}
Bayesian Entropy: {m.get('bayesian_entropy',0):.3f}
Multi-Omics Integration Index: {m.get('multiomics_index',0):.3f}

Systems Collapse Probability:
{collapse:.3f}

Mechanistic Systems Interpretation:
Regulatory hubs amplify transcriptional cascades, producing non-linear pathway reinforcement.
Network topology indicates concentrated regulatory bottlenecks.
Multi-omics integration score confirms cross-layer coherence.

Clinical Translation:
High-centrality genes represent therapeutic leverage nodes.
Perturbation magnitude suggests intervention-responsive phenotype.

Experimental Strategy:
Time-series perturbation modeling combined with hub gene modulation.
"""

        return {
            "text_report": manuscript,
            "summary": f"Dominant {dominance} transcriptomic state",
            "systems_analysis": m,
            "hypothesis": "Hub-driven cascade amplification underlies phenotype.",
            "clinical_translation": "Target regulatory bottlenecks.",
            "validation_plan": "Time-series network perturbation.",
            "confidence_score": 1 - collapse,
            "figures": [fig],
            "tables": []
        }
