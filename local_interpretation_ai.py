import pandas as pd
import numpy as np
from collections import Counter

# =========================================================
# EVIDENCE SCORING MODULE
# =========================================================

class EvidenceIntegrator:

    def __init__(self):
        self.weights = {
            "deg_presence": 2,
            "hub_gene": 3,
            "mirna_regulation": 1.5,
            "tf_regulation": 1.5,
            "enrichment_presence": 2
        }

    def score_genes(
        self,
        up_genes,
        down_genes,
        hub_genes,
        mirna_df,
        tf_df,
        enrichment_terms
    ):

        gene_scores = {}

        all_genes = list(set(up_genes + down_genes))

        for gene in all_genes:

            score = 0
            evidence = []

            # DEG Evidence
            score += self.weights["deg_presence"]
            evidence.append("Differential expression")

            # Hub Gene
            if gene in hub_genes:
                score += self.weights["hub_gene"]
                evidence.append("Network hub gene")

            # miRNA
            if mirna_df is not None and gene in mirna_df["Target"].values:
                score += self.weights["mirna_regulation"]
                evidence.append("miRNA regulation")

            # TF
            if tf_df is not None and gene in tf_df["Target"].values:
                score += self.weights["tf_regulation"]
                evidence.append("Transcription factor regulation")

            # Enrichment association
            if enrichment_terms:
                score += self.weights["enrichment_presence"]
                evidence.append("Functional enrichment association")

            gene_scores[gene] = {
                "score": score,
                "evidence": evidence
            }

        return gene_scores


# =========================================================
# BIOLOGICAL THEME DETECTION
# =========================================================

class BiologicalThemeExtractor:

    def extract_themes(self, enrichment_dict):

        themes = []

        for key, df in enrichment_dict.items():

            if df is None or df.empty:
                continue

            terms = df["Term"].head(5).tolist()
            themes.extend(terms)

        return themes


# =========================================================
# INTERPRETATION TEXT GENERATOR
# =========================================================

class ScientificInterpreter:

    def generate_report(
        self,
        up_genes,
        down_genes,
        gene_scores,
        themes,
        hub_genes
    ):

        total_genes = len(up_genes) + len(down_genes)

        # --------------------------------------------------
        # Section 1 Summary
        # --------------------------------------------------

        summary = (
            f"A total of {total_genes} differentially expressed genes "
            f"were identified, including {len(up_genes)} upregulated "
            f"and {len(down_genes)} downregulated genes. "
        )

        if hub_genes:
            summary += (
                f"Network analysis highlighted {len(hub_genes)} hub genes "
                f"suggesting potential topological importance."
            )

        # --------------------------------------------------
        # Section 2 Multi-layer Evidence
        # --------------------------------------------------

        sorted_genes = sorted(
            gene_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )

        top_genes = [g[0] for g in sorted_genes[:5]]

        multilayer = (
            "Multiple analytical layers consistently highlight "
            f"{', '.join(top_genes)}. "
            "These genes demonstrate convergence across expression, "
            "network topology and regulatory analysis."
        )

        # --------------------------------------------------
        # Section 3 Regulatory Mechanisms
        # --------------------------------------------------

        regulatory = (
            "Regulatory analysis suggests that transcriptional and "
            "post-transcriptional mechanisms may contribute to "
            "observed expression variability."
        )

        # --------------------------------------------------
        # Section 4 Hypothesis
        # --------------------------------------------------

        hypothesis = (
            f"Perturbation of genes such as {', '.join(top_genes[:3])} "
            "may influence biological processes including "
            f"{', '.join(themes[:3])}. "
            "These relationships warrant further experimental validation."
        )

        # --------------------------------------------------
        # Section 5 Limitations
        # --------------------------------------------------

        limitations = (
            "These findings are derived from computational integration "
            "of transcriptomic evidence and should be interpreted as "
            "hypothesis-generating rather than definitive conclusions."
        )

        # --------------------------------------------------
        return "\n\n".join([
            "Summary of Key Findings:\n" + summary,
            "Multi-Layer Evidence Integration:\n" + multilayer,
            "Candidate Regulatory Mechanisms:\n" + regulatory,
            "Hypotheses for Experimental Validation:\n" + hypothesis,
            "Limitations and Analytical Context:\n" + limitations
        ])


# =========================================================
# MASTER INTERPRETATION PIPELINE
# =========================================================

def run_local_interpretation(
    up_genes,
    down_genes,
    enrichment_dict,
    hub_genes,
    mirna_df=None,
    tf_df=None
):

    integrator = EvidenceIntegrator()
    theme_extractor = BiologicalThemeExtractor()
    interpreter = ScientificInterpreter()

    gene_scores = integrator.score_genes(
        up_genes,
        down_genes,
        hub_genes,
        mirna_df,
        tf_df,
        enrichment_dict
    )

    themes = theme_extractor.extract_themes(enrichment_dict)

    report = interpreter.generate_report(
        up_genes,
        down_genes,
        gene_scores,
        themes,
        hub_genes
    )

    return report
