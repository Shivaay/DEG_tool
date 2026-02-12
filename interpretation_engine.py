"""
Deterministic Transcriptomic Interpretation Engine
Author: Phoenix BioInfoSys
"""

from dataclasses import dataclass
import pandas as pd


# =====================================================
# DATA CONTAINER
# =====================================================

@dataclass
class InterpretationInput:
    deg_table: pd.DataFrame
    up_genes: pd.DataFrame
    down_genes: pd.DataFrame
    hub_genes: pd.DataFrame | None
    enrichment_up: dict | None
    enrichment_down: dict | None
    mirna_df: pd.DataFrame | None
    tf_df: pd.DataFrame | None
    bayesian_confidence: float | None
    adaptive_threshold: float | None


# =====================================================
# ENGINE
# =====================================================

class InterpretationEngine:

    def __init__(self, data: InterpretationInput):
        self.data = data

    # -----------------------------
    # Evidence aggregation
    # -----------------------------
    def collect_evidence(self):

        return {
            "n_deg": len(self.data.deg_table),
            "n_up": len(self.data.up_genes),
            "n_down": len(self.data.down_genes),
            "hub_genes": self.data.hub_genes,
            "mirna": self.data.mirna_df,
            "tf": self.data.tf_df,
            "confidence": self.data.bayesian_confidence,
            "adaptive": self.data.adaptive_threshold,
        }

    # -----------------------------
    # Biological inference rules
    # -----------------------------
    def infer_biology(self, evidence):

        insights = []

        if evidence["n_up"] > evidence["n_down"]:
            insights.append(
                "Transcriptome demonstrates dominant gene activation pattern."
            )
        else:
            insights.append(
                "Transcriptome demonstrates dominant gene suppression pattern."
            )

        if evidence["hub_genes"] is not None:
            insights.append(
                "Protein interaction analysis identifies hub genes indicating potential regulatory control points."
            )

        if evidence["mirna"] is not None and not evidence["mirna"].empty:
            insights.append(
                "miRNA interaction mapping suggests post-transcriptional regulatory modulation."
            )

        if evidence["tf"] is not None and not evidence["tf"].empty:
            insights.append(
                "Transcription factor interactions indicate upstream transcriptional control mechanisms."
            )

        if evidence["confidence"] is not None:
            insights.append(
                f"Bayesian confidence score ({evidence['confidence']:.2f}) supports DEG stability."
            )

        return insights

    # -----------------------------
    # Clinical interpretation text
    # -----------------------------
    def generate_report(self):

        evidence = self.collect_evidence()
        insights = self.infer_biology(evidence)

        report = f"""
MOLECULAR INTERPRETATION REPORT

Summary:
{evidence['n_deg']} DEGs detected
{evidence['n_up']} upregulated
{evidence['n_down']} downregulated

Key Findings:
- {" ".join(insights)}

Biological Interpretation:
Results suggest coordinated pathway-level molecular shifts rather than isolated gene effects.

Hypothesis:
Identified hub genes and enriched pathways may contribute to disease biology and require experimental validation.

Limitations:
Computational inference only. Requires laboratory confirmation.
"""

        return report
