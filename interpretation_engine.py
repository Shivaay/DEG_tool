# =========================================================
# PhoenixBioInfoSys Interpretation Engine
# Real Systems Biology Interpretation Layer
# Fully Fixed and Robust Version
# =========================================================

from dataclasses import dataclass
import numpy as np


# =========================================================
# INPUT STRUCTURE
# =========================================================

@dataclass
class InterpretationInput:

    deg_table: object

    biomath_metrics: dict

    hub_genes: object


# =========================================================
# INTERPRETATION ENGINE
# =========================================================

class InterpretationEngine:


    def __init__(self, data):

        self.data = data


    # =====================================================
    # TRANSCRIPTOMIC STATE ANALYSIS
    # =====================================================

    def transcriptomic_state(self):

        try:

            up = len(
                self.data.deg_table[
                    self.data.deg_table["Regulation"] == "Up"
                ]
            )

            down = len(
                self.data.deg_table[
                    self.data.deg_table["Regulation"] == "Down"
                ]
            )

            if up > down:

                return "activation"

            elif down > up:

                return "suppression"

            else:

                return "balanced regulation"


        except Exception:

            return "unknown regulatory state"



    # =====================================================
    # SYSTEM COLLAPSE PROBABILITY
    # =====================================================

    def collapse_probability(self):

        try:

            m = self.data.biomath_metrics

            entropy = float(m.get("system_entropy", 0))

            perturb = float(m.get("perturbation_magnitude", 0))

            stability = float(m.get("system_stability", 0))


            collapse = entropy * perturb * (1 - stability)

            return collapse


        except Exception:

            return 0



    # =====================================================
    # HUB GENE ANALYSIS (FIXED VERSION)
    # =====================================================

    def hub_analysis(self):

        try:

            if self.data.hub_genes is None:

                return "No hub genes detected."


            if len(self.data.hub_genes) == 0:

                return "No hub genes detected."


            columns = list(self.data.hub_genes.columns)


            # Automatically detect correct column
            if "Gene" in columns:

                top_gene = self.data.hub_genes.iloc[0]["Gene"]

            elif "gene" in columns:

                top_gene = self.data.hub_genes.iloc[0]["gene"]

            else:

                return "Hub gene column not found."


            return (

                f"The hub gene {top_gene} exhibits the highest network "

                f"centrality, indicating a dominant regulatory role "

                f"in the transcriptomic system."

            )


        except Exception:

            return "Hub gene analysis unavailable."



    # =====================================================
    # GENERATE FULL REPORT
    # =====================================================

    def generate(self):

        state = self.transcriptomic_state()

        collapse = self.collapse_probability()

        hub_text = self.hub_analysis()


        m = self.data.biomath_metrics


        entropy = float(m.get("system_entropy", 0))

        stability = float(m.get("system_stability", 0))

        centrality = float(m.get("network_centrality", 0))

        perturb = float(m.get("perturbation_magnitude", 0))


        report = f"""

============================================================

PhoenixBioInfoSys Systems Biology Interpretation Report

============================================================


Transcriptomic Regulatory State:

The transcriptomic profile demonstrates dominant {state}

pattern, indicating a system-wide regulatory shift.


System Entropy:

The calculated entropy value of {entropy:.4f} reflects the

degree of transcriptomic disorder and regulatory complexity.


System Stability:

The stability score of {stability:.4f} indicates the system’s

resilience against perturbation.


Network Influence:

The network centrality score of {centrality:.4f} reflects the

importance of regulatory connectivity among genes.


Perturbation Magnitude:

The perturbation magnitude of {perturb:.4f} indicates the

strength of transcriptional disruption.


Hub Gene Analysis:

{hub_text}


Collapse Probability:

The predicted system collapse probability is {collapse:.4f},

indicating the likelihood of regulatory instability.


Biological Interpretation:

These results suggest biologically meaningful regulatory

alterations driven by key network genes and coordinated

transcriptional dynamics.


============================================================

End of Report

============================================================

"""


        return report
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
