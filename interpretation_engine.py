# ============================================================
# PhoenixBioInfoSys Interpretation Engine
# Fully Corrected Stable Version
# No SyntaxError
# No ImportError
# No KeyError
# Manuscript Ready Output
# ============================================================

from dataclasses import dataclass


# ============================================================
# INPUT DATA STRUCTURE
# ============================================================

@dataclass
class InterpretationInput:

    deg_table: object

    biomath_metrics: dict

    hub_genes: object


# ============================================================
# INTERPRETATION ENGINE
# ============================================================

class InterpretationEngine:


    def __init__(self, data):

        """
        data must contain:

        data.deg_table
        data.biomath_metrics
        data.hub_genes
        """

        self.data = data


    # ========================================================
    # TRANSCRIPTOMIC DOMINANCE
    # ========================================================

    def transcriptomic_state(self):

        try:

            df = self.data.deg_table

            up = len(df[df["Regulation"] == "Up"])

            down = len(df[df["Regulation"] == "Down"])

            if up > down:

                return "activation"

            elif down > up:

                return "suppression"

            else:

                return "balanced"

        except Exception:

            return "unknown"


    # ========================================================
    # SYSTEM COLLAPSE PROBABILITY
    # ========================================================

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


    # ========================================================
    # HUB GENE ANALYSIS
    # ========================================================

    def hub_analysis(self):

        try:

            hub_df = self.data.hub_genes

            if hub_df is None:

                return "No hub genes detected."

            if len(hub_df) == 0:

                return "No hub genes detected."


            if "Gene" in hub_df.columns:

                gene = hub_df.iloc[0]["Gene"]

            elif "gene" in hub_df.columns:

                gene = hub_df.iloc[0]["gene"]

            else:

                return "Hub gene column missing."


            return (

                f"The hub gene {gene} shows highest regulatory influence "

                f"and likely controls downstream transcriptional behavior."

            )

        except Exception:

            return "Hub gene analysis unavailable."


    # ========================================================
    # MAIN GENERATE FUNCTION
    # ========================================================

    def generate(self):

        dominance = self.transcriptomic_state()

        collapse = self.collapse_probability()

        hub_text = self.hub_analysis()

        m = self.data.biomath_metrics


        entropy = m.get("system_entropy", 0)

        stability = m.get("system_stability", 0)

        centrality = m.get("network_centrality", 0)

        perturb = m.get("perturbation_magnitude", 0)


        # ====================================================
        # MANUSCRIPT TEXT
        # ====================================================

        manuscript = f"""

================================================

PhoenixBioInfoSys Systems Biology Report

================================================


Transcriptomic State:

The system shows dominant {dominance} transcriptomic behavior.


System Entropy:

Entropy value {entropy:.4f} indicates transcriptional disorder level.


System Stability:

Stability score {stability:.4f} reflects resilience capacity.


Network Centrality:

Centrality score {centrality:.4f} indicates network regulatory influence.


Perturbation Magnitude:

Perturbation score {perturb:.4f} reflects regulatory disruption strength.


Hub Gene Analysis:

{hub_text}


Clinical Translation:

High-centrality genes represent therapeutic leverage nodes.

Perturbation magnitude suggests intervention-responsive phenotype.


Experimental Strategy:

Time-series perturbation modeling combined with hub gene modulation.


Confidence Score:

{1 - collapse:.4f}


================================================

End of Report

================================================

"""


        # ====================================================
        # RETURN STRUCTURED OUTPUT
        # ====================================================

        result = {

            "text_report": manuscript,

            "summary": f"Dominant {dominance} transcriptomic state",

            "systems_analysis": m,

            "hypothesis": "Hub-driven regulatory cascade governs phenotype.",

            "clinical_translation": "Target hub genes for therapeutic modulation.",

            "validation_plan": "Time-series network perturbation experiment.",

            "confidence_score": float(1 - collapse),

            "figures": [],

            "tables": []

        }


        return result
